"""
Preprocessing script for the Vision4Action landing task data that are contained in the .nix files.

The script saves an MNE.Epoch object with neural (or behavioral) data aligned in a 
behavioral trigger. 

"""

import os
import neo
import mne
import utils
import pickle
import numpy as np
import xarray as xr
import preproc_funcs
from frites.io import logger
from pandas import DataFrame
from brainets.spectral import mt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from mne.time_frequency import (tfr_morlet, AverageTFR)

SFREQ = 1000 # Hz

# -----------------------------------------------------------------------
# -------------------------- Helper functions ---------------------------
# -----------------------------------------------------------------------

def _load_mua_pkl(v4a_dir, filename):
    """Load the mua pickle file."""
    logger.info(f"Loading the MUA data file {filename}...")
    with open(os.path.join(v4a_dir, filename),'rb') as handle:
        analogSignal = pickle.load(handle)
    logger.info("...done.")
    return analogSignal


def _parse_trials(segBehavior, targetID):
    """
    Extract trial timing, target IDs, and landing-sequence event codes.
    Returns a dict with all trial-level arrays needed.
    """
    trials = segBehavior.filter(name='All Trial Type Presentations')[0]
    attempts = segBehavior.filter(name='All Trials')[0]

    # Get the indices of trials sorted based on their landing sequence
    landing_sequence = trials.array_annotations['landing_sequence']
    n_ls = landing_sequence.max()
    ls_trials = [np.where(landing_sequence == ls+1)[0].tolist()
                for ls in range(n_ls)]

    # Get trial data: labels, times, start, end and number
    events = segBehavior.filter(name='DecodedEvents')[0]
    event_labels = events.labels
    event_times = np.array(np.round(events.times*SFREQ), dtype=int)
    tStart_ind = np.where(event_labels == 'trial_start')[0]
    tEnd_ind = np.where(event_labels == 'trial_end')[0]
    ntrials = tEnd_ind.shape[0]

    # Get target information
    currentTarget = attempts.array_annotations[f'target_0{targetID}']
    previousTarget = attempts.array_annotations[f'target_0{targetID-1}']
    successful_ind = np.where(attempts.array_annotations['successful'] == True)[0]
    currentTargetID = currentTarget[successful_ind]
    previousTargetID = previousTarget[successful_ind]
    attempts_intrials = trials.array_annotations['trials']

    # initialize
    target_onsets = np.zeros(ntrials, dtype=int)
    target_reached = np.zeros(ntrials, dtype=int)
    lstg = np.zeros(ntrials, dtype=int)
    tg_repeated = np.zeros(ntrials, dtype=bool)

    for trial, (t1, t2) in enumerate(zip(tStart_ind, tEnd_ind)):
        trial_times = event_times[t1:t2+1]
        trial_labels = event_labels[t1:t2+1]
        indON = np.where(trial_labels == f'target_0{targetID}_on')[0]
        indR = np.where(trial_labels == f'target_0{targetID}_reached')[0]
        if indON.size == 0 or indR.size == 0:
            return None

        # Select the last events (that always come from the successful trial)
        target_onsets[trial] = trial_times[indON[-1]]
        target_reached[trial] = trial_times[indR[-1]]

        # In case of multiple attempts, target 2 will be anticipated in the 
        # successful trial because of repetition, unless there was only one
        # 'target_02_on' event because of landing error in the central target.
        tg_repeated[trial] = (attempts_intrials[trial] > 1) and (indON.size > 1)

        # Generate an event code for each target onset based on the landing 
        # sequence the trial belongs to
        for ls, trial_group in enumerate(ls_trials):
            # Find the trial group (LS) on which the current trial belongs to
            if trial in trial_group:
                # Save an integer ij: format LS_i Target_j for the event codes
                lstg[trial] = (ls+1)*10 + targetID

    event_ids = {f'LS {i+1} Target {targetID}': int(f'{i+1}{targetID}')
                for i in range(n_ls)}

    return dict(ntrials = ntrials,
                target_onsets = target_onsets,
                target_reached = target_reached,
                lstg = lstg,
                tg_repeated = tg_repeated,
                attempts_intrials = attempts_intrials,
                currentTargetID = currentTargetID,
                previousTargetID = previousTargetID,
                event_ids = event_ids,
                n_ls = n_ls)


def _get_hand_onsets(target_onsets, target_reached, segBehavior):
    """Find movement onset for each trial by searching between target onset and reach."""
    hand_movs = segBehavior.filter(name='Hand Movements')[0]
    allMovOnsets = np.array(np.round(hand_movs.times * SFREQ), dtype=int)

    indices = []
    for t_on, t_r in zip(target_onsets, target_reached):
        buffer = 50  # ms to search before target onset if no onset found
        ind = np.where((allMovOnsets > t_on) & (allMovOnsets < t_r))[0]
        while ind.size == 0:
            ind = np.where((allMovOnsets > t_on - buffer) & (allMovOnsets < t_r))[0]
            buffer += 50
        indices.append(ind[0])

    return allMovOnsets[indices]


def _detect_hand_onsets(hand_x, hand_y, target_onsets, target_reached, targetID):
    """Detect hand onset per trial from peak hand velocity."""
    ntrials = target_onsets.shape[0]
    hand_onsets = np.zeros(ntrials, dtype=int)
    buffer = 200  # dwelling time (in ms) before initiating a hand movement
    coef = 0.05 # threshold coefficient 5% of v_max - v_min

    for tr, (t_on, t_r) in enumerate(zip(target_onsets, target_reached)):

        # Add buffer to allow anticipatory movement onsets before target onset
        w_start = t_on - buffer 
        w_end = t_r + 1
        x = hand_x[w_start : w_end]
        y = hand_y[w_start : w_end]

        # Compute 2d-velocity from x,y hand positions
        vx = savgol_filter(np.diff(x), 51, 2) * SFREQ
        vy = savgol_filter(np.diff(y), 51, 2) * SFREQ
        v = np.sqrt(vx**2 + vy**2)

        # Find the peak between target onset until target reached
        peak_idx = np.argmax(v[buffer:]) + buffer

        # Find where the acceleration changes sign before the peak
        v_prepeak = v[:peak_idx]
        accel = savgol_filter(np.diff(v_prepeak), 151, 2)
        local_min_idxs = np.where((accel[:-1] < 0) & (accel[1:] >= 0))[0] + 1

        if len(local_min_idxs) == 0: 
            trough_idx = np.argmin(v_prepeak)
        else: 
            trough_idx = local_min_idxs[-1]

        # In case that the last local minimum is larger than 50% of the peak
        # which means that is not a real minimum (there is hand velocity before)
        # select the second to last minimum. Continue until you find a minimum
        # that satisfies the condition.
        counter = -2
        while v[trough_idx] > 0.3 * v[peak_idx]:
            counter -= 1
            try:
                trough_idx = local_min_idxs[counter]
            except IndexError: 
                trough_idx = np.argmin(v_prepeak)
                break

        # Find all points below a velocity threshold between trough and peak
        v_search = v[trough_idx:peak_idx]
        v_thres = coef * (v[peak_idx] - v[trough_idx]) + v[trough_idx]
        below_thresh = np.where(v_search < v_thres)[0]

        # Movement onset is the last point below threshold (or the trough itself)
        if len(below_thresh) == 0: 
            onset_idx = trough_idx
        else: 
            onset_idx = trough_idx + below_thresh[-1] + 1
        
        hand_onsets[tr] = w_start + onset_idx

    return hand_onsets


def _detect_saccade_onsets(eye_x, eye_y, target_onsets, target_reached, v_thres):
    """Detect saccade onset per trial from peak eye velocity."""
    ntrials = target_onsets.shape[0]
    saccade_onsets = np.zeros(ntrials, dtype=int)
    buffer = 200  # dwelling time (in ms) before initiating a hand movement
    dt = 25 # typical durations of saccade onset to peak (in ms)

    for tr, (t_on, t_r) in enumerate(zip(target_onsets, target_reached)):

        w_start = t_on - buffer # to allow anticipatory saccades before target onset
        w_end = t_r + 1
        x = eye_x[w_start : w_end]
        y = eye_y[w_start : w_end]
        vx = savgol_filter(np.diff(x), 11, 2) * SFREQ
        vy = savgol_filter(np.diff(y), 11, 2) * SFREQ
        v = np.sqrt(vx**2 + vy**2)

        v_binary = (v > v_thres).astype(int)
        thres_crossings = np.where(np.diff(v_binary) == 1)[0]
        peak_idx = np.argmax(v)

        # If there are no threshold crossings, assign as saccade onset
        # the peak velocity minus a typical half saccade duration
        if thres_crossings.size == 0:
            saccade_onsets[tr] = w_start + peak_idx - dt
            print(f"Warning: No threshold crossing in trial {tr}")
            continue

        # Find the difference between the index of the peak and the indices  
        # of all threshold crossings. The smallest positive difference reflects  
        # the main threshold crossing (main saccade onset) because the peak 
        # comes soon after the onset.
        peak_onset_diff = peak_idx - thres_crossings
        # Replace negatives with infinite, to not be picked up as minima
        peak_onset_diff_pos = np.where(peak_onset_diff >= 0, peak_onset_diff, np.inf)

        # If there are no positives, assign as saccade onset
        # the peak velocity minus a typical half saccade duration
        if np.isinf(peak_onset_diff_pos).all():
            saccade_onsets[tr] = w_start + peak_idx - dt
            print(f"Warning: No onset followed by the peak in trial {tr}")
            continue

        # Find the index of the smallest positive difference (main saccade onset)
        main_idx = np.argmin(peak_onset_diff_pos)
        # Find the index of the main saccade onset
        s_onset_idx = thres_crossings[main_idx]
        # Save the trial's saccade onset
        saccade_onsets[tr] = w_start + s_onset_idx

    return saccade_onsets


def _build_mne_events(event_onsets, lstg):
    """Build the (n, 3) MNE events array."""
    events_zero = np.zeros(len(event_onsets), dtype=int)
    return np.column_stack((event_onsets, events_zero, lstg))


def _zscore_signal(signal, axis):
    """Z-score each channel (row) across all timepoints."""
    mean = signal.mean(axis, keepdims=True)
    std  = signal.std(axis,  keepdims=True)
    return (signal - mean) / std


def _epoch_signal(signal, channels, event_onsets, w1, w2):
    """
    Align the signal in a window around an event.
    Returns (ntrials, nchannels, w1+w2).
    """
    ntrials = len(event_onsets)
    nchannels = len(channels)
    ntimes = w1 + w2
    signal_aligned = np.zeros((ntrials, nchannels, ntimes))
    for e, ev_onset in enumerate(event_onsets):
        signal_aligned[e] = signal[channels, ev_onset-w1 : ev_onset+w2]
    return signal_aligned


def _load_area_channels(session_name, visual_nchannels):
    """
    Build channel index list and MNE info from electrode mapping.
    Returns (channels, info_mne).
    """
    subj = 'enya' if session_name[0] == 'y' else 'jazz'
    arrays_maps = preproc_funcs.mapping(subj)
    areas = list(arrays_maps.keys())
    channels, channelNames = [],[]

    # Add 128 (nchannels of the visual array) to the motor array channels 
    # to keep counting. First motor channel will be channel 129.
    for area in areas:
        ff = visual_nchannels if area == 'M1' else 0
        map_ = [idx - 1 + ff for idx in arrays_maps[area] if not np.isnan(idx)]
        names_ = [f'{area}-{idx}' for idx in arrays_maps[area] if not np.isnan(idx)]
        channels.extend(map_)
        channelNames.extend(names_)

    channelTypes = ['seeg'] * len(channelNames)
    info_mne = mne.create_info(ch_names=channelNames, ch_types=channelTypes, sfreq=SFREQ)
    return channels, info_mne



# -----------------------------------------------------------------------
# ----------------------- Epoch creator functions -----------------------
# -----------------------------------------------------------------------

def _create_bhv_epochs(anasig_behav, hand_x, hand_y, eye_x, eye_y,
                       xTarg, yTarg, event_onsets, onset, session_name,
                       targetID, trial_data, t0, k, v4a_dir):
    """Compute all behavioral signals and save a multi-channel MNE Epochs file."""

    eye_vel = preproc_funcs.calculate_vel(anasig_behav, event_onsets, onset, 'Eye')
    hand_vel = preproc_funcs.calculate_vel(anasig_behav, event_onsets, onset, 'Hand')
    hand_eye_dist = preproc_funcs.calculate_hand_eye_dist(
                         hand_x, hand_y, eye_x, eye_y, event_onsets, onset)
    hand_targ_dist = preproc_funcs.calculate_bhv_targ_dist(
                         hand_x, hand_y, xTarg, yTarg,
                         trial_data['currentTargetID'], event_onsets, onset)
    hand_x_pos, hand_y_pos = preproc_funcs.find_xy_positions_intrial(
                                 hand_x, hand_y, event_onsets, onset)
    eye_x_pos,  eye_y_pos = preproc_funcs.find_xy_positions_intrial(
                                 eye_x, eye_y, event_onsets, onset)

    hand_vel_peak = hand_vel.max(axis=1)
    sgmnt_duration = trial_data['target_reached'] - trial_data['target_onsets']
    hand_reaction_times = trial_data['hand_onsets'] - trial_data['target_onsets']
    eye_reaction_times = trial_data['eye_onsets'] - trial_data['target_onsets']
    hand_eye_delay = trial_data['hand_onsets'] - trial_data['eye_onsets']
    target_rank = np.repeat(targetID, trial_data['ntrials'])
    target_angle = preproc_funcs.get_target_angle(xTarg, yTarg,
                                                  trial_data['previousTargetID'],
                                                  trial_data['currentTargetID'])

    channel_data = {'Eye Velocity'        : eye_vel,
                    'Hand Velocity'       : hand_vel,
                    'Hand-Eye Distance'   : hand_eye_dist,
                    'Hand-Target Distance': hand_targ_dist,
                    'Hand X-Position'     : hand_x_pos,
                    'Hand Y-Position'     : hand_y_pos,
                    'Eye X-Position'      : eye_x_pos,
                    'Eye Y-Position'      : eye_y_pos}

    events_mne = _build_mne_events(event_onsets, trial_data['lstg'])
    epoch_list = []
    for ch_name, data in channel_data.items():
        data_mne  = np.expand_dims(data[:, k:-k], axis=1)
        data_info = mne.create_info(ch_names=[ch_name], ch_types=['misc'], sfreq=SFREQ)
        epoch_list.append(mne.EpochsArray(data_mne, data_info, events=events_mne,
                                          event_id=trial_data['event_ids'],
                                          tmin=t0, on_missing='ignore', verbose='ERROR'))

    bhvEpochs = epoch_list[0]
    bhvEpochs.load_data().add_channels([ep.load_data() for ep in epoch_list[1:]])

    metadata_dict = {'Segment Duration'   : sgmnt_duration,
                     'Peak Hand Velocity' : hand_vel_peak,
                     'Hand Reaction Times': hand_reaction_times,
                     'Eye Reaction Times' : eye_reaction_times,
                     'Hand-Eye Delay'     : hand_eye_delay,
                     'Number of Attempts' : trial_data['attempts_intrials'],
                     'Target Repeated'    : trial_data['tg_repeated'],
                     'Target Angle'       : target_angle,
                     'Target Rank'        : target_rank,
                     'currentTargetID'    : trial_data['currentTargetID'],
                     'previousTargetID'   : trial_data['previousTargetID']}
    bhvEpochs.metadata = DataFrame(metadata_dict)

    epoch_file = f'{session_name}_bhv_{onset}{targetID}-epo.fif'
    logger.info(f'Saving behavioral epochs: "{epoch_file}"')
    bhvEpochs.save(os.path.join(v4a_dir, epoch_file), overwrite=True)


def _create_neural_epochs(neuralSignal, content, event_onsets, trial_data,
                          w1, w2, t0_, v4a_dir, session_name,
                          onset, targetID, save=True):
    """Cut the neural signal in epochs, normalize and save.
    If save=False, return the signal epochs for downstream use."""

    if content == 'bnrmua':
        # Convolve with a Gaussian kernel (size: 2*sigma*truncate, 40ms)
        visualSig = gaussian_filter1d(neuralSignal['visual'].astype(float), sigma=5, axis=1)
        motorSig  = gaussian_filter1d(neuralSignal['motor'].astype(float),  sigma=5, axis=1)
    elif content == 'mua':
        visualSig = neuralSignal['visual'] # shape (nchannels, ntimes)
        motorSig  = neuralSignal['motor']
    else: # for hga, beta, tfr : get the LFP
        visualSig = neuralSignal['visual'].rescale('V').magnitude.T    
        motorSig = neuralSignal['motor'].rescale('V').magnitude.T 

    # Align the visual and motor signals, concatenate and normalize
    maxTime = min(visualSig.shape[1], motorSig.shape[1])
    signalAllChans = np.concatenate((visualSig[:, :maxTime], motorSig[:, :maxTime]), axis=0)
    signalNormalized = _zscore_signal(signalAllChans, axis=1)

    # Get channels and mne info
    channels, info_mne = _load_area_channels(session_name, visualSig.shape[0])

    # Align the signal in a window around the event 
    signal = _epoch_signal(signalNormalized, channels, event_onsets, w1, w2)
    events_mne = _build_mne_events(event_onsets, trial_data['lstg'])

    signalEpochs = mne.EpochsArray(signal, info_mne, events=events_mne,
                                   event_id=trial_data['event_ids'],
                                   tmin=t0_, on_missing='ignore', verbose='ERROR')
    if save:
        epoch_file = f'{session_name}_{content}_{onset}{targetID}-epo.fif'
        logger.info(f'Saving neural epochs: "{epoch_file}"')
        signalEpochs.save(os.path.join(v4a_dir, epoch_file), overwrite=True)
    else:
        return signalEpochs


def _create_spike_epochs(neuralSignal, trial_data, event_onsets, w1, w2, 
                         session_name, content, onset, v4a_dir, targetID):
    """
    Build and save Gaussian-convolved firing rate epochs from spike trains.
    """
    sorters = ['Fred','Alexa','Thomas']
    min_spike_count = 100 # minimum spike count for a neuron to be considered
    spike_trains_areas = {}
    units_id, units_type = [],[]

    for area,segment in neuralSignal.items():
        
        sp_times = segment.spiketrains._items
        # valid_trains = [s for s in sp_times 
        #                 if len(s) > min_spike_count and
        #                 s.annotations['sorter'] in sorters and
        #                 s.annotations['unit_type'] != 'noise']
        valid_trains = [s for s in sp_times 
                        if len(s) > min_spike_count]
        if not valid_trains: 
            valid_trains = sp_times
            logger.info(f'Session {session_name} does not include good sorted units.')

        spike_times = [np.array(s.magnitude * SFREQ, dtype=int)
                       for s in valid_trains]
        units_id.extend([s.annotations['implantation_site'] + '_' + \
                        str(int(''.join(filter(str.isdigit, s.annotations['id']))))
                        for s in valid_trains])
        # units_type.extend([s.annotations['unit_type']
        #                   for s in valid_trains])

        n_units = len(spike_times)
        n_times = int(float(segment.t_stop) * SFREQ)
        spike_trains = np.zeros((n_units, n_times))

        for i, spikes in enumerate(spike_times):
            spike_trains[i, spikes] = 1

        spike_trains_areas[area] = spike_trains

    area_names = list(segments.keys())
    maxTime = min(spike_trains_areas[a].shape[1] for a in area_names)
    spike_trains_array = np.concatenate([spike_trains_areas[a][:, :maxTime]
                                        for a in area_names])

    firing_rates = gaussian_filter1d(spike_trains_array, sigma=5, axis=-1)

    # Cut the firing rates in trial epochs
    ntrials = len(event_onsets)
    nunits = firing_rates.shape[0]
    ntimes = w1 + w2
    firing_rate_epochs = np.zeros((ntrials, nunits, ntimes))

    for e, ev_onset in enumerate(event_onsets):
        firing_rate_epochs[e] = firing_rates[:, ev_onset-w1 : ev_onset+w2]

    # Save
    lstg = trial_data['lstg']
    times = np.arange(-w1,w2)
    firing_rates_xr = xr.DataArray(firing_rate_epochs,
                                   dims=['trials','units','times'],
                                   coords=[lstg, units_id, times])
    # firing_rates_xr = firing_rates_xr.assign_coords(
    #                                 {'unit_type': ('units',units_type)})

    xr_file = f'{session_name}_{content}_{onset}{targetID}.nc'
    logger.info(f'Saving firing rate epochs: "{xr_file}"')
    firing_rates_xr.to_netcdf(os.path.join(v4a_dir, xr_file), engine='h5netcdf')


def _create_tfr_epochs(signalEpochs, session_name, onset, targetID, v4a_dir):
    """Compute Morlet TFR and save."""
    n_freqs = 30
    freqs = np.linspace(1, 150, n_freqs)
    n_cycles = freqs/6

    tfr = tfr_morlet(signalEpochs, freqs, n_cycles, return_itc=False, n_jobs=-1)
    tfrNorm = _zscore_signal(tfr.data, axis=-1)
    tfrNormalized = AverageTFR(tfr.info, tfrNorm, tfr.times, tfr.freqs, tfr.nave)

    tfr_file = f'{session_name}_{onset}{targetID}-tfr.h5'
    logger.info(f'Saving TFR: "{tfr_file}"')
    tfrNormalized.save(os.path.join(v4a_dir, tfr_file), overwrite=True)


def _create_power_epochs(signalEpochs, content, session_name,
                          onset, targetID, t0, k, v4a_dir):
    """Compute multitaper LFP power (beta or hga) and save."""

    power_params = {'beta': dict(ncycl=5, tb=3.5, freq=25),
                    'hga' : dict(ncycl=10, tb=4, freq=75)}
    
    p = power_params[content]
    power = mt.mt_hga(signalEpochs, f=p['freq'], n_cycles=p['ncycl'],
                          time_bandwidth=p['tb'], verbose=40)
    powerData = power.data[..., k:-k].squeeze()
    powerNormalized = _zscore_signal(powerData, axis=(0,2))

    powerEpochs = mne.EpochsArray(powerNormalized, power.info,
                                   events=power.events, event_id=power.event_id,
                                   tmin=t0, on_missing='ignore', verbose='ERROR')

    epoch_file = f'{session_name}_{content}_{onset}{targetID}-epo.fif'
    logger.info(f'Saving power epochs: "{epoch_file}"')
    powerEpochs.save(os.path.join(v4a_dir, epoch_file), overwrite=True)



# -----------------------------------------------------------------------
# ---------------------------- Main function ----------------------------
# -----------------------------------------------------------------------

def generate_epoch_files(neuralSignal, segBehavior, block, session_name,
                          onset, content, targetID, v4a_dir):
    """
    Parses trials, computes onsets, then dispatches
    to the appropriate content-specific epoch creator.
    """
    # Parameters
    k = 80 # samples to discard at edges to avoid multitaper artifacts 
    w1, w2 = preproc_funcs.windows(onset, k=k)
    t0 = -round((w1-k)/1000, 2) # for bhv
    t0_ = -round(w1/1000, 2) # for mua, tfr, lfp
    v_thres = 30 # eye velocity threshold (cm/s) to detect saccades

    # Trials
    trial_data = _parse_trials(segBehavior, targetID)
    target_onsets = trial_data['target_onsets']
    target_reached = trial_data['target_reached']

    xTarg = np.array(block.annotations['target_x_cm'])
    yTarg = np.array(block.annotations['target_y_cm'])

    anasig_behav = segBehavior.filter(name='Behavioural Signals [cm]')[0]
    anasig_ch_names = anasig_behav.array_annotations['channel_names']

    # Utility function
    def _pick(label):
        mask = anasig_ch_names == label
        return np.array(anasig_behav[:, mask]).squeeze()

    # Hand movements
    hand_x = _pick('HandXcm')
    hand_y = _pick('HandYcm')
    # hand_onsets = _get_hand_onsets(target_onsets, target_reached, segBehavior)
    hand_onsets = _detect_hand_onsets(hand_x, hand_y, target_onsets, 
                                      target_reached, targetID)
    trial_data['hand_onsets'] = hand_onsets

    # Eye movements
    eye_x = _pick('EyeXcm')
    eye_y = _pick('EyeYcm')
    saccade_onsets = _detect_saccade_onsets(eye_x, eye_y, target_onsets, 
                                        target_reached, v_thres)
    trial_data['eye_onsets'] = saccade_onsets

    # Select the event timestamps based on the onset
    onset_map = {'targ': target_onsets,
                 'eye' : saccade_onsets,
                 'hand': hand_onsets,
                 'reach': target_reached}
    event_onsets = onset_map[onset]

    # Dispatch to epoch creator function
    if content == 'bhv':
        _create_bhv_epochs(anasig_behav, hand_x, hand_y, eye_x, eye_y,
                           xTarg, yTarg, event_onsets, onset, session_name,
                           targetID, trial_data, t0, k, v4a_dir)
        
    elif content == 'units':
        _create_spike_epochs(neuralSignal, trial_data, event_onsets, w1, w2, 
                                session_name, content, onset, v4a_dir, targetID)
        
    elif content in ('mua', 'bnrmua'):
        _create_neural_epochs(neuralSignal, content, event_onsets, trial_data,
                               w1, w2, t0_, v4a_dir, session_name,
                               onset, targetID)

    elif content == 'tfr':
        signalEpochs = _create_neural_epochs(
            neuralSignal, content, event_onsets, trial_data,
            w1, w2, t0_, v4a_dir, session_name, onset, targetID, save=False)
        
        _create_tfr_epochs(signalEpochs, trial_data, session_name,
                            onset, targetID, v4a_dir)

    elif content in ('beta', 'hga'):
        signalEpochs = _create_neural_epochs(
            neuralSignal, content, event_onsets, trial_data,
            w1, w2, t0_, v4a_dir, session_name, onset, targetID, save=False)
        
        _create_power_epochs(signalEpochs, content, trial_data, session_name,
                              onset, targetID, t0_, k, v4a_dir)

    else:
        raise ValueError(f"Unknown content type '{content}'.")


if __name__ == '__main__':

    epoch_content = 'mua'

    for session_type in ['short12J','short12E']:

        session_group = utils.load_session_group(session_type)

        for session in session_group:

            v4a_dir = f'{utils.PATH}/{session}'

            # Get the Neo.Segments from the .nix file
            nix_filename = f'{session}_small.nix'   
            segments = {}
            segment_to_load = -1   # First segment(s) always out of synchronization
            logger.info(f"Loading the NIX data file {nix_filename}...")
            with neo.NixIO(os.path.join(v4a_dir, nix_filename), 'ro') as io:
                for block in io.read_all_blocks():
                    area = block.annotations['recording_area']
                    segments[area] = block.segments[segment_to_load]
            logger.info("...done.")

            # Get the Units, MUA or LFP data depending on the 'epoch_content'
            if epoch_content == 'units':
                neuralSignal = segments

            elif epoch_content == 'bnrmua':
                # Get the MUA from the .pkl file
                mua_filename = f'{session}_MUA.pkl'
                neuralSignal = _load_mua_pkl(v4a_dir, mua_filename)

            elif epoch_content == 'mua':
                # Get the MUAe from the .pkl file
                muae_filename = f'{session}_MUAe.pkl'
                neuralSignal = _load_mua_pkl(v4a_dir, muae_filename)

            elif epoch_content in ['beta','hga','tfr']: 
                # Unpack the LFP data from the segments
                neuralSignal = {}
                n_channels = {'motor':96, 'visual':128}
                anasig_name = 'Downsampled (factor 30) version of nsx6'
                for area, seg in segments.items(): 
                    neuralSignal[area] = seg.filter(name=anasig_name)[0]

            elif epoch_content == 'bhv':
                neuralSignal = {}
            else: 
                raise ValueError('Unknown "epoch_content".')

            # The timestamps of trials (events and epochs) are the same for the  
            # two blocks (synchronised) so we can keep one area to analyse the behavior.
            segBehavior = segments['visual']

            for targetID in [2,3,4]: # target 1 corresponds to initial central target of trial initiation
                for onset in ['hand']:
                    generate_epoch_files(neuralSignal, segBehavior, block, session, 
                                            onset, epoch_content, targetID, v4a_dir)
