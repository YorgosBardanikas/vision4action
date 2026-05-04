"""
Preprocessing script for the Vision4Action landing task data that are contained in the .nix files.

The script saves an MNE.Epoch object with neural (or behavioral) data aligned in a 
behavioral trigger. 

"""

import os
import neo
import mne
import numpy as np
import utils
import pickle
from frites.io import logger
from pandas import DataFrame
from brainets.spectral import mt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from mne.time_frequency import (tfr_morlet, AverageTFR)
from analysis_scripts.preprocessing import preproc_funcs

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


def _get_eye_onsets(eye_x, eye_y, target_onsets, target_reached):
    """Detect saccade onset per trial from peak eye velocity."""
    ntrials = target_onsets.shape[0]
    eye_onsets = np.zeros(ntrials, dtype=int)

    for tr, (t_on, t_r) in enumerate(zip(target_onsets, target_reached)):
        x = eye_x[t_on:t_r+1]
        y = eye_y[t_on:t_r+1]
        v_ = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        v = savgol_filter(v_, 130, 5)

        peak = np.argmax(v)
        while v[peak-5:peak].sum() > v[peak-10:peak-5].sum():
            peak -= 1
        eye_onsets[tr] = t_on + peak

    return eye_onsets


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
    signal = np.zeros((ntrials, nchannels, ntimes))
    for e, ev_onset in enumerate(event_onsets):
        signal[e] = signal[channels, ev_onset-w1 : ev_onset+w2]
    return signal


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

    eye_vel = preproc_funcs.calculate_vel(anasig_behav, event_onsets,
                                         onset, session_name, targetID, 'Eye')
    hand_vel = preproc_funcs.calculate_vel(anasig_behav, event_onsets,
                                         onset, session_name, targetID, 'Hand')
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


def _create_neural_epochs(analogSignal, content, event_onsets, trial_data,
                          w1, w2, t0_, v4a_dir, session_name,
                          onset, targetID, save=True):
    """Cut the neural signal in epochs, normalize and save.
    If save=False, return the signal epochs for downstream use."""

    if content == 'bnrmua':
        # Convolve with a Gaussian kernel (size: 2*sigma*truncate, 40ms)
        visualSig = gaussian_filter1d(analogSignal['visual'].astype(float), sigma=5, axis=1)
        motorSig  = gaussian_filter1d(analogSignal['motor'].astype(float),  sigma=5, axis=1)
    elif content == 'mua':
        visualSig = analogSignal['visual'] # shape (nchannels, ntimes)
        motorSig  = analogSignal['motor']
    else: # for hga, beta, tfr : get the LFP
        visualSig = analogSignal['visual'].rescale('V').magnitude.T    
        motorSig = analogSignal['motor'].rescale('V').magnitude.T 

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

def generate_epoch_files(analogSignal, segBehavior, block, session_name,
                          onset, content, targetID, v4a_dir):
    """
    Main entry point. Parses trials, computes onsets, then dispatches
    to the appropriate content-specific epoch creator.
    """
    # Parameters
    k = 80 # samples to discard at edges to avoid multitaper artifacts 
    w1, w2 = preproc_funcs.windows(onset, k=k)
    t0 = -round((w1-k)/1000, 2) # for bhv
    t0_ = -round(w1/1000, 2) # for mua, tfr, lfp

    # Utility function
    def _pick(label):
        mask = anasig_ch_names == label
        return np.array(anasig_behav[:, mask]).squeeze()

    # Trials
    trial_data = _parse_trials(segBehavior, targetID)
    target_onsets = trial_data['target_onsets']
    target_reached = trial_data['target_reached']

    xTarg = np.array(block.annotations['target_x_cm'])
    yTarg = np.array(block.annotations['target_y_cm'])

    anasig_behav = segBehavior.filter(name='Behavioural Signals [cm]')[0]
    anasig_ch_names = anasig_behav.array_annotations['channel_names']

    # Hand movements
    hand_x = _pick('HandXcm')
    hand_y = _pick('HandYcm')
    hand_onsets = _get_hand_onsets(target_onsets, target_reached, segBehavior)
    trial_data['hand_onsets'] = hand_onsets

    # Eye movements
    eye_x = _pick('EyeXcm')
    eye_y = _pick('EyeYcm')
    eye_onsets = _get_eye_onsets(eye_x, eye_y, target_onsets, target_reached)
    trial_data['eye_onsets'] = eye_onsets

    # Select the event timestamps based on the onset
    onset_map = {'targ': target_onsets,
                 'eye' : eye_onsets,
                 'hand': hand_onsets,
                 'reach': target_reached}
    event_onsets = onset_map[onset]

    # Dispatch to epoch creator function
    if content == 'bhv':
        _create_bhv_epochs(anasig_behav, hand_x, hand_y, eye_x, eye_y,
                           xTarg, yTarg, event_onsets, onset, session_name,
                           targetID, trial_data, t0, k, v4a_dir)

    elif content in ('mua', 'bnrmua'):
        _create_neural_epochs(analogSignal, content, event_onsets, trial_data,
                               w1, w2, t0_, v4a_dir, session_name,
                               onset, targetID)

    elif content == 'tfr':
        signalEpochs = _create_neural_epochs(
            analogSignal, content, event_onsets, trial_data,
            w1, w2, t0_, v4a_dir, session_name, onset, targetID, save=False)
        
        _create_tfr_epochs(signalEpochs, trial_data, session_name,
                            onset, targetID, v4a_dir)

    elif content in ('beta', 'hga'):
        signalEpochs = _create_neural_epochs(
            analogSignal, content, event_onsets, trial_data,
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

            # Get the MUA or LFP data depending on the 'epoch_content'
            if epoch_content == 'bnrmua':
                # Get the MUA from the .pkl file
                mua_filename = f'{session}_MUA.pkl'
                analogSignal = _load_mua_pkl(v4a_dir, mua_filename)

            elif epoch_content == 'mua':
                # Get the MUAe from the .pkl file
                muae_filename = f'{session}_MUAe.pkl'
                analogSignal = _load_mua_pkl(v4a_dir, muae_filename)

            elif epoch_content in ['beta','hga','tfr']: 
                # Unpack the LFP data from the segments
                analogSignal = {}
                n_channels = {'motor':96, 'visual':128}
                anasig_name = 'Downsampled (factor 30) version of nsx6'
                for area, seg in segments.items(): 
                    analogSignal[area] = seg.filter(name=anasig_name)[0]

            elif epoch_content == 'bhv':
                analogSignal = {}
            else: 
                raise ValueError('Unknown "epoch_content".')

            # The timestamps of trials (events and epochs) are the same for the  
            # two blocks (synchronised) so we can keep one area to analyse the behavior.
            segBehavior = segments['visual']

            for targetID in [2,3,4]: # target 1 corresponds to initial central target of trial initiation
                for onset in ['targ','hand']:
                    generate_epoch_files(analogSignal, segBehavior, block, session, 
                                            onset, epoch_content, targetID, v4a_dir)
