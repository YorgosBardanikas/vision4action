"""
Script to compute the cross-validated CCA (Canonical Correlation Analysis) 
between 7A and M1 neural activity.
Use only channels with highest directional information to make sure that
CCA will converge to a stable result.
Downsample for computational efficiency.
"""

import os
import mne
import pickle
import numpy as np
import xarray as xr
from analysis_scripts import utils
from joblib import Parallel, delayed
from frites.io import logger
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split

# ----------------
# Helper functions
# ----------------

def _pick_channels(epochs, area_prefix, best_features, use_best):
    """Return epoch data (ntrials, nch, ntimes) for a given area."""
    ch_names = np.array(epochs.ch_names)
    if use_best:
        inds = mne.pick_channels_regexp(best_features, f'^{area_prefix}')
        picks = best_features[inds]
    else:
        inds = mne.pick_channels_regexp(ch_names, f'^{area_prefix}')
        picks = ch_names[inds]
    return epochs.copy().pick(picks).get_data()


def _downsample(data, decim, edge):
    """Downsample along time axis by decimation factor."""
    return savgol_filter(data,51,2)[..., edge:-edge][..., ::decim]


def _run_cca(epoch_7a_t, epoch_m1_t, idx_train, idx_test):
    """Fit and transform CCA at a single timepoint. Returns scores and weights."""
    cca = CCA(n_components=1) # select top componenet
    cca.fit(epoch_7a_t[idx_train], epoch_m1_t[idx_train])
    x_cca, y_cca = cca.transform(epoch_7a_t[idx_test], epoch_m1_t[idx_test])

    cca_res = (x_cca.squeeze(), y_cca.squeeze(),
               cca.x_weights_.squeeze(),cca.y_weights_.squeeze())
    return cca_res


def _cca_perm(epoch_7a_t, epoch_m1_t, idx_train, idx_test, ntrials, seed):
    """
    Fit one permuted CCA model at a single timepoint.
    Shuffles M1 trial labels before fitting, leaving 7A intact.
    Returns projected test scores and weights for both areas.
    """
    rng = np.random.default_rng(seed=seed+100)
    epoch_m1_t_sh = epoch_m1_t[rng.permutation(ntrials)]

    cca = CCA(n_components=1)
    cca.fit(epoch_7a_t[idx_train], epoch_m1_t_sh[idx_train])
    x_p, y_p = cca.transform(epoch_7a_t[idx_test], epoch_m1_t_sh[idx_test])

    return (x_p.squeeze(), y_p.squeeze(), 
            cca.x_weights_.squeeze(), cca.y_weights_.squeeze())


def _save(path, filename, data, ext):
    """Save an xarray DataArray (.nc) or any object (.pkl)."""
    full_path = os.path.join(path, filename)
    if ext == 'nc':
        data.to_netcdf(full_path, engine='h5netcdf')
    elif ext == 'pkl':
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unknown format '{ext}'. Use 'nc' or 'pkl'.")
    logger.info(f'Saved: "{full_path}"')



if __name__ == '__main__':

    # Parameters setting
    subj = 'enya'
    content = 'mua'
    onset = 'targ'
    rgr = 'goal'
    nperms = 192
    use_best = True   # False = use all channels in each area (CCA converging unstable)
    session_type = 'short12J' if subj == 'jazz' else 'short12E'
    path = '/path_to_directory/CCA/'
    decim_factor = 5
    edge_crop = 100   # samples trimmed from each edge after smoothing
    onset_windows = {'targ': (-0.5, 0.8), 'hand': (-0.6, 0.6)}

    # Event codes per target rank
    target_codes = {2: [32,42,52,62,72,82,112,122],
                    3: [53,93,33,103,13,123,23,73],
                    4: [44,124,64,74,54,84,34,114]}

    # Event codes to exclude (directions not repeated across all targets)
    drop = [12, 22, 92, 102, 43, 14, 83, 94, 63, 24, 113, 104]

    # Load the MUA data 
    epoch_list, epoch_list_bhv = [], []
    for tt in [2,3,4]:
        epoch_list.append(utils.load_epochs(session_type, onset, tt, content=content))
        epoch_list_bhv.append(utils.load_epochs(session_type, onset, tt, content='bhv'))

    epochs = mne.concatenate_epochs(epoch_list, on_mismatch='ignore')
    epochs_bhv = mne.concatenate_epochs(epoch_list_bhv, on_mismatch='ignore')
    epochs_bhv, epochs = utils.keep_1attempt_trials(epochs_bhv, epochs)

    # Keep trials that are repeated across targets
    events = epochs.events[:, 2]
    directions, _ = utils.group_events(events, 'motor')
    keep = np.isin(events, drop, invert=True)
    epochs = epochs[keep]
    trials = epochs.events[:, 2]
    directions = directions[keep]

    # Channel selection for ensuring the convergence of CCA to stable result
    filename = f'{subj}-{onset}-{content}_best-channels.npy'
    best_features = np.load(os.path.join(path, filename))
    epochs_7a = _pick_channels(epochs, '7A', best_features, use_best)
    epochs_m1 = _pick_channels(epochs, 'M1', best_features, use_best)

    # Downsample to 200 Hz for computational efficiency
    tmin, tmax = onset_windows[onset]
    epochs = epochs.crop(tmin=tmin, tmax=tmax)
    times = epochs.times[edge_crop:-edge_crop][::decim_factor]
    ntimes = times.size
    epochs_7A = _downsample(epochs_7a, decim_factor, edge_crop)
    epochs_M1 = _downsample(epochs_m1, decim_factor, edge_crop)

    nch_7a, nch_m1 = epochs_7A.shape[1], epochs_M1.shape[1]
    perms = np.arange(nperms)

    # CCA per target 
    for targetID, codes in target_codes.items():

        logger.info(f'Running CCA for target {targetID}...')

        idxs = np.where(np.isin(trials, codes))[0]
        ntrials = idxs.size
        dirs = directions[idxs]
        ep_7a = epochs_7A[idxs]
        ep_m1 = epochs_M1[idxs]

        idx = np.arange(ntrials)
        idx_train, idx_test = train_test_split(idx, test_size=0.5, random_state=10)
        ntest = idx_test.size
        new_trials = dirs[idx_test]

        # Initialize output arrays
        cca_7A = np.zeros((ntest, ntimes))
        cca_M1 = np.zeros((ntest, ntimes))
        perms_7A = np.zeros((nperms, ntest, ntimes))
        perms_M1 = np.zeros((nperms, ntest, ntimes))
        weights_7a = np.zeros((nch_7a, ntimes))
        weights_m1 = np.zeros((nch_m1, ntimes))
        weights_7a_shuf = np.zeros((nperms, nch_7a, ntimes))
        weights_m1_shuf = np.zeros((nperms, nch_m1, ntimes))

        for t in range(ntimes):

            logger.info(f'  Target {targetID} | time {t+1}/{ntimes}')

            ep_7a_t = ep_7a[..., t]
            ep_m1_t = ep_m1[..., t]

            # Real CCA
            x_c, y_c, wx, wy = _run_cca(ep_7a_t, ep_m1_t, idx_train, idx_test)
            cca_7A[:,t] = x_c
            cca_M1[:,t] = y_c
            weights_7a[:,t] = wx
            weights_m1[:,t] = wy

            # Permuted CCAs
            perm_results = Parallel(n_jobs=-1)(delayed(_cca_perm)
                            (ep_7a_t, ep_m1_t, idx_train, idx_test, ntrials, p)
                            for p in range(nperms))
            
            for p, (xp, yp, wxp, wyp) in enumerate(perm_results):
                perms_7A[p,:,t] = xp
                perms_M1[p,:,t] = yp
                weights_7a_shuf[p,:,t] = wxp
                weights_m1_shuf[p,:,t] = wyp

        # Format in xarrays
        cca_dimensions = xr.DataArray(np.stack([cca_7A, cca_M1], axis=0),            
                                    dims=['areas','trials','times'],
                                    coords=[['7A','M1'], new_trials, times])
        
        cca_perms = xr.DataArray(np.stack([perms_7A, perms_M1], axis=0),
                                dims=['areas','perms','trials','times'],
                                coords=[['7A','M1'], perms, new_trials, times])
        
        cca_weights = {'7A':weights_7a, 'M1':weights_m1}
        cca_weights_shuf = {'7A':weights_7a_shuf, 'M1':weights_m1_shuf}

        # Save the CCA results 
        tag = f'{subj}-{onset}-{content}_iCCA_{{}}_{rgr}_t{targetID}_50ms_cv_cond'
        _save(path, tag.format('dimensions_best') + '.nc', cca_dimensions, 'nc')
        _save(path, tag.format('perms') + '.nc',  cca_perms, 'nc')
        _save(path, tag.format('weights') + '.pkl', cca_weights, 'pkl')
        _save(path, tag.format('weights_shuf') + '.pkl', cca_weights_shuf,'pkl')

        logger.info(f'Target {targetID} done.')