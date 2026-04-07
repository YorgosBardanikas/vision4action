"""Script to estimate the Feature-specific Information Transfer
and the null hypothesis based on shuffled trials."""

import os
import mne
import utils
import numpy as np
import xarray as xr
from frites.io import logger
from FIT_core import conn_fit_
from scipy.signal import savgol_filter
from joblib import Parallel, delayed


def compute_fit_perms(inputEpochs, bhv_var, roi, times, sfreq=1000, nperms=None):
    "Wrapping function to compute the FIT and its permutations."

    fit = conn_fit_(inputEpochs, y=bhv_var, roi=roi, times=times, 
                mi_type='cd', net=False, avg_delay=True, max_delay=.1, sfreq=sfreq)

    if isinstance(nperms,int):

        # Create partitions of the trials to shuffle the sources
        ntrials = bhv_var.size
        bhv_var_perms = np.zeros((nperms,ntrials),dtype=int)
        for p in range(nperms):
            rng = np.random.default_rng(seed=p)
            bhv_var_perms[p] = rng.permutation(bhv_var)

        # Parallelize jobs across permutations
        fit_tmp = Parallel(n_jobs=-1)(delayed(conn_fit_)(inputEpochs, 
                        y=bhv_, roi=roi, times=times, 
                        mi_type='cd', net=False, avg_delay=True, 
                        max_delay=.1, sfreq=sfreq, nLoop=n) 
                        for n,bhv_ in enumerate(bhv_var_perms))
        fit_perms = xr.concat(fit_tmp, dim='perms').astype(np.float32)
        return fit, fit_perms
    
    else: return fit, 0


# Load the MUA epochs
subj = 'jazz'
onset = 'targ'
content = 'mua'
nperms = 128
sfreq = 1000
session_type = 'short12J' if subj == 'jazz' else 'short12E'
epochList = [utils.load_epochs(session_type, onset, targetID, content=content) 
             for targetID in [2,3,4]]
epochs = mne.concatenate_epochs(epochList, on_mismatch='ignore')

codes = epochs.events[:,2]
ch_names = np.array(epochs.ch_names)
ch_inds = mne.pick_channels_regexp(ch_names, '^7A|^M1')
epochs.pick(ch_inds)
roi = ch_names[ch_inds]
decim_factor = 5 # sampling rate: from 1000 Hz to 200 Hz
times = epochs.times[::decim_factor]*sfreq
sfreq = int(sfreq/decim_factor)

# Trial codes and their remapping
# Example: trial code 32 -> landing sequence 3, target rank 2
# Example remapping: trial code 32 -> direction 1 (42 has the same direction)
code_groups = [[32,42,52,62,72,82,112,122],
               [93,53,103,33,13,123,23,73],
               [44,124,64,74,54,84,34,114]]
new_codes_mapping = [{32:1, 42:1, 52:2, 62:2, 72:3, 82:3, 112:4, 122:4},
                     {93:1, 53:1, 103:2, 33:2, 13:3, 123:3, 23:4, 73:4},
                     {44:1, 124:1, 64:2, 74:2, 54:3, 84:3, 34:4, 114:4}]

fit_group, fit_perms_group = [],[]
for group, mapping in zip(code_groups, new_codes_mapping):

    logger.info(f'   Group: {group}')

    # Remap the trial codes (e.g. each 32 becomes 1, each 52 becomes 2, etc)
    epochs_group = epochs[np.isin(codes, group)]
    codes_ = epochs_group.events[:, 2]
    new_codes = np.array([mapping.get(code, 0) for code in codes_])

    # Channels/times selection and downsampling for computational efficiency
    epochs_ = epochs_group.copy().get_data(picks=roi)
    epochs_ = savgol_filter(epochs_,11,2)
    inputEpochs = epochs_[..., ::decim_factor]
    fit_, fit_p = compute_fit_perms(inputEpochs, new_codes, roi, 
                                    times, nperms=nperms, sfreq=sfreq)
    fit_group.append(fit_)
    fit_perms_group.append(fit_p)


# Save the data
path = f'{utils.v4a_dir}/FIT/'
fit = xr.concat(fit_group, dim='groups')
fit_filename = f'{subj}-{onset}-{content}-FIT.nc'
fit.to_netcdf(os.path.join(path, fit_filename),engine='h5netcdf')

fit_perms = xr.concat(fit_perms_group, dim='groups')
perms_filename = f'{subj}-{onset}-{content}-FIT_perms'
np.save(os.path.join(path, perms_filename), fit_perms.data)