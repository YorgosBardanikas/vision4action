"""Script to estimate the Mutual Information
and the null hypothesis based on shuffled trials."""

import os
import mne
import numpy as np
import xarray as xr
from analysis_scripts import utils
from frites.io import logger
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from scipy.signal import savgol_filter

# Load the MUA epochs
subj = 'enya'
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
# Example 2: code 114 -> landing sequence 11, target rank 4
# Example remapping: trial code 32 -> direction 1 (42 has the same direction)
code_groups = [[32,42,52,62,72,82,112,122],
                [93,53,103,33,13,123,23,73],
                [44,124,64,74,54,84,34,114]]
new_codes_mapping = [{32:1, 42:1, 52:2, 62:2, 72:3, 82:3, 112:4, 122:4},
                        {93:1, 53:1, 103:2, 33:2, 13:3, 123:3, 23:4, 73:4},
                        {44:1, 124:1, 64:2, 74:2, 54:3, 84:3, 34:4, 114:4}]

wf = WfMi(mi_type='cd', inference='ffx', verbose=False)
mi_group, mi_perms_group = [],[]

for group, mapping in zip(code_groups, new_codes_mapping):

    logger.info(f'   Group: {group}')

    # Remap the trial codes (e.g. each 32 becomes 1, each 52 becomes 2, etc)
    epochs_group = epochs[np.isin(codes, group)]
    codes_ = epochs_group.events[:, 2]
    new_codes = np.array([mapping.get(code, 0) for code in codes_])

    # Channels selection and downsampling for computational efficiency
    epochs_ = epochs_group.copy().get_data(picks=roi)
    epochs_ = savgol_filter(epochs_,11,2)
    inputEpochs = epochs_[..., ::decim_factor]

    ds = DatasetEphy([inputEpochs], y=new_codes, roi=roi, times=times)
    mi_, mi_p = wf.fit(ds, mcp='cluster', n_perm=nperms, stats=False)
    mi_ = mi_.T
    mi_group.append(mi_)
    mi_perms_group.append(mi_p)
    wf.clean()

mi = xr.concat(mi_group, dim='groups')
mi_perms = xr.concat(mi_perms_group, dim='groups')

# Save the data
path = f'{utils.PATH}/MI/'
mi_filename = f'{subj}-{onset}-{content}-MI.nc'
mi.to_netcdf(os.path.join(path, mi_filename),engine='h5netcdf')
# save permutations in ndarrays to avoid duplicating metadata in the mi xarray.DataArray
perms_filename = f'{subj}-{onset}-{content}-MI_perms.npy'
np.save(os.path.join(path, perms_filename), mi_perms.data)
logger.info(f'DataArray object is saved in the file: "{perms_filename}".')