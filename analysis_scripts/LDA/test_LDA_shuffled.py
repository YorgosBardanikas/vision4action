"""Script to test the LDA models created by shuffled classes on all trials.
Same structure as the test_LDA_model.py"""

import os
import mne
import utils
import pickle
import numpy as np
import xarray as xr

# Parameters setting
subj = 'enya'
onset = 'targ'
content = 'mua'
rgr = 'goal'
nperms = 192
session_type = 'short12J' if subj == 'jazz' else 'short12E'
path = f'{utils.v4a_dir}/LDA/'

# Load the LDA model
with open(path + f'{subj}-{onset}-{content}_LDA_model_{rgr}_shuffled.pkl','rb') as handle:
    lda_models_perms = pickle.load(handle)

# Load all the data
epochList,epochListBhv = [],[]
for tt in [2,3,4]:
    epochList.append(utils.load_epochs(session_type, onset, tt, content=content))
    epochListBhv.append(utils.load_epochs(session_type, onset, tt, content='bhv'))
epochs = mne.concatenate_epochs(epochList, on_mismatch='ignore')
epochs_bhv = mne.concatenate_epochs(epochListBhv, on_mismatch='ignore')
epochs_bhv, epochs = utils.keep_1attempt_trials(epochs_bhv, epochs)
ch_names = epochs.ch_names
times = epochs.times
areas = ['7A','M1']

# Exclude segments with horizontal/vertical movements
codes_to_keep = [f'LS {i} Target 2' for i in [3,4,5,6,7,8,11,12]] \
              + [f'LS {i} Target 3' for i in [1,2,9,10]] \
              + [f'LS {i} Target 4' for i in [4,6,8,11]] \
              + [f'LS {i} Target {j}' for i in [3,5,7,12] for j in [3,4]]
epochs = epochs[codes_to_keep]
true_trials = epochs.events[:,2]

# Test the LDA
predictions_areas = []
for area, lda_perms in zip(areas, lda_models_perms):
    ch_inds = mne.pick_channels_regexp(ch_names,f'{area}')
    ch_names_ = np.array(ch_names)[ch_inds]
    epochs_ = epochs.copy().pick(ch_names_)
    epochs_tmp = epochs_.get_data().transpose(0,2,1) # (trials, times, channels)
    ntr,nt,nch = epochs_tmp.shape
    epochs_lda = epochs_tmp.reshape((ntr*nt,nch))

    predictions = []
    for lda in lda_perms:
        predictions.append(lda.predict(epochs_lda).reshape((ntr,nt)))
    predictions_areas.append(predictions)

# Format and save the xarray
data = np.array(predictions_areas).astype(np.float32)
lda_predictions = xr.DataArray(data, dims=['areas','perms','trials','times'],
                        coords=[areas, np.arange(nperms), true_trials, times])

filename = f'{subj}-{onset}-{content}_LDA_LS{rgr}_shuffled.nc'
lda_predictions.to_netcdf(os.path.join(path, filename), engine='h5netcdf')