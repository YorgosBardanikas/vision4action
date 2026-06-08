"""Script to test the LDA model on all trials. 
It also tests the LDA models created by shuffled classes on all trials.
Separation to train and test trials will be done in the plotting script
based on the trial codes (true_trials)."""

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
nld = 2
lds = [f'Ax{l+1}' for l in range(nld)]
path = f'{utils.PATH}/LDA_6dirs/'

### ----- Load all the data -----

# Load the LDA model and the shuffled models
lda_loaded = []
filenames = [f'{subj}-{onset}-{content}_LDA_model_{rgr}.pkl',
             f'{subj}-{onset}-{content}_LDA_model_{rgr}_shuffled.pkl']
for filename in filenames:
    with open(os.path.join(path, filename),'rb') as handle:
        lda_loaded.append(pickle.load(handle))

ldas, lda_models_perms = lda_loaded

# Load the relevant epochs of MUA and behavior
session_type = 'short12J' if subj == 'jazz' else 'short12E'
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

# # Exclude segments with horizontal/vertical movements because 
# # they are not repeated across targets
# codes_to_keep = [f'LS {i} Target 2' for i in [3,4,5,6,7,8,11,12]] \
#               + [f'LS {i} Target 3' for i in [1,2,9,10]] \
#               + [f'LS {i} Target 4' for i in [4,6,8,11]] \
#               + [f'LS {i} Target {j}' for i in [3,5,7,12] for j in [3,4]]
# epochs = epochs[codes_to_keep]
true_trials = epochs.events[:,2]


### ----- Test the LDA models (real and shuffled) -----

# Project new trials and predict movement direction
predictions,projections,predictions_shuf = [],[],[]

for area, lda, lda_perms in zip(areas, ldas, lda_models_perms):

    ch_inds = mne.pick_channels_regexp(ch_names,f'{area}')
    ch_names_ = np.array(ch_names)[ch_inds]
    epochs_ = epochs.copy().pick(ch_names_)
    epochs_tmp = epochs_.get_data().transpose(0,2,1) # (trials, times, channels)
    ntr,nt,nch = epochs_tmp.shape
    epochs_lda = epochs_tmp.reshape((ntr*nt,nch))

    # Compute predictions of the movement direction from the real LDA model
    predictions.append(lda.predict(epochs_lda).reshape((ntr,nt)))
    # Project new trials on the discriminant axes from the real LDA model
    projections.append(lda.transform(epochs_lda).reshape((ntr,nt,nld)).transpose(2,0,1))

    # Compute predictions of the movement direction from LDA models trained on shuffled directions
    preds = []
    for lda_ in lda_perms:
        preds.append(lda_.predict(epochs_lda).reshape((ntr,nt)))
    predictions_shuf.append(preds)


### ----- Format in xarrays and save -----

# The real predictions and projections
trial_counts = np.arange(true_trials.shape[0])
lda_projections = xr.DataArray(np.array(projections), dims=['areas','PCs','trials','times'],
                                    coords=[areas, lds, trial_counts, times])
lda_predictions = xr.DataArray(np.array(predictions), dims=['areas','trials','times'],
                                                    coords=[areas, trial_counts, times])
lda_predictions = lda_predictions.assign_coords(true_trials=('trials',true_trials))

lda_set = xr.Dataset({'projections':lda_projections, 'predictions':lda_predictions})
filename1 = f'{subj}-{onset}-{content}_LDA_LS{rgr}.nc'
lda_set.to_netcdf(os.path.join(path, filename1), engine='h5netcdf')

# The shuffled predictions
data = np.array(predictions_shuf).astype(np.float32)
lda_predictions_shuf = xr.DataArray(data, dims=['areas','perms','trials','times'],
                        coords=[areas, np.arange(nperms), true_trials, times])

filename2 = f'{subj}-{onset}-{content}_LDA_LS{rgr}_shuffled.nc'
lda_predictions_shuf.to_netcdf(os.path.join(path, filename2), engine='h5netcdf')