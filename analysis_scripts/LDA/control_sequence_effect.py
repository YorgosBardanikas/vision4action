"""
"""

import os
import mne
import utils
import pickle
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the MUA and behavioral data
subj = 'jazz'
onset = 'targ'
content = 'mua'
nperms = 192
session_type = 'short12J' if subj == 'jazz' else 'short12E'
path = f'{utils.PATH}/LDA_2d/'

epochList,epochListBhv = [],[]
for targetID in [2,3,4]:
    epochList.append(utils.load_epochs(session_type, onset, targetID, content=content))
    epochListBhv.append(utils.load_epochs(session_type, onset, targetID, content='bhv'))

epochs = mne.concatenate_epochs(epochList, on_mismatch='ignore')
epochs_bhv = mne.concatenate_epochs(epochListBhv, on_mismatch='ignore')
epochs_bhv, epochs = utils.keep_1attempt_trials(epochs_bhv, epochs)

ch_names = epochs.ch_names
times = epochs.times
areas = ['7A','M1']

codes = ['LS 1 Target 3','LS 2 Target 3']
predictions, predictions_shuf = [],[]

for area in areas:
    ch_inds = mne.pick_channels_regexp(ch_names,f'{area}')
    ch_names_ = np.array(ch_names)[ch_inds]
    tmn, tmx = 0, 0.2

    # Train in 60% of trials
    epochs_ = epochs[codes].copy().pick(ch_names_).crop(tmin=tmn,tmax=tmx)
    classes = epochs_.events[:,2]
    idxs = np.arange(classes.size)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=10)    
    train_idx, test_idx = next(sss.split(idxs, classes))
    epochs_ = epochs_.get_data().mean(2) # average across time
    lda = LDA()
    lda.fit(epochs_[train_idx], classes[train_idx])

    # Train many LDA models after shuffling the trial classes
    lda_models_perm = Parallel(n_jobs=-1)(delayed(utils.fit_lda)
                                        (epochs_[train_idx], classes[train_idx], seed) 
                                        for seed in range(nperms))
    
    # Test in the rest 40%
    epochs_test = epochs[codes][test_idx].copy().pick(ch_names_)
    ntr,nch,nt = epochs_test.get_data().shape
    true_trials = epochs_test.events[:,2]
    test_input = epochs_test.get_data().transpose(0,2,1).reshape((ntr*nt,nch))
    predictions.append(lda.predict(test_input).reshape((ntr,nt)))

    # Compute predictions from models trained on shuffled directions
    preds = []
    for lda_perms in lda_models_perm:
        preds.append(lda_perms.predict(test_input).reshape((ntr,nt)))
    predictions_shuf.append(preds)


# Format and save xarray
lda_predictions = xr.DataArray(np.array(predictions), 
                        dims=['areas','trials','times'],
                        coords=[areas, true_trials, times])
filename = f'{subj}-{onset}-{content}_LDA_control_sequence.nc'
lda_predictions.to_netcdf(os.path.join(path, filename),engine='h5netcdf')

# The shuffled predictions
data = np.array(predictions_shuf).astype(np.float32)
lda_predictions_shuf = xr.DataArray(data, dims=['areas','perms','trials','times'],
                        coords=[areas, np.arange(nperms), true_trials, times])

filename2 = f'{subj}-{onset}-{content}_LDA_shuffled_control_sequence.nc'
lda_predictions_shuf.to_netcdf(os.path.join(path, filename2), engine='h5netcdf')