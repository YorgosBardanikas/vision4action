"""Script to train an LDA model to maximize trial split across movement directions.
The discriminant axes compose a low-dimensional space in which neural activity
occupies distinct states related to different movement directions.
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
subj = 'enya'
onset = 'targ'
content = 'mua'
rgr = 'goal'
nperms = 192
session_type = 'short12J' if subj == 'jazz' else 'short12E'
path = f'{utils.v4a_dir}/LDA/'

epochList,epochListBhv = [],[]
for targetID in [2,3,4]:
    epochList.append(utils.load_epochs(session_type, onset, targetID, content=content))
    epochListBhv.append(utils.load_epochs(session_type, onset, targetID, content='bhv'))

epochs = mne.concatenate_epochs(epochList, on_mismatch='ignore')
epochs_bhv = mne.concatenate_epochs(epochListBhv, on_mismatch='ignore')
epochs_bhv, epochs = utils.keep_1attempt_trials(epochs_bhv, epochs)
events_group, events_ids = utils.group_events(epochs.events[:,2],'motor')

epochs.events[:,2] = events_group
epochs.event_id = events_ids
ch_names = epochs.ch_names
times = epochs.times
areas = ['7A','M1']

# regressor 'goal':     train on the first target and test on the second and third targets.
# regressor 'goal8020': train on the 80% of the first target trials and test on the 20% of
#                       the rest 20% for cross-validation.

if rgr == 'goal':

    train = [f'Move_{i}' for i in [2,3,4,6]]
    test = [f'Move_{i}' for i in [11,15,9,16,7,18,8,13]]
    nld = 2
    lds = [f'Ax{l+1}' for l in range(nld)]
    lda_list, lda_models_perms = [],[]

    for area in areas:
        ch_inds = mne.pick_channels_regexp(ch_names,f'{area}')
        ch_names_ = np.array(ch_names)[ch_inds]
        tmn, tmx = 0, 0.2 # time window: first 200 ms after target onset

        # Train the LDA model
        epochs_train = epochs[train].copy().pick(ch_names_).crop(tmin=tmn,tmax=tmx)
        classes = epochs_train.events[:,2]
        lda_input = epochs_train.get_data().mean(2) # average across time
        lda = LDA(n_components=nld)
        lda.fit(lda_input, classes)
        lda_list.append(lda)

        # Train many LDA models after shuffling the trial classes
        lda_models_perm = Parallel(n_jobs=-1)(delayed(utils.fit_lda)
                                            (epochs_train, classes, seed) 
                                            for seed in range(nperms))
        lda_models_perms.append(lda_models_perm)
        
    # Save the results
    with open(path + f'{subj}-{onset}-{content}_LDA_model_{rgr}.pkl','wb') as handle:
        pickle.dump(lda_list, handle)
    with open(path + f'{subj}-{onset}-{content}_LDA_model_{rgr}_shuffled.pkl','wb') as handle:
        pickle.dump(lda_models_perms, handle)


elif rgr == 'goal8020':

    codes = [f'Move_{i}' for i in [2,3,4,6]]
    predictions, projections = [],[]
    for area in areas:
        ch_inds = mne.pick_channels_regexp(ch_names,f'{area}')
        ch_names_ = np.array(ch_names)[ch_inds]
        tmn, tmx = 0, 0.2

        # Train in 80% of trials (of the first movement, center-out)
        epochs_train = epochs[codes].copy().pick(ch_names_).crop(tmin=tmn,tmax=tmx)
        classes = epochs_train.events[:,2]
        idxs = np.arange(classes.size)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)    
        train_idx, test_idx = next(sss.split(idxs, classes))
        epochs_train = epochs_train.get_data().mean(2) # average across time
        lda = LDA()
        lda.fit(epochs_train[train_idx], classes[train_idx])

        # Test in the rest 20%
        epochs_test = epochs[codes][test_idx].copy().pick(ch_names_)
        ntr,nch,nt = epochs_test.get_data().shape
        true_trials = epochs_test.events[:,2]
        test_input = epochs_test.get_data().transpose(0,2,1).reshape((ntr*nt,nch))
        predictions.append(lda.predict(test_input).reshape((ntr,nt)))

    # Format and save xarray
    lda_predictions = xr.DataArray(np.array(predictions), 
                            dims=['areas','trials','times'],
                            coords=[areas, true_trials, times])
    filename = f'{subj}-{onset}-{content}_LDA_predictions_{rgr}.nc'
    lda_predictions.to_netcdf(os.path.join(path, filename),engine='h5netcdf')