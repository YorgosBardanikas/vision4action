"""Script to control for initial position effects using LDA as a classifier."""

import os
import mne
import utils
import numpy as np
import xarray as xr
from frites.io import logger
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the MUA and behavioral data
subj = 'enya'
# alignement on hand movement onset to avoid misclassification purely due to 
# early-initiated trials, and not due to anticipation/initial position effects.
onset = 'hand'
content = 'mua'
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
events = epochs.events[:,2]

# trial codes to identify movements on the train_group
train2 = [[11,7,8,9],[15,18,13,16],[15,7,13,9],[11,18,8,16]]
train_lists = [[2,3,4,6], train2]

# trial codes to identify movements on the test_group
test_lists = [[15,18,13,16],[11,7,8,9],[11,18,8,16],[15,7,13,9]]

exts = ['left','right','top','bottom']
# codes to remap 
n_list = [100,101]

for train_list, test_list, ext in zip(train_lists, test_lists, exts):

    # remap [2,3,4,6] to 100 and train2 to 101
    for tr, n in zip(train_list, n_list):
        events = np.where(np.isin(events, tr), n, events)

    # balance the number of trials in the two classes
    tr1 = np.where(events==100)[0]
    tr2 = np.where(events==101)[0]
    minsize = min([tr1.size,tr2.size])
    rng = np.random.default_rng(seed=10)
    tr1 = rng.choice(tr1, size=minsize, replace=False)
    tr2 = rng.choice(tr2, size=minsize, replace=False)
    train_ = np.concatenate((tr1,tr2))

    test_ = [f'Move_{i}' for i in test_list]
    predictions, predictions_shuff = [],[]
    tmins, tmaxs = np.arange(-0.2, 0.001, 0.01), np.arange(-0.15, 0.051, 0.01)

    for area in areas:

        logger.info(f'Fit LDA for area {area} ...')
        ch_inds = mne.pick_channels_regexp(ch_names,f'{area}')
        ch_names_ = np.array(ch_names)[ch_inds]
        pred,pred_shuff = [],[]

        for tmn, tmx in zip(tmins, tmaxs):

            # Train
            epochs_train = epochs[train_].copy().pick(ch_names_).crop(tmin=tmn,tmax=tmx)
            classes = events[train_]
            epochs_train = epochs_train.get_data().mean(2) # average across time
            lda = LDA()
            lda.fit(epochs_train, classes)

            # Train models on shuffled classes
            lda_models_perm = Parallel(n_jobs=-1)(delayed(utils.fit_lda)
                                                (epochs_train, classes, seed) 
                                                for seed in range(nperms))
            
            # Test
            epochs_test = epochs[test_].copy().pick(ch_names_).crop(tmin=tmn,tmax=tmx)
            test_input = epochs_test.get_data().mean(2)
            true_trials = epochs_test.events[:,2]
            pred.append(lda.predict(test_input))

            # Test on the model created by shuffled classes
            preds = np.zeros((nperms, true_trials.size))
            for npr,lda_model in enumerate(lda_models_perm):
                preds[npr] = lda_model.predict(test_input)
            pred_shuff.append(preds)
            
        predictions.append(pred)
        predictions_shuff.append(pred_shuff)
        
    lda_predictions = xr.DataArray(np.array(predictions), 
                        dims=['areas','timebins','trials'],
                        coords=[areas, tmins, true_trials])
    lda_predictions_shuff = xr.DataArray(np.array(predictions_shuff), 
                        dims=['areas','timebins','perms','trials'],
                        coords=[areas, tmins, np.arange(nperms), true_trials])

    filename = f'{subj}-{onset}-{content}_LDA_predictions_test_{ext}.nc'
    lda_predictions.to_netcdf(os.path.join(path, filename),engine='h5netcdf')

    filename_shuff = f'{subj}-{onset}-{content}_LDA_predictions_shuffled_{ext}.nc'
    lda_predictions_shuff.to_netcdf(os.path.join(path, filename_shuff),engine='h5netcdf')