
import os
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d
from sklearn.svm import SVC
from analysis_scripts import utils

# Load the units data
subj = 'jazz'
onset = 'targ'
content = 'units'
session_type = 'short12J' if subj == 'jazz' else 'short12E'
sessions = utils.load_session_group(session_type)
sessions = ['j210215-land-003']

for session in sessions:

    v4a_dir = f'{utils.PATH}/{session}'
    path = f'{utils.PATH}/LDA_spikes/'
    pmd_rates_all = []
    dirs_train = [2,3,4,6]
    dirs_test = [11,15,9,16,7,18,8,13]
    best_units = []
    n = 15 # amount of neurons with largest mean firing rate to keep

    for t in [2,3,4]:

        filename = f'{session}_{content}_{onset}{t}.nc'
        firing_rates = xr.load_dataarray(os.path.join(v4a_dir, filename))

        pmd_idx = firing_rates.units.str.contains('M1-PMd')
        pmd_rates = firing_rates.sel(units=pmd_idx)

        lstg = pmd_rates.trials.values
        directions, _ = utils.group_events(lstg, 'motor')
        pmd_rates['trials'] = directions

        mean_frate = pmd_rates.mean(('trials','times'))
        idx = np.argsort(mean_frate)[::-1][:n] 
        best_units.append(idx)

        pmd_rates = xr.apply_ufunc(uniform_filter1d, pmd_rates, kwargs={'size':50, 'axis':-1})
        pmd_rates_all.append(pmd_rates)

    best_units_common = np.intersect1d(
                        np.intersect1d(best_units[0], best_units[1]), best_units[2])
    pmd_rates_common = [arr.isel(units=best_units_common) for arr in pmd_rates_all]

    # Train in 2nd target
    directions_tg2 = pmd_rates_all[0].trials
    pmd_rates_train = pmd_rates_all[0].sel(trials=directions_tg2.isin(dirs_train), 
                                        times=slice(0,200)).mean('times')
    classes = pmd_rates_train.trials
    lda = SVC(C=1, kernel='linear')
    lda.fit(pmd_rates_train.values, classes.values)

    # Test in all targets
    predictions, projections, true_trials, targets = [],[],[],[]
    for p,pmd_rates_t in enumerate(pmd_rates_all):

        if p == 0: dirs_to_test = dirs_train
        else: dirs_to_test = dirs_test

        directions_tg = pmd_rates_t.trials
        pmd_rates_test = pmd_rates_t.sel(trials=directions_tg.isin(dirs_to_test))
        ntr,nu,nt = pmd_rates_test.trials.size, pmd_rates_test.units.size, pmd_rates_test.times.size
        pmd_rates_test_np = pmd_rates_test.transpose('trials','times','units').values
        lda_test = pmd_rates_test_np.reshape((ntr*nt,nu))

        predictions.append(lda.predict(lda_test).reshape((ntr,nt)))
        true_trials.extend(pmd_rates_test.trials.values)
        targets.extend([p+2]*ntr)

    predictions = np.concatenate(predictions, axis=0)

    ### ----- Format in xarrays and save -----

    # The real predictions and projections
    times = firing_rates.times.values
    trial_counts = np.arange(len(true_trials))

    lda_predictions = xr.DataArray(predictions, dims=['trials','times'],
                                                coords=[trial_counts, times])
    lda_predictions = lda_predictions.assign_coords(true_trials=('trials',true_trials))
    lda_predictions = lda_predictions.assign_coords(targets=('trials',targets))
    
    filename = f'{subj}-{onset}-{content}-{session}_SVC_new.nc'
    lda_predictions.to_netcdf(os.path.join(path, filename), engine='h5netcdf')