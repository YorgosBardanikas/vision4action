
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from analysis_scripts import utils

subj = 'jazz'
onset = 'targ'
content = 'units'
plt.rcParams.update({'font.size': 14})
path = f'{utils.PATH}/LDA_spikes/'
session_type = 'short12J' if subj == 'jazz' else 'short12E'
sessions = utils.load_session_group(session_type)
t1,t2 = 400,1400
clrs = ['teal','darkviolet','goldenrod']
tgs = [2,3,4]
dirs_train = [2,3,4,6]
remap = [[11,15],[9,16],[7,18],[8,13]]
sessions = ['j210215-land-003']

for session in sessions:

    # Load the data
    filename = f'{subj}-{onset}-{content}-{session}_SVC_new.nc'
    lda_predictions = xr.open_dataarray(f'{path}{filename}',engine='h5netcdf')
    # lda_predictions = lda_predictions['predictions']
    targets = lda_predictions.targets.values
    times = lda_predictions.times.values

    plt.figure()
    for tg in tgs:

        predictions = lda_predictions.sel(trials=targets==tg)
        true_trials_tg = predictions.true_trials.values

        for train, rm in zip(dirs_train, remap):
            true_trials_tg = np.where(np.isin(true_trials_tg, rm), train, true_trials_tg)

        true_trials_ = true_trials_tg[:,None]
        accuracy = (predictions == true_trials_).mean(axis=0)
        accuracy = savgol_filter(accuracy,51,2)

        plt.plot(times[t1:t2], accuracy[t1:t2], color=clrs[tg-2], lw=3)
        plt.gca().spines[['right','top']].set_visible(False)
        plt.axvline(0,color='k',linestyle='--')

plt.show()