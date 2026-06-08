
import numpy as np
import xarray as xr
from analysis_scripts import utils
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

subj = 'jazz'
onset = 'targ'
content = 'mua'
rgr = 'goal'
areas = ['M1','7A']
plt.rcParams.update({'font.size': 14})
path = f'{utils.v4a_dir}/LDA_2d/'
v = 0.35 if subj == 'jazz' else 0.4

# Load the center out data 80/20
filename = f'{subj}-{onset}-{content}_LDA_predictions_{rgr}8020.nc'
lda_predictions_cout = xr.open_dataarray(f'{path}{filename}',engine='h5netcdf')

# Load the 3rd/4th peripheral targets
filename = f'{subj}-{onset}-{content}_LDA_LS{rgr}.nc'
lda = xr.open_dataset(f'{path}{filename}',engine='h5netcdf')
lda_predictions = lda['predictions']

# Load the permutations of all targets
filename = f'{subj}-{onset}-{content}_LDA_LS{rgr}_shuffled.nc'
# shuffled_predictions = xr.open_dataarray(f'{path}{filename}',engine='h5netcdf')

true_trials = lda_predictions.true_trials
times = lda_predictions.times*1000
t1,t2 = 400,1400
clrs = ['teal','darkviolet','goldenrod']

# Map true labels to test labels (same as train)
train_inds = [1,2,3,4,5,6]
remap2 = [[32,42],[52,62],[72,82],[112,122]]
remap3 = [[53,93],[33,103],[13,123],[23,73]]
remap4 = [[44,124],[64,74],[54,84],[34,114]]
plt.subplots(1,2,sharey=True,figsize=(10,5))

for a,area in enumerate(areas):

    lda_2 = lda_predictions_cout.sel(areas=area)
    lda_34 = lda_predictions.sel(areas=area)
    # lda_34_shuf = shuffled_predictions.sel(areas=area)

    true_trials_2 = lda_2.trials.data[:,None]
    accuracies = (lda_2.data == true_trials_2).mean(axis=0)
    accuracies_cout = savgol_filter(accuracies,81,1)

    plt.subplot(1,2,a+1)
    plt.plot(times[t1:t2], accuracies_cout[t1:t2], color=clrs[0], lw=3)
    plt.gca().spines[['right','top']].set_visible(False)
    plt.axvline(0,color='k',linestyle='--')

    for i,remap in enumerate([remap2, remap3, remap4]):

        remap_flat = [r for rs in remap for r in rs]
        test_inds = np.where(np.isin(true_trials, remap_flat))[0]
        true_trials_ = true_trials[test_inds].data
        lda_predictions_ = lda_34.isel(trials=test_inds).data
        # shuffled_predictions_ = lda_34_shuf.isel(trials=test_inds).data

        for ind, rm in zip(train_inds, remap):
            true_trials_ = np.where(np.isin(true_trials_, rm), ind, true_trials_)

        true_trials_34 = true_trials_[:,None]
        accuracies = (lda_predictions_ == true_trials_34).mean(axis=0)
        accuracies = savgol_filter(accuracies,81,1)
        true_trials_34_s = true_trials_[None,:,None]
        # accuracies_shuffled = (shuffled_predictions_ == true_trials_34_s).mean(axis=1)
        # accuracies_shuffled = savgol_filter(accuracies_shuffled,81,1)
        # l,u = np.percentile(accuracies_shuffled, [1,99], axis=0)

        if i!=0: plt.plot(times[t1:t2], accuracies[t1:t2], color=clrs[i], lw=3)
        # plt.fill_between(times[t1:t2], l[t1:t2], y2=u[t1:t2], 
                                        # color=clrs[i], alpha=0.15)

    plt.xticks([-200,0,200,400,600],[])
    plt.yticks([0.25,v],[])

# fig_dir = f'{utils.v4a_dir}/Figures/'
# plt.savefig(f'{fig_dir}{subj}-LDA_accuracy_new.svg')
plt.show()