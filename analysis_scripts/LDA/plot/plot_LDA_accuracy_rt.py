
import os
import numpy as np
import xarray as xr
from analysis_scripts import utils
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

subj = 'jazz'
onset = 'targ'
content = 'mua'
rgr = 'goal'
shuffles = True
areas = ['M1','7A']
plt.rcParams.update({'font.size': 14})
path = f'{utils.PATH}/LDA_2d/'
v = 0.35 if subj == 'jazz' else 0.4
session_type = 'short12J' if subj == 'jazz' else 'short12E'

# Load the center out data 80/20
filename = f'{subj}-{onset}-{content}_LDA_predictions_{rgr}8020.nc'
lda_predictions_cout = xr.open_dataarray(os.path.join(path, filename), engine='h5netcdf')

# Load the 3rd/4th peripheral targets
filename = f'{subj}-{onset}-{content}_LDA_LS{rgr}_rt.nc'
lda = xr.open_dataset(os.path.join(path, filename), engine='h5netcdf')
lda_predictions = lda['predictions']

if shuffles:
    # Load the permutations of all targets
    filename = f'{subj}-{onset}-{content}_LDA_LS{rgr}_shuffled.nc'
    shuffled_predictions = xr.open_dataarray(os.path.join(path, filename), engine='h5netcdf')

reaction_times = lda_predictions.reaction_times
true_trials = lda_predictions.true_trials
times = lda_predictions.times*1000
t1,t2 = 400,1400
clrs = ['darkviolet','violet','pink']
clrs = ['goldenrod',"#ecc65d","#b7b359"]

# Map true labels to test labels (same as train)
train_inds = [2,3,4,6]
remap2 = [[32,42],[52,62],[72,82],[112,122]]
remap3 = [[53,93],[33,103],[13,123],[23,73]]
remap4 = [[44,124],[64,74],[54,84],[34,114]]
plt.subplots(1,2,sharey=True,figsize=(10,5))

for a,area in enumerate(areas):

    plt.subplot(1,2,a+1)
    lda_34 = lda_predictions.sel(areas=area)
    remap = remap3 # select to plot target 3 or 4

    remap_flat = [r for rs in remap for r in rs]
    test_inds = np.where(np.isin(true_trials, remap_flat))[0]

    true_trials_ = true_trials[test_inds]
    reaction_times_ = reaction_times[test_inds]
    lda_predictions_ = lda_34.isel(trials=test_inds)
    
    if shuffles:
        true = true_trials_
        for ind, rm in zip(train_inds, remap):
            true = np.where(np.isin(true, rm), ind, true)

        lda_34_shuf = shuffled_predictions.sel(areas=area)
        lda_shuff_ = lda_34_shuf.isel(trials=test_inds)
        true_trials_34_s = true[None,:,None]
        accuracies_shuffled = (lda_shuff_ == true_trials_34_s).mean(axis=1)
        accuracies_shuffled = savgol_filter(accuracies_shuffled,81,1)
        l,u = accuracies_shuffled.min(0), accuracies_shuffled.max(0)
        plt.fill_between(times[t1:t2], l[t1:t2], y2=u[t1:t2], 
                                    color=clrs[0], alpha=0.15)
        
    # Filter reaction times
    idx_above = np.where(reaction_times_>150)[0]
    lda_predictions_above = lda_predictions_.isel(trials=idx_above)
    true_trials_above = true_trials_.isel(trials=idx_above)
    n_above = idx_above.size

    idx_below = np.where(reaction_times_<150)[0]
    rng = np.random.default_rng(seed=1)
    idx_below = rng.choice(idx_below, size=n_above, replace=False)
    lda_predictions_below = lda_predictions_.isel(trials=idx_below)
    true_trials_below = true_trials_.isel(trials=idx_below)

    for i,(lda_predictions__, true_trials__) in \
         enumerate(zip([lda_predictions_, lda_predictions_above, lda_predictions_below],
                        [true_trials_, true_trials_above, true_trials_below])):

        for ind, rm in zip(train_inds, remap):
            true_trials__ = np.where(np.isin(true_trials__, rm), ind, true_trials__)

        true_trials_34 = true_trials__[:,None]
        accuracies = (lda_predictions__ == true_trials_34).mean(axis=0)
        accuracies = savgol_filter(accuracies,81,1)
        
        plt.plot(times[t1:t2], accuracies[t1:t2], color=clrs[i], lw=3)

    plt.xticks([-200,0,200,400,600],[])
    plt.yticks([0.25,v],[])
    plt.gca().spines[['right','top']].set_visible(False)
    plt.axvline(0,color='k',linestyle='--')

# fig_dir = f'{utils.v4a_dir}/Figures/'
# fig_name = f'{subj}-LDA_accuracy_new.svg'
# plt.savefig(os.path.join(fig_dir, fig_name))
plt.show()