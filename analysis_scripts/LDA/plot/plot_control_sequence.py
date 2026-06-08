import utils
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

subj = 'jazz'
onset = 'targ'
content = 'mua'
rgr = 'goal'
areas = ['M1','7A']
plt.rcParams.update({'font.size': 14})
path = f'{utils.PATH}/LDA_2d/'
v = 0.35 if subj == 'jazz' else 0.4

# Load 
filename = f'{subj}-{onset}-{content}_LDA_control_sequence.nc'
lda = xr.open_dataarray(f'{path}{filename}',engine='h5netcdf')

# Load the permutations of all targets
filename = f'{subj}-{onset}-{content}_LDA_shuffled_control_sequence.nc'
shuffled_predictions = xr.open_dataarray(f'{path}{filename}',engine='h5netcdf')

true_trials = lda.trials
times = lda.times*1000
t1,t2 = 400,1400

plt.subplots(1,2,sharey=True,figsize=(10,5))

for a,area in enumerate(areas):

    lda_ = lda.sel(areas=area)
    lda_shuf = shuffled_predictions.sel(areas=area)

    true_trials_ = lda_.trials.data[:,None]
    accuracies = (lda_.data == true_trials_).mean(axis=0)
    accuracies = savgol_filter(accuracies,51,2)

    true_trials_ = lda_.trials.data[None,:,None]
    accuracies_shuffled = (lda_shuf.data == true_trials_).mean(axis=1)
    accuracies_shuffled = savgol_filter(accuracies_shuffled,51,2)
    l,u = np.percentile(accuracies_shuffled, [1,99], axis=0)
    
    plt.subplot(1,2,a+1)
    plt.plot(times[t1:t2], accuracies[t1:t2], color='k', lw=3)
    plt.fill_between(times[t1:t2], l[t1:t2], y2=u[t1:t2], 
                                        color='k', alpha=0.15)
    plt.gca().spines[['right','top']].set_visible(False)
    plt.axvline(0,color='k',linestyle='--')
    plt.xticks([-200,0,200,400,600],[])
    # plt.yticks([0.25,v],[])

plt.show()