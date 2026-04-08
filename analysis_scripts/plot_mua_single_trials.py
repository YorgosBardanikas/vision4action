"""Script that plots the single-trial and trial-average MUA of one example 
channel per area per monkey, illustrated in figure 3B."""

import mne
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from frites.stats.stats_nonparam import confidence_interval

# Load the MUA data
subj = 'enya'
onset = 'targ'
content = 'mua'
session_type = 'short12J' if subj == 'jazz' else 'short12E'

epochList, epochListBhv = [],[]
for t in [2,3,4]:
    epochList.append(utils.load_epochs(session_type, onset, t, content=content))
    epochListBhv.append(utils.load_epochs(session_type, onset, t, content='bhv'))
epochs = mne.concatenate_epochs(epochList, on_mismatch='ignore')
epochs_bhv = mne.concatenate_epochs(epochListBhv, on_mismatch='ignore')

epochs_bhv, epochs = utils.keep_1attempt_trials(epochs_bhv, epochs)
tgRank = epochs_bhv.metadata['Target Rank']
times = epochs.times*1000
areas = ['M1','7A']
if onset == 'targ': t1,t2 = 400,1300
elif onset == 'hand': t1,t2 = 400,900
clrs = ['teal','darkviolet','goldenrod']
plt.rcParams.update({'font.size': 14})
plt.subplots(2,2,sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10,10))
subs = [[1,3],[2,4]]

for a,area in enumerate(areas):

    # Select one example channel per area per monkey
    if subj == 'jazz':
        if area == 'M1': ch = 2
        elif area == '7A': ch = 119
    else: 
        if area == 'M1': ch = 15
        elif area == '7A': ch = 101

    ntr = len(epochs)//3
    epochs_np = epochs.get_data(picks=f'{area}-{ch}').squeeze()
    epochs_np = savgol_filter(epochs_np,101,2)
    epochs_ch = epochs_np[:,t1:t2]
    epochs_mean, epochs_conf = [],[]

    # Average across trials (confidence intervals represent the 95th percentile)
    for tg in [2,3,4]:
        epochs_tg = epochs_ch[tgRank==tg]
        epochs_mean.append(epochs_tg.mean(0)) 
        epochs_conf.append(confidence_interval(epochs_tg,axis=0,cis=95).squeeze())
    
    # Plot all single-trials
    subs_ = subs[a]
    plt.subplot(2,2,subs_[0])
    if subj == 'jazz' and area == '7A': v = 0.4
    else: v = 0.6
    plt.pcolormesh(times[t1:t2],np.arange(3*ntr),epochs_ch,
                   cmap=utils.parula(),vmin=-0.1,vmax=v)
    plt.axvline(0,color='k',linestyle='--',lw=3)
    plt.hlines([ntr-0.5, 2*ntr-0.5],times[t1],times[t2],color='w',linestyle='--',lw=4)
    plt.xlim([times[t1],times[t2]])
    plt.yticks([ntr-0.5, 2*ntr-0.5],[])

    # Plot the trial-average mua for the three targets
    plt.subplot(2,2,subs_[1])
    for m,(mua,conf) in enumerate(zip(epochs_mean,epochs_conf)):
        plt.plot(times[t1:t2], mua, color=clrs[m], lw=4)
        plt.fill_between(times[t1:t2], conf[0], y2=conf[1], color=clrs[m], alpha=0.2)
    plt.axvline(0,color='k',linestyle='--',lw=3)
    plt.xticks([-200,0,200,400,600],[])
    plt.yticks([-0.2,0,0.5],[])
    plt.gca().spines[['right','top']].set_visible(False)

plt.show()