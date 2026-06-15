"""Script that plots the single-trial and trial-average MUA of one example 
channel per area per monkey, illustrated in figure 3B."""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from analysis_scripts import utils
from scipy.signal import savgol_filter
from frites.stats.stats_nonparam import confidence_interval

# Load the MUA data
subj = 'jazz'
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
codes = epochs.events[:,2]
times = epochs.times*1000

# [top-right, right, bottom-right, bottom-left, left, top-left]
code_list2 = [[32,42],[12,22],[52,62],[112,122],[92,102],[72,82]]
# [top-right, bottom-right, bottom-left, top-left, bottom, top]
code_list3 = [[53,93],[33,103],[23,73],[13,123],[43,83],[63,113]]
# same as for code_list3
code_list4 = [[44,124],[64,74],[34,114],[54,84],[14,94],[24,104]]
code_lists = {2:code_list2, 3:code_list3, 4:code_list4}

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

    epochs_np = epochs.get_data(picks=f'{area}-{ch}').squeeze()
    epochs_np = savgol_filter(epochs_np,101,2)
    epochs_ch = epochs_np[:,t1:t2]
    epochs_mean, epochs_conf = [],[]
    ep_drct_list = []

    for tg in [2,3,4]:
        epochs_tg = epochs_ch[tgRank==tg]
        codes_tg = codes[tgRank==tg]

        # Sort trials based on directions within each target rank
        code_list_tg = code_lists[tg]

        for code_pair in code_list_tg:
            code_idx = np.isin(codes_tg, code_pair)
            ep_drct_list.append(epochs_tg[code_idx])
    
        # Average across trials within each target rank
        epochs_mean.append(epochs_tg.mean(0)) 
        epochs_conf.append(confidence_interval(epochs_tg,axis=0,cis=99).squeeze())
    
    epochs_sorted = np.concatenate(ep_drct_list, axis=0)

    # Plot all single-trials
    ntr = len(epochs)//3
    subs_ = subs[a]
    plt.subplot(2,2,subs_[0])
    plt.pcolormesh(times[t1:t2],np.arange(3*ntr),epochs_sorted,
                   cmap=utils.parula(),vmin=-0.1,vmax=0.6)
    plt.axvline(0,color='k',linestyle='--',lw=3)
    plt.hlines([ntr-0.5, 2*ntr-0.5],times[t1],times[t2],color='w',linestyle='--',lw=4)
    plt.xlim([times[t1],times[t2]])
    plt.yticks([ntr-0.5, 2*ntr-0.5],[])
    plt.gca().invert_yaxis()

    # Plot the trial-average mua for the three targets
    plt.subplot(2,2,subs_[1])
    for m,(mua,conf) in enumerate(zip(epochs_mean,epochs_conf)):
        plt.plot(times[t1:t2], mua, color=clrs[m], lw=4)
        plt.fill_between(times[t1:t2], conf[0], y2=conf[1], color=clrs[m], alpha=0.2)
    plt.axvline(0,color='k',linestyle='--',lw=3)
    plt.xticks([-200,0,200,400,600],[])
    plt.yticks([-0.2,0,0.5],[])
    plt.gca().spines[['right','top']].set_visible(False)

# fig_dir = f'{utils.PATH}/Figures_revisions/MUA'
# fig_name = f'{subj}_mua_single_trials_&_average.png'
# plt.savefig(os.path.join(fig_dir, fig_name), dpi=100)
plt.show()