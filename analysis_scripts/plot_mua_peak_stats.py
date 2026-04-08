"""Script to compute the statistics of single-trial peak features 
(amplitude and latency) and to plot the figures 3C and 3D."""

import mne
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def perm_test(x, y, n_perms=1000, stat=np.mean):
    """Function to compute statistical significance through permutation testing.
    Purely based on empirical data, and requires no assumptions.
    It is preferred when x and y are related but not paired (equal sizes)"""

    diff = np.abs(stat(x)-stat(y)) # compute true difference
    pooled = np.concatenate([x, y]) # pool all data
    count = 0
    for n in range(n_perms):
        rng = np.random.default_rng(n)
        rng.shuffle(pooled) # shuffle pooled data
        # resplit data into two groups of same length as original x,y
        x_perm = pooled[:len(x)] 
        y_perm = pooled[len(x):]
        diff_perm = np.abs(stat(x_perm)-stat(y_perm)) # compute shuffled difference
        # if permuted > true difference, count the event
        if diff_perm >= diff:
            count += 1
    # The p-values show the probability that the permuted difference  
    # is higher than the real. Small p-values show that the true 
    # difference is significantly larger than differences obtained by chance.
    pv = count / n_perms
    return pv


if __name__ == '__main__':

    # Load the MUA
    subj = 'enya'
    onset = 'targ'
    content = 'mua'
    session_type = 'short12J' if subj == 'jazz' else 'short12E'
    modes = ['max','lat']

    epochList, epochListBhv = [],[]
    for tt in [2,3,4]:
        epochList.append(utils.load_epochs(session_type, onset, tt, content=content))
        epochListBhv.append(utils.load_epochs(session_type, onset, tt, content='bhv'))
    epochs = mne.concatenate_epochs(epochList, on_mismatch='ignore')
    epochs_bhv = mne.concatenate_epochs(epochListBhv, on_mismatch='ignore')

    epochs_bhv, epochs = utils.keep_1attempt_trials(epochs_bhv, epochs)
    tgRank = epochs_bhv.metadata['Target Rank']
    times = epochs.times*1000
    areas = ['M1','7A']
    clrs = ['teal','darkviolet','goldenrod']
    t11, t22 = 500, 1100  # -200, 400 around target onset
    plt.rcParams.update({'font.size': 14})
    f1,ax1 = plt.subplots(1,2,sharey=True,figsize=(8,4))
    f2,ax2 = plt.subplots(1,2,sharey=True,figsize=(8,4))
    ax12,verts = [ax1,ax2],[True,False]

    for a,area in enumerate(['M1','7A']):

        chans = mne.pick_channels_regexp(epochs.ch_names,f'{area}-')
        epochs_all = epochs.get_data(picks=chans)
        epochs_all = savgol_filter(epochs_all,101,2)
        peak_max, peak_lat = [],[]

        # For each target and each channel, calculate the amplitude and latency 
        # of the peak for each single trial
        for tg in [2,3,4]:
            
            epochs_tg = epochs_all[tgRank==tg][...,t11:t22]
            peak_max.append(epochs_tg.max(-1)) # max across times
            lat_inds = epochs_tg.argmax(-1)+t11 # argmax across times
            peak_lat.append(times[lat_inds])
        
        # Average across trials so each channel has a representative amplitude and latency
        peak_max_mean = [p.mean(0) for p in peak_max]
        peak_lat_mean = [p.mean(0) for p in peak_lat]

        for p,peak_stat in enumerate([peak_max_mean, peak_lat_mean]):

            # Plot the mean peak statistic for each channel as a dot behind
            # the boxplots with some jitter on the x-axis for better visualization
            ax_list,vert = ax12[p],verts[p]
            n = peak_stat[0].size
            np.random.seed(12)
            for i, stat in enumerate(peak_stat):
                jittered = np.full_like(stat, i+1, dtype=float) + np.random.uniform(-0.1, 0.1, size=n)
                if vert: ax_list[a].scatter(jittered, stat, alpha=0.2, color=clrs[i], s=20, zorder=2)
                else: ax_list[a].scatter(stat, jittered, alpha=0.2, color=clrs[i], s=20, zorder=2)
                
            # Plot the boxplots
            boxes = ax_list[a].boxplot(peak_stat, vert=vert, showfliers=False, 
                                       patch_artist=True, zorder=1) 
            if vert: 
                ax_list[a].set_xticks([])
                if subj == 'jazz': ax_list[a].set_yticks([0.3,0.5,0.7])
                elif subj == 'enya': ax_list[a].set_yticks([0.25,0.5,0.75,1,1.25])
                ax_list[a].spines[['right','bottom','top']].set_visible(False)
            else: 
                ax_list[a].axvline(0,color='k',linestyle='--')
                ax_list[a].set_xticks([-200,0,200,400])
                ax_list[a].set_yticks([])
                ax_list[a].spines[['right','left','top']].set_visible(False)
            
            # Style the boxplots
            lw=3
            for patch, color in zip(boxes['boxes'], clrs):
                patch.set(facecolor='w', edgecolor=color, linewidth=lw)
            for med, clr in zip(boxes['medians'], clrs):
                med.set(color=clr, linewidth=lw, linestyle='-')
            for i, cap in enumerate(boxes['caps']):
                if i in [0,1]: cap.set(color=clrs[0], linewidth=lw)
                elif i in [2,3]: cap.set(color=clrs[1], linewidth=lw)
                else: cap.set(color=clrs[2], linewidth=lw)
            for i, w in enumerate(boxes['whiskers']):
                if i in [0,1]: w.set(color=clrs[0], linewidth=lw, linestyle='--')
                elif i in [2,3]: w.set(color=clrs[1], linewidth=lw, linestyle='--')
                else: w.set(color=clrs[2], linewidth=lw, linestyle='--')

            # Compute the pvalues pairwise across targets and correct 
            # for multiple comparisons (3 tests)
            # Significant pvalues are plotted as stars in the figures.
            id1, id2 = [0,0,1], [1,2,2] # pairwise combinations
            for i1,i2 in zip(id1,id2):

                pv = perm_test(peak_stat[i1],peak_stat[i2])
                pv = np.clip(3*pv,0,1) # bonferroni correction
                print(f'{area}, {modes[p]}, {i1}-{i2}: {pv}')

    plt.show()