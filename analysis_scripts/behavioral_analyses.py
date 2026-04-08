"""Script that loads the behavioral data, performs the behavioral analyses, 
and plots the figures 2B,2C,2D."""

import utils
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import ttest_1samp
from frites.stats.stats_nonparam import confidence_interval


def plot_initial_deviation(epochs):

    assert onset == 'hand', "Onset must be 'hand'."

    # Keep trials with no anticipation (target2_repetition == False)
    epochs = utils.keep_1attempt_trials(epochs, None)
    currentTargetID = epochs.metadata['currentTargetID'].to_numpy()-1

    # Load and concatenate the x,y positions of the hand
    hand_x = epochs.get_data(picks=['Hand X-Position']).squeeze()
    hand_y = epochs.get_data(picks=['Hand Y-Position']).squeeze()
    hand_x = savgol_filter(hand_x,201,5)
    hand_y = savgol_filter(hand_y,201,5)
    handSnglTrials_ = np.array([hand_x, hand_y]).transpose(1,0,2)  # ntrials, 2 (xy), ntimes
    ht1,ht2 = 620,820 # first 200 ms after MO
    handSnglTrials = handSnglTrials_[...,ht1:ht2] 
    handSnglTrials_avg = handSnglTrials.mean(axis=-1) # find average x,y position across the time window

    # Load the x,y positions of the visual targets in the workspace
    fname = f'targets_xy_positions_{subj}'
    targXY = np.load(f'{utils.v4a_dir}/{fname}.npy')
    x_, y_ = targXY[0, currentTargetID], targXY[1, currentTargetID]
    targXY_ = np.stack((x_, y_), axis=1)

    # Find the magnitude of the two vectors 
    # one pointing towards the hand position and the other towards the target
    n1 = np.linalg.norm(handSnglTrials_avg, axis=1, keepdims=True)
    n2 = np.linalg.norm(targXY_, axis=1, keepdims=True)
    # Compute the angle between the two vectors
    dot_prod = np.einsum('ij,ij->i', handSnglTrials_avg / n1, targXY_ / n2)
    initial_deviation = np.arccos(dot_prod)

    # Plot cumulative distribution of the initial deviation from straigth path
    plt.figure()
    bin_counts, bin_edges, _ = plt.hist(initial_deviation, bins=100)
    cumulative = np.cumsum(bin_counts) / bin_counts.sum()
    percent_below_45 = np.where(initial_deviation < np.pi/4)[0].size / initial_deviation.size
    print(percent_below_45*100)

    plt.figure()
    plt.plot(bin_edges[:-1], cumulative, color='k')
    plt.xticks([0,0.78],['0','π/4']), plt.yticks([0,percent_below_45,1],['0','','1'])
    plt.gca().spines[['right','top']].set_visible(False)


def plot_directional_alignment(epochs):

    assert onset == 'targ', "Onset must be 'hand'."

    epochs = utils.keep_1attempt_trials(epochs, None)
    target_rank = epochs.metadata['Target Rank'].to_numpy()
    currentTargetID = epochs.metadata['currentTargetID'].to_numpy()-1

    # Load and concatenate the x,y positions of the hand
    hand_x = epochs.get_data(picks=['Hand X-Position']).squeeze()
    hand_y = epochs.get_data(picks=['Hand Y-Position']).squeeze()
    hand_x = savgol_filter(hand_x,201,5)
    hand_y = savgol_filter(hand_y,201,5)
    handSnglTrials = np.array([hand_x, hand_y]).transpose(1,0,2)  # ntrials, 2 (xy), ntimes
    ht1,ht2 = 320,1220 # -300 to +600 ms around target onset
    handSnglTrials = handSnglTrials[...,ht1:ht2]
    ntimes = handSnglTrials.shape[-1]

    # Load the x,y positions of the visual targets in the workspace
    fname = f'targets_xy_positions_{subj}'
    targXY = np.load(f'{utils.v4a_dir}/{fname}.npy')
    x_, y_ = targXY[0, currentTargetID], targXY[1, currentTargetID]
    xy_ = np.stack((x_, y_), axis=1)
    targXY_broadcasted = np.repeat(xy_[...,np.newaxis], ntimes, axis=-1)
    # Find the vector starting from the hand position and pointing towards the 
    # current target for each timepoint
    targXY_SnglTrials = targXY_broadcasted - handSnglTrials

    # Find the vector of the instantaneous hand trajectory direction
    derivHand_SnglTrials = np.diff(handSnglTrials, axis=-1)
    derivHand_SnglTrials = savgol_filter(derivHand_SnglTrials,101,3)

    # Find the magnitude of the two vectors 
    n1 = np.linalg.norm(targXY_SnglTrials[...,:-1], axis=1, keepdims=True)
    n2 = np.linalg.norm(derivHand_SnglTrials, axis=1, keepdims=True)

    # Compute the cosine similarity (dot product) between the two vectors
    directional_alignment = np.einsum('ijk,ijk->ik', targXY_SnglTrials[...,:-1] / n1, 
                                                     derivHand_SnglTrials / n2)


    # Find the correct null hypothesis for the peripheral targets
    # taking into account the geometry of the workspace.
    # First find the null for a movement starting from a target in the square
    targXY_centered = targXY - targXY[:,[5]]
    a,b = targXY_centered[:,6], targXY_centered[:,1]
    a1,b1 = np.linalg.norm(a),np.linalg.norm(b)
    dot_square = np.dot(a/a1,b/b1)
    # Then find the null for a movement starting from a target in the lateral corners
    targXY_centered = targXY - targXY[:,[1]]
    a,b = targXY_centered[:,5], targXY_centered[:,3]
    a1,b1 = np.linalg.norm(a),np.linalg.norm(b)
    dot_corners = np.dot(a/a1,b/b1)
    # In the hexagon, 4 targets form the square and 2 are in the lateral corners,
    # so compute the weighted mean of the null dot product.
    dot_total = (4*dot_square + 2*dot_corners)/6
    null = (1 + dot_total)/2 # null is 0.25 (instead of 0 for center-out movements)

    # Plot the cosine similarity
    plt.figure()
    clrs = ['teal','darkviolet','goldenrod']
    v = [-0.8,-0.85,-0.9]
    times = np.arange(ht1-619,ht2-620)
    for tg in [2,3,4]:
        
        dir_al_tg = directional_alignment[target_rank==tg]
        mean_directional_alignment = dir_al_tg.mean(axis=0) # average across trials
        null_ = 0 if tg == 2 else null # null hypothesis for the first target: cosine similarity is 0
        p = ttest_1samp(dir_al_tg, null_, axis=0, alternative='greater')[1]
        pv = np.where(p < 0.05, v[tg-2], np.nan).squeeze()
        conf = confidence_interval(dir_al_tg, axis=0).squeeze()
        lci, uci = conf[0,...], conf[1,...]
        plt.plot(times, mean_directional_alignment, color=clrs[tg-2], lw=3)
        plt.scatter(times, pv, s=3, color=clrs[tg-2])
        plt.fill_between(times, lci, y2=uci, color=clrs[tg-2], alpha=0.3)

    plt.axvline(0,color='k',linestyle='--')
    plt.axhline(0,color='grey',linestyle=':')
    plt.axhline(0.25,color='grey',linestyle=':')
    plt.gca().spines[['right','top']].set_visible(False)
    plt.xticks([-200,0,200,400,600],[])
    plt.yticks([-1,0,0.25,1],[])


def plot_eye_kinematics (epochs):

    assert onset == 'targ', "Onset must be 'hand'."

    # Select the eye velocity data
    epochs = utils.keep_1attempt_trials(epochs, None)
    target_rank = epochs.metadata['Target Rank'].to_numpy()
    times = epochs.times
    channel = 'Eye Velocity'
    bhv_ = epochs.get_data(picks=[channel]).squeeze()
    bhvFilt = savgol_filter(bhv_,201,2)

    # Discard full trials where the eye velocity is saturated across time
    s = 100 if subj == 'jazz' else 200
    mask = np.zeros((3,len(epochs)//3), dtype=bool)
    for tg in [2,3,4]:
        sums = bhvFilt[target_rank==tg].sum(-1)
        mask[tg-2] = sums < s
    mask_tg = np.where(mask[0] & mask[1] & mask[2])[0]

    # Plot the eye velocity for the rest of the trials
    t1,t2 = 320,1220
    clrs = ['teal','darkviolet','goldenrod']
    plt.subplots(1,3,sharex=True,sharey=True)
    for tg in [2,3,4]:
        bhv_tg = bhvFilt[target_rank==tg][mask_tg]
        ntr = bhv_tg.shape[0]
        plt.subplot(1,3,tg-1)
        plt.pcolormesh(times[t1:t2], np.arange(ntr), bhv_tg[:,t1:t2], vmin=0, vmax=0.15)
    plt.axvline(0,color='k',linestyle='--')
    plt.gca().spines[['right','top']].set_visible(False)

    # Plot the average eye velocity
    yt = [0.02,0.05,0.08] if subj == 'jazz' else [0.05,0.1,0.15]
    plt.figure()
    for tg in [2,3,4]:
        bhv_tg = bhvFilt[target_rank==tg][mask_tg]
        bhvAvg = bhv_tg.mean(axis=0)
        conf = confidence_interval(bhv_tg,axis=0,cis=95).squeeze()
        plt.plot(times[t1:t2],bhvAvg[t1:t2],color=clrs[tg-2],linewidth=3)
        plt.fill_between(times[t1:t2], conf[0,t1:t2], y2=conf[1,t1:t2],
                         color=clrs[tg-2], alpha=0.3)
    plt.xticks([-0.2,0,0.2,0.4,0.6],[])
    plt.yticks(yt,[])
    plt.axvline(0,color='k',linestyle='--')
    plt.gca().spines[['right','top']].set_visible(False)


if __name__ == '__main__':

    # Load the data
    subj = 'jazz'
    onset = 'targ'
    session_type = 'short12J' if subj == 'jazz' else 'short12E'
    epochList = [utils.load_epochs(session_type, onset, targetID, content='bhv')
                for targetID in [2,3,4]]
    epochs = mne.concatenate_epochs(epochList, on_mismatch='ignore')

    # Plot Figure 2B
    plot_initial_deviation(epochs)
    # Plot Figure 2C
    plot_directional_alignment(epochs) 
    # Plot Figure 2D
    plot_eye_kinematics(epochs)
    plt.show()