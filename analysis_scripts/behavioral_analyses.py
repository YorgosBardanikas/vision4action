"""Script that loads the behavioral data, performs the behavioral analyses,
and plots the figures 2B, 2C, 2D.
"""

import utils
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import ttest_1samp
from frites.stats.stats_nonparam import confidence_interval


### ------ Helper functions ------

def load_hand_xy(epochs):
    """Return smoothed hand trajectory as (ntrials, 2, ntimes)."""
    h_x = epochs.get_data(picks=['Hand X-Position']).squeeze()
    h_y = epochs.get_data(picks=['Hand Y-Position']).squeeze()
    hand_x = savgol_filter(h_x,201,5)
    hand_y = savgol_filter(h_y,201,5)
    hand_xy = np.array([hand_x, hand_y]).transpose(1, 0, 2)
    return hand_xy


def get_target_xy(epochs, targXY):
    """Return (ntrials, 2) array of target positions for each trial."""
    currentTargetID = epochs.metadata['currentTargetID'].to_numpy()-1
    x_ = targXY[0, currentTargetID]
    y_ = targXY[1, currentTargetID]
    target_xy = np.stack((x_, y_), axis=1)
    return target_xy



### ------ Computing functions ------

def compute_initial_deviation(epochs, targXY):
    """
    Compute per-trial angle between hand position and target direction.
    Requires onset == 'hand'.
    Returns initial_deviation (ntrials,) in radians.
    """
    HT1, HT2 = 620, 820 # first 200 ms after movement onset

    epochs = utils.keep_1attempt_trials(epochs, None)
    hand_xy = load_hand_xy(epochs)
    # average x,y position across the time window
    hand_avg = hand_xy[..., HT1:HT2].mean(axis=-1)
    targ_xy = get_target_xy(epochs, targXY)

    n1 = np.linalg.norm(hand_avg, axis=1, keepdims=True)
    n2 = np.linalg.norm(targ_xy, axis=1, keepdims=True)
    dot = np.einsum('ij,ij->i', hand_avg / n1, targ_xy / n2)

    return np.arccos(dot)


def compute_directional_alignment(epochs, targXY):
    """
    Compute cosine similarity between hand trajectory direction and hand-to-target vector.
    Requires onset == 'targ'.
    Returns directional_alignment (ntrials, ntimes-1), target_rank.
    """
    HT1, HT2 = 320, 1220 # -300 ms to +600 ms around target onset

    epochs = utils.keep_1attempt_trials(epochs, None)
    target_rank = epochs.metadata['Target Rank'].to_numpy()

    # Load the hand positions for each single trial
    hand_xy = load_hand_xy(epochs)
    hand_xy = hand_xy[..., HT1:HT2]
    ntimes = hand_xy.shape[-1]

    # Get the x,y positions of the visual targets in the workspace
    targ_xy = get_target_xy(epochs, targXY)
    targ_xy_broadcast = np.repeat(targ_xy[..., np.newaxis], ntimes, axis=-1)

    # Find the vector starting from the hand position and pointing towards the 
    # current target for each timepoint
    hand_targ_vec = targ_xy_broadcast - hand_xy

    # Find the vector of the instantaneous hand trajectory direction
    hand_drct = np.diff(hand_xy, axis=-1)
    hand_drct = savgol_filter(hand_drct, 101, 3)

    # Find the magnitude of the vectors
    n1 = np.linalg.norm(hand_targ_vec, axis=1, keepdims=True)
    n2 = np.linalg.norm(hand_drct, axis=1, keepdims=True)

    # Compute the cosine similarity (dot product) between the two vectors
    directional_alignment = np.einsum('ijk,ijk->ik', 
                                    hand_targ_vec[...,:-1] / n1, 
                                    hand_drct / n2)
    
    return directional_alignment, target_rank


def _compute_null(targXY):
    """
    Compute the geometric null cosine similarity for peripheral targets
    in the hexagonal workspace (weighted mean over square and corner targets).
    """
    # Null for movements starting from a square target
    t_centered = targXY - targXY[:,[5]]
    a, b = t_centered[:,6], t_centered[:,1]
    dot_square = np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))

    # Null for movements starting from a lateral-corner target
    t_centered = targXY - targXY[:,[1]]
    a, b = t_centered[:,5], t_centered[:,3]
    dot_corners = np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))

    # Weighted mean: 4 square targets, 2 corner targets
    dot_total = (4*dot_square + 2*dot_corners)/6
    null = (1 + dot_total) / 2   # = 0.25 (instead of 0 for center-out movements)
    return null


def compute_eye_kinematics(epochs, subj):
    """
    Filter saturated trials and return smoothed eye velocity per target rank.
    Returns dict {target_rank: (ntrials, ntimes)} and epochs.times.
    """
    epochs = utils.keep_1attempt_trials(epochs, None)
    target_rank = epochs.metadata['Target Rank'].to_numpy()
    times = epochs.times
    eye_vel = epochs.get_data(picks=['Eye Velocity']).squeeze()
    eye_vel_filt = savgol_filter(eye_vel, 201, 2)

    # Discard full trials if the eye velocity is saturated even in only
    # one of the 2nd, 3rd or 4th targets.
    sat_thres = 100 if subj == 'jazz' else 200
    mask = np.zeros((3,len(epochs)//3), dtype=bool)

    for i,tg in enumerate([2,3,4]):
        # sum the eye velocity across time
        sums = eye_vel_filt[target_rank==tg].sum(-1)
        # keep only the indexes that are smalled than a saturation threshold
        mask[i] = sums < sat_thres
    # Keep trial indexes only if they are valid in all three targets
    valid_trials = np.where(mask[0] & mask[1] & mask[2])[0]

    eye_vel_tg = {tg: eye_vel_filt[target_rank==tg][valid_trials] 
                  for tg in [2,3,4]}
    return eye_vel_tg, times



### ------ Plotting functions ------

def plot_initial_deviation(epochs, targXY):
    """Plot cumulative distribution of initial deviation (Figure 2B)."""

    assert onset == 'hand', "plot_initial_deviation requires onset='hand'."

    initial_deviation = compute_initial_deviation(epochs, targXY)

    plt.figure()
    bin_counts, bin_edges, _ = plt.hist(initial_deviation, bins=100)
    plt.close()

    cumulative = np.cumsum(bin_counts) / bin_counts.sum()
    percent_below_45 = (initial_deviation < np.pi/4).mean()
    print(f'{percent_below_45 * 100:.1f}% of trials below π/4')

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(bin_edges[:-1], cumulative, color='k')
    ax.set_xticks([0,0.78],['0','π/4'])
    ax.set_yticks([0,percent_below_45,1],['0','','1'])
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax


def plot_directional_alignment(epochs, targXY):
    """Plot cosine similarity between hand direction and target vector (Figure 2C)."""

    assert onset == 'targ', "plot_directional_alignment requires onset='targ'."

    HT1, HT2 = 320, 1220 # -300 ms to +600 ms around target onset
    times = np.arange(HT1-619, HT2-620) # time labels for passing the xticks
    null = _compute_null(targXY)
    null_per_target = {2:0, 3:null, 4:null}
    colors = {2:'teal', 3:'darkviolet', 4:'goldenrod'}
    y  = {2:-0.8, 3:-0.85, 4:-0.9}
    directional_alignment, target_rank = compute_directional_alignment(epochs, targXY)

    fig = plt.figure()
    ax = plt.gca()
    for tg in [2,3,4]:
        da_tg = directional_alignment[target_rank==tg]
        mean_da_tg = da_tg.mean(axis=0)
        p = ttest_1samp(da_tg, null_per_target[tg], axis=0, alternative='greater')[1]
        pv = np.where(p < 0.05, y[tg], np.nan)
        conf = confidence_interval(da_tg, axis=0).squeeze()

        ax.plot(times, mean_da_tg, color=colors[tg], lw=3)
        ax.scatter(times, pv, s=3, color=colors[tg])
        ax.fill_between(times, conf[0], conf[1], color=colors[tg], alpha=0.3)

    ax.axvline(0, color='k',    linestyle='--')
    ax.axhline(0, color='grey', linestyle=':')
    ax.axhline(0.25, color='grey', linestyle=':')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([-200, 0, 200, 400, 600], [])
    ax.set_yticks([-1, 0, 0.25, 1], [])
    return fig, ax


def plot_eye_kinematics(epochs, subj):
    """Plot trial-averaged eye velocity traces (Figure 2D)."""
    
    assert onset == 'targ', "plot_directional_alignment requires onset='targ'."

    T1, T2 = 320, 1220 # -300 ms to +600 ms around target onset
    colors = {2: 'teal', 3: 'darkviolet', 4: 'goldenrod'}
    y = [0.02, 0.05, 0.08] if subj == 'jazz' else [0.05, 0.1, 0.15]
    eye_vel, times = compute_eye_kinematics(epochs, subj)

    # Average eye velocity
    fig = plt.figure()
    ax = plt.gca()
    for tg in [2,3,4]:
        avg  = eye_vel[tg].mean(axis=0)
        conf = confidence_interval(eye_vel[tg], axis=0, cis=95).squeeze()
        ax.plot(times[T1:T2], avg[T1:T2], color=colors[tg], linewidth=3)
        ax.fill_between(times[T1:T2], conf[0, T1:T2], conf[1, T1:T2],
                            color=colors[tg], alpha=0.3)

    ax.axvline(0, color='k', linestyle='--')
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6], [])
    ax.set_yticks(y, [])
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax



if __name__ == '__main__':

    subj  = 'jazz'
    onset = 'targ'
    session_type = 'short12J' if subj == 'jazz' else 'short12E'

    # Load epochs of behavior
    epochList = [utils.load_epochs(session_type, onset, targetID, content='bhv')
                 for targetID in [2, 3, 4]]
    epochs = mne.concatenate_epochs(epochList, on_mismatch='ignore')

    # Load target positions
    targXY = np.load(f'path_to_directory/targets_xy_positions_{subj}.npy')

    # Plot Figure 2B
    plot_initial_deviation(epochs, targXY)
    # Plot Figure 2C
    plot_directional_alignment(epochs, targXY)
    # Plot Figure 2D
    plot_eye_kinematics(epochs, subj)
    
    plt.show()