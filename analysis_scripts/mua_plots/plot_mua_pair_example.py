
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from analysis_scripts import utils
from frites.stats.stats_nonparam import confidence_interval

# Parameters setting
subj = 'jazz'
onset = 'targ'
content = 'mua'
targetID = 2
session_type = 'short12J' if subj == 'jazz' else 'short12E'
plt.rcParams.update({'font.size': 14})
fig_dir = f'{utils.PATH}/Figures_revisions/'

epochs = utils.load_epochs(session_type, onset, targetID, content=content)
epochs_bhv = utils.load_epochs(session_type, onset, targetID, content='bhv') 

directions, _ = utils.group_events(epochs.events[:,2],'motor')
times = epochs.times*1000
if onset == 'targ': t1,t2 = 400,1300
elif onset == 'hand': t1,t2 = 400,900

drcts = [2,3,6,4]
clrs = ['#7aaacf','#9e9896','#e8c166','#ba805b']
channels = [['M1-65','7A-98'],['7A-30','M1-96']]

for i,ch_pair in enumerate(channels):

    plt.subplots(2,1,sharex=True,sharey=True)

    for c,ch in enumerate(ch_pair):

        epochs_np = epochs.get_data(picks=ch).squeeze()
        epochs_np = savgol_filter(epochs_np,101,2)
        epochs_ch = epochs_np[:,t1:t2]
        epochs_drct, epochs_mean, epochs_conf = [],[],[]

        for drct in drcts:
            mua = epochs_ch[directions==drct]
            epochs_drct.append(mua)
            epochs_mean.append(mua.mean(0))
            epochs_conf.append(confidence_interval(mua,axis=0,cis=99).squeeze())

        # Plot the trial-average mua
        plt.subplot(2,1,c+1)
        for m,(mua,conf) in enumerate(zip(epochs_mean,epochs_conf)):
            plt.plot(times[t1:t2], mua, color=clrs[m], lw=3)
            plt.fill_between(times[t1:t2], conf[0], y2=conf[1], color=clrs[m], alpha=0.2)
        plt.axvline(0,color='k',linestyle='--',lw=3)
        plt.xticks([-200,0,200,400,600],[])
        plt.yticks([-0.2,0,0.5],[])
        plt.gca().spines[['right','top']].set_visible(False)

    fig_file = f'{subj}_fit_mua_pair{i+1}.svg'
    # plt.savefig(os.path.join(fig_dir, fig_file)), plt.close()
plt.show()