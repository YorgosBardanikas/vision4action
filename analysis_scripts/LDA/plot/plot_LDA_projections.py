
import numpy as np
import xarray as xr
from analysis_scripts import utils
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.patches import Polygon
from frites.stats.stats_nonparam import confidence_interval

subj = 'enya'
onset = 'targ'
content = 'mua'
rgr = 'goal'
area = 'M1'
plt.rcParams.update({'font.size': 14})
path = f'{utils.PATH}/LDA_6dirs/'
fig_dir = utils.PATH + '/Figures_revisions/LDA'
filename = f'{subj}-{onset}-{content}_LDA_LS{rgr}.nc'
lda = xr.open_dataset(f'{path}{filename}',engine='h5netcdf')
lda = lda['projections'].sel(areas=area)
lda['trials'] = lda.true_trials
trials = lda.trials.data
times = lda.times*1000
ntimes = times.data.size
nlds = lda.PCs.data.size
ntargs = 3 # three target ranks

# train: [top-right, right, bottom-right, bottom-left, left, top-left]
train_code_list = [[32,42],[12,22],[52,62],[112,122],[92,102],[72,82]]
# test: [top-right, bottom-right, bottom-left, top-left, bottom, top]
test_code_list3 = [[53,93],[33,103],[23,73],[13,123],[43,83],[63,113]]
test_code_list4 = [[44,124],[64,74],[34,114],[54,84],[14,94],[24,104]]
test_codes_list = [test_code_list3, test_code_list4]
ncl = len(test_codes_list)
ndirs = len(train_code_list)

train_code_flat = [i for train in train_code_list for i in train]
train_inds = np.where(np.isin(trials, train_code_flat))[0]
test_inds = np.where(np.isin(trials, train_code_flat, invert=True))[0]
train_proj = lda.isel(trials=train_inds)
test_proj = lda.isel(trials=test_inds)

# Plot the train projections across time
scores_all = np.zeros((nlds,ntargs,ndirs,ntimes))
clrs = ["#5245de","#3c8ac6","#a5bcda","#ec90b0","#ca575e","#c22d3f"]
clrs2 = ["#5245de","#a5bcda","#ec90b0","#c22d3f","#000000","#838383"]
t1,t2 = 400,1300
i = 1
plt.subplots(nlds,1,sharex=True)
for s,train in enumerate(train_proj):
    plt.subplot(nlds,1,i)

    for tc,train_codes in enumerate(train_code_list):
        inds = np.where(np.isin(train.trials, train_codes))[0]
        train_ = train.isel(trials=inds).data
        train_ = savgol_filter(train_,51,1)
        conf = confidence_interval(train_, axis=0).squeeze()
        lci, uci = conf[0,...], conf[1,...]
        train_avg = train_.mean(0)
        scores_all[s,0,tc,:] = train_avg
        plt.plot(times[t1:t2], train_avg[t1:t2], c=clrs[tc], lw=2)
        plt.fill_between(times[t1:t2], lci[t1:t2], y2=uci[t1:t2], 
                                    color=clrs[tc], alpha=0.15)

    plt.gca().spines[['right','top']].set_visible(False)
    plt.axvline(0,color='k',linestyle='--')
    plt.xticks([-200,0,200,400,600],[])
    plt.yticks([-5,0,5],['','0',''])
    i+=1

# Plot the test projections across time
i = 1
plt.subplots(nlds,ncl,sharex=True,sharey=True)
for ts,test in enumerate(test_proj):
    for tsc,test_codes in enumerate(test_codes_list):
        plt.subplot(nlds,ncl,i)
        for tc,codes in enumerate(test_codes[:-2]):
            inds = np.where(np.isin(test.trials, codes))[0]
            test_ = test.isel(trials=inds).data
            test_ = savgol_filter(test_,81,1)
            conf = confidence_interval(test_, axis=0).squeeze()
            lci, uci = conf[0,...], conf[1,...]
            test_avg = test_.mean(0)
            scores_all[ts,tsc+1,tc,:] = test_avg # tsc+1 bc 0 is filled with the 2nd target
            plt.plot(times[t1:t2], test_avg[t1:t2], c=clrs2[tc], lw=2)
            plt.fill_between(times[t1:t2], lci[t1:t2], y2=uci[t1:t2], 
                                        color=clrs2[tc], alpha=0.15)

        plt.gca().spines[['right','top']].set_visible(False)
        plt.axvline(0,color='k',linestyle='--')
        plt.xticks([-200,0,200,400,600],[])
        plt.yticks([-5,0,5],['','',''])
        i+=1

# Find the moments of maximum cross-condition variance
crossCondVar = scores_all.var(axis=(0,2))   # variance across the 2 LDs and 4 directions -> (ntargs,ntimes)
ccvMax_time = crossCondVar.argmax(axis=-1)  # argmax across times -> (ntargs)

# Find the points of max ccv in neural space coordinates
points = np.zeros((ntargs,ndirs,nlds))
for nc in range(nlds):
    for nt in range(ntargs):
        ccv_ = ccvMax_time[nt]
        points[nt,:,nc] = scores_all[nc,nt,:,ccv_]

# Plot the neural space
plt.figure(figsize=(6,6))
plt.axis(False)

# Plot the neural trajectory for the center-out movements
v = 850 if subj == 'jazz' else 900
for r in range(6):
    plt.plot(scores_all[0,0,r,700:v], scores_all[1,0,r,700:v], 
             c=clrs[r], lw=3) # ncomps, ntargs, ndirs, ntimes

# Plot the points of max ccv per target rank
ls = ['-','--',':']
for tg,points_tg in enumerate(points):
        
    if tg == 0: 
        ndirs = 6
        colors = clrs
    else: 
        ndirs = 4
        colors = clrs2

    for nd in range(ndirs):
        plt.scatter(points_tg[nd,0], points_tg[nd,1], s=200, c=colors[nd])

    if tg == 0:
        polygon = Polygon(points_tg, closed=True, fill=None, edgecolor='k', lw=3, ls=ls[tg])
        plt.gca().add_patch(polygon)
    else: 
        polygon = Polygon(points_tg[:-2], closed=True, fill=None, edgecolor='k', lw=3, ls=ls[tg])
        plt.gca().add_patch(polygon)

# for tg,points_tg in enumerate(points[1:]): # no 2nd target

#     for n in range(2):
#         nd = 4 + n
#         plt.scatter(points_tg[nd,0], points_tg[nd,1], s=200, c=clrs2[nd])

# plt.savefig(f'{fig_dir}/{subj}_{onset}_LDA_space_topbottom.svg')
plt.show()