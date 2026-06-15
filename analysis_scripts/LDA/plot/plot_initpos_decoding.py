"""Script to plot the decoding of initial position effects."""

import os
import numpy as np
import xarray as xr
from analysis_scripts import utils
import matplotlib.pyplot as plt

subj = 'enya'
onset = 'hand'
content = 'mua'
area = 'M1'
shuffle = True
path = f'{utils.PATH}/LDA_6dirs/initpos/'
labels = ['left','right','top','bottom'] # which side was tested
colors = ["#0C2A5C","#0B5D26","#8D710E","#641028"]
ntests = len(labels)
plt.rcParams.update({'font.size': 14})
predicted_trials, shuffled_predicted_trials = [],[]

for label in labels:
    filename = f'{subj}-{onset}-{content}_LDA_predictions_test_{label}.nc'
    pred = xr.open_dataarray(os.path.join(path, filename),
                             engine='h5netcdf').sel(areas=area).T
    times = pred.timebins*1000
    ntimes = times.size
    predicted_trials.append(pred.data)

    if shuffle: 
        filename = f'{subj}-{onset}-{content}_LDA_predictions_shuffled_{label}.nc'
        shuf_pred = xr.open_dataarray(os.path.join(path, filename), 
                                      engine='h5netcdf').sel(areas=area)
        shuffled_predicted_trials.append(shuf_pred.transpose('perms','trials','timebins').data)
        nperms = shuf_pred.perms.size

# Calculate the accuracies of real and shuffled decoders, by testing predicted labels
# [center-out, peripheral] -> [100,101] against the peripheral movement codes (101).
accuracies = np.zeros((ntests,ntimes))
for i,predictions in enumerate(predicted_trials):
    peripheral_code = 101*np.ones_like(predictions)
    accuracies[i] = (predictions == peripheral_code).mean(axis=0)
accuracy = accuracies.mean(0)
acc_std = accuracies.std(0)
l_acc, u_acc = accuracy-(acc_std/2), accuracy+(acc_std/2)

if shuffle:
    shuffled_accuracies = np.zeros((ntests,nperms,ntimes))
    for j,shuffled_predictions in enumerate(shuffled_predicted_trials):
        peripheral_code = 101*np.ones_like(shuffled_predictions)
        shuffled_accuracies[j] = (shuffled_predictions == peripheral_code).mean(axis=1)
    l,m,u = np.percentile(shuffled_accuracies, [0.5,50,99.5], axis=(0,1))

# Plot the accuracy across all 4 movement directions
plt.figure(figsize=(5,4))
# for i,accuracy in enumerate(accuracies):
#     plt.plot(times, accuracy, color=colors[i])
#     plt.scatter(times, accuracy, s=10, c=colors[i])
plt.plot(times, accuracy, color='navy')
plt.scatter(times, accuracy, s=10, c='navy')
plt.fill_between(times, l_acc, y2=u_acc, color='navy', alpha=0.2)

if shuffle: 
    plt.plot(times, m, color='k', alpha=0.5)
    plt.fill_between(times, l, y2=u, color='k', alpha=0.2)
plt.xticks([-200,-100,0],[])
plt.yticks([0,0.25,0.5,0.75,1], ['0','0.25','0.5','0.75','1'])
plt.gca().spines[['right','top']].set_visible(False)

fig_dir = f'{utils.PATH}/Figures_revisions/LDA'
fig_name = f'{subj}-{area}_initpos_decoding_accuracy.svg'
plt.savefig(os.path.join(fig_dir, fig_name))
# plt.show()