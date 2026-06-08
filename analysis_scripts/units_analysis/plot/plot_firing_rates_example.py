
import os
import utils
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from scipy.stats import sem

# Load the units data
subj = 'jazz'
onset = 'targ'
content = 'units'
session = 'j210215-land-003'
v4a_dir = f'{utils.PATH}/{session}'
path = f'{utils.PATH}/LDA_spikes/'
pmd_rates_all = []
dirs_train = [2,3,4,6]
dirs_test = [11,15,9,16,7,18,8,13]
best_units = []
n = 30 # amount of neurons with largest mean firing rate to keep
plt.rcParams.update({'font.size': 14})
t1,t2 = 400,1300
colors = ['#7aaacf','#9e9896','#e8c166','#ba805b']
clrs = ['teal','darkviolet','goldenrod']

for t in [2,3,4]:

    filename = f'{session}_{content}_{onset}{t}.nc'
    firing_rates = xr.load_dataarray(os.path.join(v4a_dir, filename))

    pmd_idx = firing_rates.units.str.contains('M1-PMd')
    pmd_rates = firing_rates.sel(units=pmd_idx)
    pmd_rates = xr.apply_ufunc(uniform_filter1d, pmd_rates, kwargs={'size':50, 'axis':-1})

    mean_frate = pmd_rates.var(('trials','times'))
    idx = np.argsort(mean_frate)[::-1][:n] 
    best_units.append(idx)
    pmd_rates_all.append(pmd_rates)

best_units_common = np.intersect1d(
                    np.intersect1d(best_units[0], best_units[1]), best_units[2])
pmd_rates_common = [arr.isel(units=best_units_common) for arr in pmd_rates_all]
times = pmd_rates_common[0].times.values
ntimes = times.size

# for tg,pmd_rates_tg in enumerate(pmd_rates_common):

#     nunits = pmd_rates_tg.units.size
#     lstg = pmd_rates_tg.trials.values
#     directions, _ = utils.group_events(lstg, 'motor')
#     pmd_rates_tg['trials'] = directions

#     if tg==0: dirs_tg = [2,3,4,6]
#     else: dirs_tg = [[11,15],[9,16],[7,18],[8,13]]

#     plt.subplots(5,6,sharex=True,sharey=True,figsize=(40,20))

#     for u_id in range(nunits):
#         pmd_unit_rate = pmd_rates_tg.isel(units=u_id)
#         pmd_unit_trials = pmd_unit_rate.trials
        
#         plt.subplot(5,6,u_id+1)
#         plt.axvline(0, color='k', ls='--')
#         plt.gca().spines[['right','top']].set_visible(False)

#         for i,dr in enumerate(dirs_tg):
#             unit_rate_dr = pmd_unit_rate.sel(trials=pmd_unit_trials.isin(dr)).mean('trials')
#             plt.plot(times[t1:t2], unit_rate_dr[t1:t2], color=colors[i])
#             plt.title(u_id)

unit_id = 24
pmd_xmpl = [xmpl.isel(units=unit_id) for xmpl in pmd_rates_common]
plt.subplots(1,3,sharex=True,sharey=True)

for tg,pmd_xmpl_tg in enumerate(pmd_xmpl):

    lstg = pmd_xmpl_tg.trials.values
    directions, _ = utils.group_events(lstg, 'motor')
    pmd_xmpl_tg['trials'] = directions
    neuron_trials = pmd_xmpl_tg.trials

    if tg==0: dirs_tg = [2,3,4,6]
    else: dirs_tg = [[11,15],[9,16],[7,18],[8,13]]

    plt.subplot(1,3,tg+1)
    plt.axvline(0, color='k', ls='--')
    plt.gca().spines[['right','top']].set_visible(False)

    for i,dr in enumerate(dirs_tg):
        neuron_xmpl_dr = pmd_xmpl_tg.sel(trials=neuron_trials.isin(dr))
        drct_mean = neuron_xmpl_dr.mean('trials')
        drct_sem = sem(neuron_xmpl_dr.values, axis=0)
        l,u = drct_mean-drct_sem, drct_mean+drct_sem
        
        plt.plot(times[t1:t2], drct_mean[t1:t2], color=colors[i])
        plt.fill_between(times[t1:t2], l[t1:t2], u[t1:t2], color=colors[i], alpha=0.2)

plt.show()