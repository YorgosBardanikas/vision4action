
import os
import numpy as np
import xarray as xr
from analysis_scripts import utils
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from itertools import permutations

# Load the units data
subj = 'jazz'
onset = 'targ'
content = 'units'
session = 'j210215-land-003'
# session = 'j210216-land-001'
v4a_dir = f'{utils.PATH}/{session}'
path = f'{utils.PATH}/LDA_spikes/'
pmd_rates_all = []
dirs_train = [2,3,4,6]
dirs_test = [11,15,9,16,7,18,8,13]
best_units = []
n = 30 # amount of neurons with largest firing rate features to keep
nboots = False
plt.rcParams.update({'font.size': 14})
t1,t2 = 400,1300
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

# unit_ids 22,24
m = best_units_common.size

for unit_id in [22,24]:
    trial_vector = pmd_rates_common[0].isel(units=unit_id).sel(times=slice(100,300)).mean('times')
    unit_name = trial_vector.units.values

    lstg = trial_vector.trials.values
    directions, _ = utils.group_events(lstg, 'motor')
    trial_vector['trials'] = directions
    dirs = [2,3,4,6]
    remap = [[11,15],[9,16],[7,18],[8,13]]
    ndirs = len(dirs)

    tuning_vector = np.zeros(ndirs)
    for i,d in enumerate(dirs):
        tuning_vector[i] = float(trial_vector.sel(trials=d).mean('trials').values)

    tuning_vec = tuning_vector - tuning_vector.mean()

    plt.figure()
    plt.axvline(0, color='k', ls='--', lw=3)
    plt.gca().spines[['right','top']].set_visible(False)

    for tg,pmd_rates_tg in enumerate(pmd_rates_common):

        neuron_xmpl = pmd_rates_tg.isel(units=unit_id)
        lstg = neuron_xmpl.trials.values
        directions, _ = utils.group_events(lstg, 'motor')
        
        for dr, rm in zip(dirs, remap):
            directions = np.where(np.isin(directions, rm), dr, directions)

        neuron_xmpl['trials'] = directions


        # Compute correlation between tuning vector template 
        # and the unit activity vector
        unit_vector = np.zeros((ndirs, ntimes))
        for i,d in enumerate(dirs):
            unit_vector[i] = neuron_xmpl.sel(trials=d).mean('trials')

        unit_vec = unit_vector - unit_vector.mean(axis=0, keepdims=True)

        num = np.sum(tuning_vec[:, None] * unit_vec, axis=0)
        denom = np.sqrt(np.sum(tuning_vec**2) * np.sum(unit_vec**2, axis=0))
        r = num / denom
        r = uniform_filter1d(r, size=100)
        plt.plot(times[t1:t2], r[t1:t2], color=clrs[tg], lw=4)

        # Permutations for estimating the chance level
        r_perm = []
        for perm in permutations(range(ndirs)):
            num_ = np.sum(tuning_vec[:, None] * unit_vec[list(perm)], axis=0)
            r_p = num_ / denom # denom can be the same since its independent
            r_perm.append(uniform_filter1d(r_p, size=100))

        mean_perm = np.mean(r_perm, axis=0)
        std_perm = np.std(r_perm, axis=0, ddof=1)
        low, up = mean_perm - std_perm, mean_perm + std_perm
        plt.fill_between(times[t1:t2], low[t1:t2], up[t1:t2], color=clrs[tg], alpha=0.2)
        plt.yticks([-1,-0.5,0,0.5,1])


        # Bootstraps for estimating variability of the correlation
        if type(nboots) is int:
            
            r_boot = np.zeros((nboots, ntimes))
            for b in range(nboots):

                unit_vector_bt = np.zeros((ndirs, ntimes))

                for i,d in enumerate(dirs):
                    neuron_d = neuron_xmpl.sel(trials=d)
                    ntr_d = neuron_d.trials.size
                    boot_idx = np.random.choice(ntr_d, size=ntr_d, replace=True)
                    neuron_d_boot = neuron_d.isel(trials=boot_idx)
                    unit_vector_bt[i] = neuron_d_boot.mean('trials')

                unit_vec_bt = unit_vector_bt - unit_vector_bt.mean(axis=0, keepdims=True)

                num = np.sum(tuning_vec[:, None] * unit_vec_bt, axis=0)
                denom = np.sqrt(np.sum(tuning_vec**2) * np.sum(unit_vec_bt**2, axis=0))
                r_ = num / denom
                r_boot[b] = uniform_filter1d(r_, size=100)

            low, high = np.percentile(r_boot, [2.5,97.5], axis=0)
            plt.fill_between(times[t1:t2], low[t1:t2], high[t1:t2], color=clrs[tg], alpha=0.2)

    fig_dir = f'{utils.PATH}/Figures_revisions/SUA'
    fig_name = f'tuning_decoding_{session}_unit_{unit_name}.svg'
    plt.savefig(os.path.join(fig_dir, fig_name)), plt.close()
# plt.show()