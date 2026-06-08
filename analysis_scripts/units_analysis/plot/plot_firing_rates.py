
import os
import utils
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Load the units data
subj = 'jazz'
onset = 'targ'
content = 'units'
# session = 'j210219-land-001'
session = 'j210215-land-003'
v4a_dir = f'{utils.PATH}/{session}'
plt.rcParams.update({'font.size': 14})
t1,t2 = 400,1300
colors = ['#7aaacf','#9e9896','#e8c166','#ba805b']
best_units = []

for t in [2,3,4]:

    filename = f'{session}_{content}_{onset}{t}.nc'
    firing_rates = xr.load_dataarray(os.path.join(v4a_dir, filename))

    # sua_idx = firing_rates.unit_type == 'sua'
    # sua_rates = firing_rates.sel(units=sua_idx)
    sua_rates = firing_rates
    pmd_idx = sua_rates.units.str.contains('M1-PMd')
    pmd_rates = sua_rates.sel(units=pmd_idx)

    mean_frate = pmd_rates.mean(('trials','times'))
    idx = np.argsort(mean_frate)[::-1][:10] # 10 largest mean firing rate neurons
    best_units.append(idx)

    lstg = pmd_rates.trials.values
    directions, _ = utils.group_events(lstg, 'motor')
    pmd_rates['trials'] = directions
    mask_sort = directions.argsort()
    pmd_rates_sorted = pmd_rates.isel(trials=mask_sort)
    times = pmd_rates_sorted.times.values
    ntrials, nunits, _ = pmd_rates_sorted.shape

    if t==2: dirs_tg = [2,3,4,6]
    else: dirs_tg = [[11,15],[9,16],[7,18],[8,13]]

    c = 10
    r = math.ceil(nunits/c)
    plt.subplots(r,c,sharex=True,sharey=True,figsize=(40,20))

    for u_id in range(nunits):
        pmd_unit_rate = pmd_rates_sorted.isel(units=u_id)
        pmd_unit_trials = pmd_unit_rate.trials
        
        plt.subplot(r,c,u_id+1)
        plt.axvline(0, color='k', ls='--')
        plt.gca().spines[['right','top']].set_visible(False)

        for i,dr in enumerate(dirs_tg):

            unit_rate_dr = pmd_unit_rate.sel(trials=pmd_unit_trials.isin(dr)).mean('trials')
            unit_rate_dr_conv = uniform_filter1d(unit_rate_dr, size=50)
            plt.plot(times[t1:t2], unit_rate_dr_conv[t1:t2], color=colors[i])

plt.show()