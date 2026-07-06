
import os
import numpy as np
import xarray as xr
from analysis_scripts import utils

def is_significant(data, window_size):
    kernel = np.ones(window_size, dtype=int) / window_size
    conv = np.convolve(data, kernel, mode='valid')
    return np.any(conv > 0.99)

# Parameters setting
subj = 'jazz'
onset = 'targ'
content = 'mua'
path = f'{utils.PATH}/MI/'
filename = f'{subj}-{onset}-{content}-MI_pv.nc'
pv = xr.open_dataarray(os.path.join(path,filename), engine='h5netcdf')
roi = pv.roi.data
window_size = 10 # x5 ms
pv_binary = np.where(pv < 0.05, 1, 0)

for i,pv_group in enumerate(pv_binary):

    signif_ch_names = []
    for pv_ch, roi_ in zip(pv_group, roi):

        if is_significant(pv_ch, window_size):
            signif_ch_names.append(roi_)

    n_7A = sum(ch.startswith('7A-') for ch in signif_ch_names)
    n_M1 = sum(ch.startswith('M1-') for ch in signif_ch_names)

    print(f'Target {i+2}')
    print(f'Number of 7A channels: {n_7A}')
    print(f'Number of PMd/M1 channels: {n_M1}')
    print(f'Total channel pairs: {n_7A*n_M1*2}') # 2 for both directions
