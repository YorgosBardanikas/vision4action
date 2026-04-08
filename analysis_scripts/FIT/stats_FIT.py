"""Script to compute the significant clusters of FIT.
WfStats workflow is used from 
https://github.com/brainets/frites/tree/master/frites/workflow"""

import os
import utils
import xarray as xr
import numpy as np
from frites.workflow import WfStats
from scipy.signal import savgol_filter

# Load the true FIT and its permutations
subj = 'jazz'
onset = 'targ'
content = 'mua'
path = f'{utils.v4a_dir}/FIT/'
fit_filename = f'{subj}-{onset}-{content}-FIT.nc'
fit = xr.open_dataarray(os.path.join(path, fit_filename),engine='h5netcdf')
roi = fit.roi.data.tolist()
times = fit.times.data
fit = savgol_filter(fit.data,3,1)
perms_filename = f'{subj}-{onset}-{content}-FIT_perms.npy'
fit_perms = np.load(os.path.join(path, perms_filename))
fit_perms = savgol_filter(fit_perms,3,1)

# Perform cluster-based statistics
pvs = []
for fit_, fit_perms_ in zip(fit, fit_perms):
    fit_perms_ = fit_perms_.transpose(1,0,2) # (roi,perms,times)
    fit_stats = [item[np.newaxis,:] for item in fit_]
    fitp_stats = [item[:,np.newaxis,:] for item in fit_perms_]

    pv,tv = WfStats().fit(fit_stats, fitp_stats, inference='ffx', mcp='cluster')
    pv = xr.DataArray(pv.T, coords=[roi, times], dims=['roi','times'])
    pvs.append(pv)
pv_all = xr.concat(pvs, dim='groups')

# Save the p-values
pv_filename = f'{subj}-{onset}-{content}-FIT_pv.nc'
pv_all.to_netcdf(os.path.join(path, pv_filename),engine='h5netcdf')