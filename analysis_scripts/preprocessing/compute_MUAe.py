"""
Script to compute the MUA envelope according to the method presented in
E. Stark and M. Abeles, J Neurosci., 2007, doi:10.1523/JNEUROSCI.1321-07.2007.

"""

import os
import neo
import pickle
import numpy as np
from frites.io import logger
from mne.filter import filter_data
from scipy.signal import decimate
from analysis_scripts.utils import load_session_group

SUBJ = 'jazz'
parent_dir = 'parent_directory'
path_to_save = 'path_to_save'
session_type = 'short12J' if SUBJ == 'jazz' else 'short12E'
session_group = load_session_group(session_type)

for session in session_group:
    MUAe = {}
    for implant_site in ['visual','motor']:
        utah = implant_site[0]
        filename = f'{session[:-3]}{utah}{session[-4:]}.ns6'
        logger.info(f"Loading the data file {filename}...")
        io = neo.io.BlackrockIO(f'{parent_dir}/{session}/{filename}') # load Blackrock ns6 files
        block = io.read_block()
        logger.info("...done.")
        segment = block.segments[-1]
        analogSignal = segment.analogsignals[-1].T
        sfreq = int(analogSignal._sampling_rate)
        del block, segment
        data1kHzAll = []

        for i in range(3): # 3 chunks of channels
            logger.info(f'Loop {i+1}')
            # Split the channels into 3 groups (due to computational cost)
            if i == 0: m,n = 0, int(analogSignal.shape[0]/3)
            elif i == 1: m,n = int(analogSignal.shape[0]/3), int(2*analogSignal.shape[0]/3)
            else: m,n = int(2*analogSignal.shape[0]/3), None
            rawData = np.array(analogSignal, dtype=float)[m:n]
            # High-pass filter the raw data
            lfreq, hfreq = 600, 6000 # Hz
            dataHighPass = filter_data(rawData, sfreq, lfreq, hfreq, method='iir',verbose='ERROR')
            del rawData
            # Z-score the high-pass data
            dHP_mean = dataHighPass.mean(1, keepdims=True)
            dHP_std = dataHighPass.std(1, keepdims=True)
            dataHP = (dataHighPass - dHP_mean) / dHP_std
            del dataHighPass
            # Clip spikes in the 30kHz filtered data
            dataHPclip = np.clip(dataHP, -2, 2)
            del dataHP
            # Rectify by squaring and then taking the square root
            dataHPclip = dataHPclip**2
            # Instead of downsampling by 30 (bad use), downsample by 10 and then 3
            data3khz = decimate(dataHPclip, 10)
            del dataHPclip
            data1khz = decimate(data3khz, 3)
            del data3khz
            data1khzroot = np.sqrt(np.clip(data1khz, 0, None))
            # Save in a list
            data1kHzAll.append(data1khzroot)
        MUAe[implant_site] = np.concatenate((data1kHzAll[0], data1kHzAll[1], data1kHzAll[2]))
        del analogSignal

    # Make directory and save
    try: os.mkdir(os.path.join(path_to_save, session))
    except FileExistsError: pass
    filename = f'{session}_MUAe.pkl'
    with open(f'{path_to_save}/{session}/{filename}','wb') as handle:
        pickle.dump(MUAe, handle)
    logger.info(f'MUAe is saved in the file: "{filename}".\n')
    