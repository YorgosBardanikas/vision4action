"""
Collection of functions that are used in the 
preprocessing script 'generate_epochs.py'.

"""

import numpy as np


def windows (onset, k=80):
    """Time around the onset to take into account.

    Parameters
    ----------
    onset : str
            It defines the onset on which the epochs are aligned. 
            Must be either 'targ', 'eye' or 'hand'. 
    k : int | default: 80
            Time index that will be subtracted after the multitaper, 
            to ignore edge artifact.               
    """

    # Time in samples (or in msec) 
    if onset == 'targ': w1,w2 = 620, 921
    elif onset == 'hand': w1,w2 = 620, 621
    elif onset == 'eye': w1,w2 = 620, 621
    return int(w1+k), int(w2+k)



def calculate_vel (anasig_behav, event_onsets, onset, st):
    """
    Calculate the eye or hand velocity.

    Parameters
    ----------
    anasig_behav : neo.AnalogSignal
            The behavioral analogsignal that contains the x,y coordinates of
            the eye/hand movements.
    event_onsets : ndarray
            The timestamps of the events on which the epochs will be aligned.
    onset : str
            It defines the onset on which the epochs are aligned. 
            Must be either 'targ', 'eye' or 'hand'.                  
    st : str
            Can be either 'Eye' or 'Hand'.

    Returns
    -------
    velocity : ndarray
            The eye or hand velocity aligned on different onsets.      
    """ 
    w1, w2 = windows(onset)
    ntimes = w1+w2
    ntrials = event_onsets.shape[0]
    anasig_ch_names = anasig_behav.array_annotations['channel_names']

    if st == 'Eye': labels = ['EyeXcm','EyeYcm']
    elif st == 'Hand': labels = ['HandXcm','HandYcm']
    else: raise ValueError('The last input argument must be either "Eye" or "Hand".')

    maskx = anasig_ch_names == labels[0]
    masky = anasig_ch_names == labels[1]
    data_x = np.array(anasig_behav[:,maskx]).squeeze()
    data_y = np.array(anasig_behav[:,masky]).squeeze()
    vel = np.zeros((ntrials, ntimes))

    for tr,ev_on in enumerate(event_onsets):

        x = data_x[ev_on-w1 : ev_on+w2+1].squeeze()  # shape (ntimes+1)
        y = data_y[ev_on-w1 : ev_on+w2+1].squeeze()
        vel[tr] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    return vel



def calculate_hand_eye_dist (hand_x, hand_y, eye_x, eye_y, event_onsets, onset):
    """
    Calculate the hand-eye distance.

    Parameters
    ----------
    hand_x, hand_y: ndarray
            The x,y positions of the hand.
    eye_x, eye_y: ndarray
            The x,y positions of the eye.
    event_onsets : ndarray
            The timestamps of the events on which the epochs will be aligned.
    onset : str
            It defines the onset on which the epochs are aligned. 
            Must be either 'targ', 'eye' or 'hand'.                  

    Returns
    -------
    he_dist : ndarray
            The hand-eye distance aligned on the different onsets.      
    """                                 
    w1, w2 = windows(onset)
    ntimes = w1+w2
    ntrials = event_onsets.shape[0]
    he_dist = np.zeros((ntrials, ntimes))

    for tr,on in enumerate(event_onsets):

        hx, hy = hand_x[on-w1: on+w2], hand_y[on-w1: on+w2]
        ex, ey = eye_x[on-w1: on+w2], eye_y[on-w1: on+w2]
        he_dist[tr] = np.sqrt((hx-ex)**2 + (hy-ey)**2)

    return he_dist



def calculate_bhv_targ_dist (x, y, xTarg, yTarg, currentTargetID, 
                                            event_onsets, onset):
    """
    Calculate the hand-target (or eye-target) distance.

    Parameters
    ----------
    x, y: ndarray
            The x,y positions of the hand/eye.
    xTarg, yTarg: ndarray
            The x,y positions of the current target.
    event_onsets : ndarray
            The timestamps of the events on which the epochs will be aligned.
    currentTargetID : ndarray
            The id of the target that appeared for each epoch.            
    onset : str
            It defines the onset on which the epochs are aligned. 
            Must be either 'targ', 'eye' or 'hand'.                  

    Returns
    -------
    bhvt_dist : ndarray
            The hand (or eye)-target distance aligned on the different onsets.      
    """   
    w1, w2 = windows(onset)
    ntimes = w1+w2
    ntrials = event_onsets.shape[0]
    bhvt_dist = np.zeros((ntrials, ntimes))

    for tr, (on, tid) in enumerate(zip(event_onsets, currentTargetID)):

        xTr, yTr = x[on-w1: on+w2], y[on-w1: on+w2]   # x,y hand position after event onset
        xTar, yTar = xTarg[tid-1], yTarg[tid-1]       # x,y target position for the given epoch
        bhvt_dist[tr] = np.sqrt((xTr-xTar)**2 + (yTr-yTar)**2)

    return bhvt_dist


def find_xy_positions_intrial (x, y, event_onsets, onset):
    """
    Find the instantaneous hand (or eye) x,y positions.

    Parameters
    ----------
    x, y: ndarrays
            The x,y positions of the eye/hand.
    event_onsets : ndarray
            The timestamps of the events on which the epochs will be aligned.
    onset : str
            It defines the onset on which the epochs are aligned. 
            Must be either 'targ', 'eye' or 'hand'.                  

    Returns
    -------
    x, y : ndarrays
            The instantaneous x,y positions of the hand/eye.     
    """ 
    w1, w2 = windows(onset)

    # For each event_onset, indices start from -w1 and reach up to +w2 (total length w1+w2)
    idx = np.arange(w1+w2) + event_onsets[:, None] - w1 # index mask of shape (ntrials,ntimes)

    return x[idx], y[idx]


def get_target_angle(xTarg, yTarg, previousTargetID, currentTargetID):
    """
    Calculate the direction of the target.

    Parameters
    ----------
    xTarg, yTarg: ndarrays
            The x,y positions of the targets.              
    previousTargetID, currentTargetID: ndarrays
            The ID of the previous and current visual targets
            for each trial.
    
    Returns
    -------
    th : ndarray
            The angles of the targets for each trial.     
    """

    if currentTargetID.max() == 7: 
        previousTargetID, currentTargetID = previousTargetID-1, currentTargetID-1

    xPre, yPre = xTarg[previousTargetID], yTarg[previousTargetID]
    xPost, yPost = xTarg[currentTargetID], yTarg[currentTargetID]

    # Target vector
    xRef = xTarg[1]-xTarg[0]
    yRef = yTarg[1]-yTarg[0]
    vecRefMag = np.sqrt(xRef**2 + yRef**2)

    # Movement vectors
    dx, dy  = xPost-xPre, yPost-yPre
    vecMag  = np.sqrt(dx**2 + dy**2)

    # Angle between movement vector and target vector
    dotProduct = xRef * dx + yRef * dy
    th = np.arccos(dotProduct / (vecRefMag * vecMag))

    # Sign correction: downward movements get negative angle
    th[yPost <= yPre] = -th[yPost <= yPre]

    return th
    

def mapping(subj):
    """
    Returns
    -------
    arrays_maps : dict [str, list] 
                The map of each array (keys) with the reordered channels (values), as they are implanted.
    """

    # Real positions of the electrodes in the brain - Data: Channel number positions
    V1_map = [122,np.nan,6,11,25,20,64,1,9,13,21,27,61,3,17,19,np.nan,29,
                63,4,15,10,14,31,np.nan,7,np.nan,12,120,18,2,5,8,23,16,22]
    V2_map = [95,41,43,44,118,np.nan,33,45,47,51,56,57,np.nan,34,49,53,55,np.nan,
                35,36,np.nan,50,54,62,37,38,48,46,52,60,39,40,42,116,58,59]
    DP_map = [65,77,np.nan,80,83,90,67,79,70,np.nan,86,91,69,114,72,81,85,92,71,
                66,74,82,87,93,73,68,76,84,88,94,75,np.nan,78,np.nan,89,96]
    A7_map = [24,32,101,106,111,119,126,97,102,107,np.nan,121,26,99,np.nan,108,113,123,124,
                98,103,np.nan,112,125,28,100,104,109,115,127,30,np.nan,105,110,117,128]

    # Check if the animal is Enya (y) or Jazz (j) because the mapping of the motor arrays is different
    if subj == 'enya':
 
        M1_map = [np.nan,2,1,3,4,6,8,10,14,np.nan,65,66,33,34,7,9,11,12,16,18,
                67,68,35,36,5,17,13,23,20,22,69,70,37,38,48,15,19,25,27,24,
                71,72,39,40,42,50,54,21,29,26,73,74,41,43,44,46,52,62,31,28,
                75,76,45,47,51,56,58,60,64,30,77,78,82,49,53,55,57,59,61,32,
                79,80,84,86,87,89,91,94,63,95,np.nan,81,83,85,88,90,92,93,96,np.nan]

    elif subj == 'jazz':  
                  
        M1_map = [65,2,1,3,4,np.nan,8,10,14,6,np.nan,66,33,34,7,9,11,12,16,18,
                67,68,35,36,5,17,13,23,20,22,69,70,37,38,48,15,19,25,27,24,
                71,72,39,40,42,50,54,21,29,26,73,74,41,43,44,46,52,62,31,28,
                75,76,45,47,51,56,58,60,64,30,77,78,82,49,53,55,57,59,61,32,
                np.nan,80,84,86,87,89,91,94,63,np.nan,79,81,83,85,88,90,92,93,96,95]
    else:
        raise ValueError('Session name must start with y (for Enya) or j (for Jazz)')

    arrays_maps = {'V1': V1_map,
                   'V2': V2_map,
                   'DP': DP_map,
                   '7A': A7_map,
                   'M1': M1_map}
    return arrays_maps   