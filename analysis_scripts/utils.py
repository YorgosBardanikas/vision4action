"""
Collection of utility functions that are used in the V4A analysis scripts.
"""

import mne
import numpy as np
from frites.io import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def load_session_group(session_type):
    """
    Load the session group based on the landing time type.

    Parameters
    ----------
    session_type : str
            Can either be 'short' for landing times of 200ms
            or 'long' for landing times of 800 to 1000ms
            or 'short12' for the 12 trial-protocols.

    Returns
    -------
    session_group : list of str
            The group with all pre-selected sessions.
    """

    if session_type == 'short':
        session_group = ['y180911-land-002','y180911-land-003','y180913-land-001',
                        'y180914-land-001','y180914-land-003',
                        'y180924-land-001','y180924-land-002','y180924-land-003',
                        'y180925-land-001','y180925-land-002','y180925-land-003']
    elif session_type == 'long':    
        session_group = ['y181015-land-002','y181015-land-003','y181015-land-004',
                        'y181016-land-001','y181016-land-002','y181016-land-003',
                        'y181018-land-001','y181018-land-002','y181018-land-003',
                        'y181019-land-001','y181019-land-002','y181019-land-003',
                        'y181019-land-004','y181030-land-001','y181102-land-001']
    elif session_type == 'short12':
        session_group = ['y180118-land-001','y180122-land-001',
                        'y180124-land-001','y180125-land-001','y180219-land-001',
                        'y180219-land-002','y180219-land-003','y180219-land-004',
                        'y180220-land-001','y180220-land-002','y180221-land-002', 
                        'y180222-land-001', 'y180222-land-002'] 
    elif session_type == 'short12E':  
        session_group = ['y190829-land-001','y190902-land-001','y190903-land-001', 
                        'y190903-land-002','y190905-land-001','y190906-land-001', 
                        'y190906-land-002','y190909-land-001','y190909-land-002',
                        'y190909-land-003','y190911-land-001','y190911-land-002', 
                        'y190913-land-001','y190913-land-002']

    elif session_type == 'shortJ': 
        session_group = ['j210615-land-002','j210616-land-002',
                        'j210617-land-001','j210618-land-001',
                        'j210618-land-002','j210618-land-003',
                        'j210625-land-002','j210625-land-003','j210628-land-001',
                        'j210628-land-002','j210628-land-003','j210629-land-001']  
    elif session_type == 'short12J':
        session_group = ['j210204-land-001','j210204-land-002',
                        'j210208-land-001','j210211-land-001',
                        'j210211-land-002','j210212-land-001','j210215-land-002',
                        'j210215-land-003','j210215-land-004','j210216-land-001',
                        'j210216-land-003','j210217-land-001','j210218-land-003',
                        'j210219-land-001','j210225-land-001']
    return session_group



def concatenate_sessions (session_group, n_sessions, onset, targetID, epoch_content='hga'):
    """
    Concatenate the epochs of high gamma or velocity for the selected sessions.

    Parameters
    ----------
    session_group : str | list of str
                A string of format 'y181016-land-001' for the initialization.
                OR a list of sessions of the same format can be provided.
    n_sessions : int | str
                Number of sessions of the same day to be concatenated.
                OR the string 'group' to use a group of sessions of different days.
    onset : str
                It defines the onset on which the epochs are aligned. 
                Must be either 'targ' or 'hand' or 'eye'.
    targetID : int
            The number ID of the target. Can be 2, 3 or 4.                 
    epoch_content : str 
                Epoch content. Can either be 'hga' (default) or 'bhv'.            

    Returns
    ----------                
    epochs : mne.Epoch
                The concatenated mne.Epochs for the selected sessions.
    epochs_list: list
                A list of the mne.Epochs before concatenation.
    """

    if isinstance(n_sessions,int):
        session_names = [session_group]
        if n_sessions > 1:
            session_names = []
            session = list(session_group)
            init = int(session[-1])
            for idx in range(init,n_sessions+init):
                session[-1] = str(idx)
                session_names.append("".join(session))
    elif n_sessions == 'group':
        session_names = session_group
    else: raise ValueError('n_sessions must be an integer or the string "group"')

    # Load the epochs objects (mne.Epochs) and append them in a list for concatenation
    v4a_dir = 'path_to_directory'
    epochs_list = []
    for session_name in session_names:
        data_dir = f'{v4a_dir}/{session_name}'
        epoch_file = f'{session_name}_{epoch_content}_{onset}{targetID}-epo.fif'
        try:
            mne.read_epochs(f'{data_dir}/{epoch_file}',verbose='error')
        except FileNotFoundError:
            print(f'File {epoch_file} was not found.')
            continue
        else: epoch = mne.read_epochs(f'{data_dir}/{epoch_file}',verbose='error')
        epochs_list.append(epoch)   

    # Create a dictionary with all session names and epochs
    epochs_dict = dict(zip(session_names, epochs_list))

    # Concatenate the epochs of the selected sessions 
    epochs = mne.concatenate_epochs(epochs_list, on_mismatch='ignore')
    print(f'The sessions that were concatenated are: {session_names}')

    return epochs, epochs_dict



def load_epochs(session_type, onset, targetID, content='hga', 
                                concat='group', dctn=False):
    """
    Concatenate the epochs of high gamma or velocity for the selected sessions.

    Parameters
    ----------
    session_type : str
            Can either be 'short' for landing times of 200ms
            or 'long' for landing times of 800 to 1000ms
            or 'short12' for the 12 trial-protocols.
    onset : str
            It defines the onset on which the epochs are aligned. 
            Must be either 'targ' or 'hand' or 'eye'.
    targetID : int
            The number ID of the target. Can be 2, 3 or 4.             
    content : str 
            Epoch content. Can either be 'hga' (default) or 'bhv'.            
    concat : str
            Concatenation mode. Can either be 'group' (default) or 'day'.
    dctn: bool | False
            If True, it returns a dictionary with all session names and 
            epochs separately.

    Returns
    ----------                
    epochs : mne.Epoch
            The selected mne.Epoch object.

    """

    if concat == 'day':
        # Concatenate the epochs for the sessions of the same day
        session_id = 'y180118-land-'
        n_sessions = 1   # number of sessions to be concatenated
        epochs, epochs_dict = concatenate_sessions(session_id+'001', n_sessions, 
                                        onset, targetID, epoch_content=content)

    elif concat == 'group':
        # Concatenate a group of selected sessions for short or long landing times
        session_group = load_session_group(session_type)                  
        n_sessions = concat
        epochs, epochs_dict = concatenate_sessions(session_group, n_sessions, 
                                        onset, targetID, epoch_content=content)

    if dctn: return epochs, epochs_dict
    else: return epochs



def keep_1attempt_trials (epochs_bhv, epochs_mua):
    """Keep only trials that the monkey accomplished without
    an error, because the same sequence is presented after an
    error.

    Parameters
    ----------
    epochs_bhv: mne.Epochs
            The behavioral epochs.
    epochs_mua: mne.Epochs
            The MUA epochs.
            
    Returns
    -------
    epochs_bhv, epochs_mua: mne.Epochs
            The filtered epochs for trials accomplished 
            without error.
    """

    tgRank = epochs_bhv.metadata['Target Rank']
    tg2_rep = epochs_bhv.metadata['Target Repeated'][tgRank==2]
    tg_rep = np.tile(tg2_rep, 3)
    epochs_bhv = epochs_bhv[tg_rep==False]

    if hasattr(epochs_mua, 'events'):
        epochs_mua = epochs_mua[tg_rep==False]
        return epochs_bhv, epochs_mua
    else: return epochs_bhv



def group_events (trials, regressor):
    """
    Group the event codes according to the specific regressor.

    Parameters
    ----------
    trials : ndarray
            The trial codes of format ij for LS_i Target_j.
    regressor: str
        The regressor of grouping.      

    Returns
    -------
    group : ndarray
            The new event codes for the chosen grouping.
    group_dict : dict
            The event ids that correspond to the new event codes.
    """

    group = []
    if regressor == 'motor': # all the different submovements
        for ev in trials:
            if ev in [12,22]: group.append(1)
            elif ev in [32,42]: group.append(2)
            elif ev in [52,62]: group.append(3)
            elif ev in [72,82]: group.append(4)
            elif ev in [92,102]: group.append(5)
            elif ev in [112,122]: group.append(6)
            elif ev in [13,54]: group.append(7)
            elif ev in [23,34]: group.append(8)
            elif ev in [33,64]: group.append(9)
            elif ev in [43,14]: group.append(10)
            elif ev in [53,44]: group.append(11)
            elif ev in [63,24]: group.append(12)
            elif ev in [73,114]: group.append(13)
            elif ev in [83,94]: group.append(14)
            elif ev in [93,124]: group.append(15)
            elif ev in [103,74]: group.append(16)
            elif ev in [113,104]: group.append(17)
            elif ev in [123,84]: group.append(18)
            else: group.append(0)  
        group_dict = {'ANY':0}
        group_dict.update({f'Move_{i}':i for i in range(1,19)})

    return np.array(group), group_dict


def fit_lda(epochs, classes, seed):
    """
    Wrapping function to parallelize. 
    Fit the LDA model after shuffling the classes.

    Parameters
    ----------
    epochs : ndarray
            The data of shape (nsamples,nfeatures)
    classes: ndarray
            The class labels (nsamples,)
    seed: int
            The seed of the random state for reproducibility.

    Returns
    -------
    lda : sklearn.LDA
            The fitted LDA model
    """

    logger.info(f'Permutation: {seed+1}')
    rng = np.random.default_rng(seed=seed)
    shuffled_classes = rng.permutation(classes)
    lda_input = epochs.get_data().mean(2) # average across time
    lda = LDA()
    lda.fit(lda_input, shuffled_classes)
    return lda

