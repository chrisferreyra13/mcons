"""
    Complexity analysis on SEEG.
    Author: Christian ferreyra, chrisferreyra13@gmail.com
    Date: 2022
"""

import numpy as np
import mne
from lempel_ziv_complexity import lempel_ziv_complexity
from utils.util import *
import matplotlib
matplotlib.use('TkAgg')

# NOT WORKING FILES -> mne doesn't like them
# r'D:\ferreyra\cflab\complexity\dataset\sub-e0c4a002f130\ses-postimp\ieeg\sub-e0c4a002f130_ses-postimp_task-sleep_run-01_ieeg.vhdr'
# r'D:\ferreyra\cflab\complexity\dataset\sub-4714cefba870\ses-postimp\ieeg\sub-4714cefba870_ses-postimp_task-sleep_run-01_ieeg.vhdr'


FILENAME = r'D:\ferreyra\cflab\complexity\dataset\sub-d69c8d61df45\ses-postimp\ieeg\sub-d69c8d61df45_ses-postimp_task-sleep_run-01_ieeg.vhdr'
ANYWAVE_MTG=r'D:\ferreyra\cflab\complexity\dataset\derivatives\anywave\ferreyra\sub-d69c8d61df45\ses-postimp\ieeg\sub-d69c8d61df45_ses-postimp_task-sleep_run-01_ieeg.vhdr.mtg'

def compute_complex_metrics(filename):

    # reading file
    raw=mne.io.read_raw_brainvision(filename)

    # read and set montage
    try:
        mtg=read_anywave_mtg(ANYWAVE_MTG)
        raw=set_montage(raw,mtg)
    except Exception as ex:
        print(ex)
        return
    

    print(raw.info)
    #load data
    # ch_indices=list(np.linspace(0,14,15,dtype=int)) # mne.pick_channels()
    # data=raw.get_data(picks=ch_indices)

    #compute LZc metric
    # LZc=lempel_ziv_complexity(data)

    # print(LZc)

    return


if __name__=='__main__':
    compute_complex_metrics(FILENAME)


