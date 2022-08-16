"""
    Useful functions.
    Author: Christian ferreyra, chrisferreyra13@gmail.com
    Date: 2022
"""

import os
import xml.etree.ElementTree as ET
import mne
from pandas import read_feather


def read_anywave_mtg(anywave_path):
    """Reads Anywave's mtg file."""

    mtg={}
    
    try:
        anywave_mtg=ET.parse(anywave_path)

    except Exception as ex:
        raise ex

    root=anywave_mtg.getroot()

    for element in root:
        if element.tag=='Channel':
            # get channel name
            mtg[element.attrib["name"]]={}
            mtg[element.attrib["name"]]["type"]=element[0].text
            mtg[element.attrib["name"]]["reference"]=element[1].text

    return mtg

def set_montage(raw,montage,drop_refs=True):
    """Applies montage dictionary to raw mne object."""
    anodes=[]
    reference=[]
    channel_type={}
    for ch,props in montage.items():
        anodes.append(ch)
        reference.append(props["reference"])
        channel_type[ch]=props["type"].lower()
    
    # get subset of channels
    raw=raw.pick_channels(ch_names=list(set(reference+anodes)))

    # load data to apply montage
    raw.load_data()

    # set channel types
    try:
        raw.set_channel_types(channel_type)
    except Exception as ex:
        raise ex

    reference=set(reference)
    if "AVG" in reference and len(reference)==1:
        # average reference
        try:
            # in mne, set_eeg_reference sets the reference for eeg, seeg and ecog
            raw.set_eeg_reference(read_channels='average', ch_type='auto')
        except Exception as ex:
            raise ex
    elif len(reference)==1:
        # monopolar reference
        try:
            raw.set_eeg_reference(read_channels=list(reference), ch_type='auto')
        except Exception as ex:
            raise ex
    else:
        # bipolar 
        try:
            raw=mne.set_bipolar_reference(raw,anode=anodes,cathode=list(reference), drop_refs=drop_refs)
        except Exception as ex:
            raise ex

    return raw