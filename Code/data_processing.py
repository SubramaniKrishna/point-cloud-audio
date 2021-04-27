"""
ESC-50 Dataset Helper functions to load and split the data
"""

import pandas as pd
import numpy as np

def load_esc(loc = '../ESC-50-master/meta/esc50.csv',loc_audio = '../ESC-50-master/audio/', list_categories = ['dog','chainsaw','crackling_fire','helicopter','rain','crying_baby','clock_tick','sneezing','rooster','sea_waves']):
    """
    Read and load custom classes from ESC-50
    Returns list of audio files and corresponding labels (relative to the input provided)
    loc -> location of the esc50.csv
    loc_audio -> location of the esc50 audio folder
    list_categories -> Categories of esc50 to select
    (Default categories correspond to the esc10 split)
    """
    # loc = '../ESC-50-master/meta/esc50.csv'
    # loc_audio = '../ESC-50-master/audio/'
    annotation_data = pd.read_csv(loc)

    # Selecting the ESC-10 subset
    # list_categories = ['dog','chainsaw','crackling_fire','helicopter','rain','crying_baby','clock_tick','sneezing','rooster','sea_waves']
    dict_newlabels = {k:it for it,k in enumerate(list_categories)}
    annotation_data = annotation_data[annotation_data.category.isin(list_categories)]

    indices_filtered = annotation_data.index.to_list()
    for idx,cat in enumerate(annotation_data.category.to_list()):
        annotation_data.at[indices_filtered[idx],'target'] = dict_newlabels[cat]

    # Load audio
    list_labels = annotation_data['target'].to_list()
    l = np.array(list_labels)

    list_audio_file_names = annotation_data['filename'].to_list()
    list_audio_locs = [loc_audio + f for f in list_audio_file_names]
    list_audio_locs = np.array(list_audio_locs)

    return list_audio_locs,l

def tt_split(list_audio_locs,l,f = 0.8):
    """
    Split audio into train/test given by fraction (Will split the audio files, not the generated frames)
    Works in conjunction with load_esc (takes as input the output of load_esc)
    """

    nclass = max(l) + 1
    dict_audio = {k:[] for k in np.arange(nclass)}
    for idx,fn in enumerate(list_audio_locs):
        dict_audio[l[idx]].append(fn)
    
    audio_train = []
    l_train = []
    audio_test = []
    l_test = []
    for k in dict_audio.keys():
        N = len(dict_audio[k])
        inds = np.random.permutation(N)
        for idx in inds[:(int)(f*N)]:
            audio_train.append(dict_audio[k][idx])
            l_train.append(k)
        for idx in inds[(int)(f*N):]:
            audio_test.append(dict_audio[k][idx])
            l_test.append(k)
    
    return audio_train,l_train,audio_test,l_test
