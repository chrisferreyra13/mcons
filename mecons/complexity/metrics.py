"""Complexity class to compute different complexity metrics."""
# This code was created based on Michael Schartner's code.
# Author: Michael Schartner, m.schartner@sussex.ac.uk
#         Christian ferreyra, chrisferreyra13@gmail.com
# Date: 09.12.14 - 2022
# License : BSD-3-Clause


import numpy as np
from scipy import signal
from random import shuffle




def _detrending_normalization(data, first_mean=True):
    """Detrending and baseline subtraction on input data."""
    #TODO: think about the first_mean
    ro, co = np.shape(data)
    data_processed = np.zeros((ro, co))

    for i in range(ro):
        if first_mean:
            data_processed[i, :] = signal.detrend(
                data[i, :]-np.mean(data[i, :]), axis=0)
        else:
            data_processed[i, :] = signal.detrend(data[i, :], axis=0)
            data_processed[i, :] = data_processed[i, :]-np.mean(data[i, :])

    return data_processed


def _lempel_ziv_welch_compression(binary_str):
    """Return the size of the dictionary of binary words after Lempel-Ziv-Welch compression."""
    d = {}
    w = ''
    i = 1
    for c in binary_str:
        wc = w + c
        if wc in d:
            w = wc
        else:
            d[wc] = wc
            w = c
        i += 1

    return len(d)


def _binarize_matrix(data):
    """Binarizes the input multidimensional time series based on Hilbert transform amplitude."""
    ro, co = np.shape(data)
    th = 0
    m = np.zeros((ro, co))
    binary_matrix = np.zeros((ro, co))
    for i in range(ro):
        m[i, :] = abs(signal.hilbert(data[i, :]))
        th = np.mean(m[i, :])
        for j in range(co):
            if m[i, :] >= th :
                binary_matrix[i,j]=1

    return binary_matrix


def _binary_matrix_to_string(binary_matrix):
    """Creates one string being the binarized input matrix concatenated comlumn-by-column."""
    ro, co = np.shape(binary_matrix)
    binary_str = ''
    for j in range(co):
        for i in range(ro):
            if binary_matrix[i, j]==1:
                binary_str += '1'
            else:
                binary_str += '0'

    return binary_str


def _map_matrix_to_integer(binary_matrix):
    """Bijection, mapping each binary column of binary matrix psi onto an integer."""
    ro,co=np.shape(binary_matrix) 
    c=np.zeros(co)
    for t in range(co):
        for j in range(ro):
            c[t]=c[t]+binary_matrix[j,t]*(2**j)

    return c


def _compute_entropy(string):
    """Computes the Shannon entropy of a string."""
    string = list(string)
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
    entropy = - sum([ p * np.log2(p) for p in prob ])

    return entropy
    

def _compute_synchrony(p1,p2,threshold=0.8):
    """Computes a binary synchrony time series between two phase time series."""
    differences=np.array(abs(p1-p2))
    sync_time_series=np.zeros(len(differences))
    for i in range(len(differences)):
        if differences[i]>np.pi:
            differences[i]=2*np.pi-differences[i]
        if differences[i]<threshold:
            sync_time_series[i]=1

    return sync_time_series


def _create_random_binary_matrix(n_rows,n_columns):
    """Creates a random binary matrix."""
    binary_matrix=np.random.rand(n_rows,n_columns)
    for i in range(n_rows):
        for j in range(n_columns):
            if binary_matrix[i,j]>0.5:
                binary_matrix[i,j]=1
            else:
                binary_matrix[i,j]=0

    return binary_matrix


def _compute_synchrony_matrix(self,data):
    """Computes binary synchrony matrix (by column) based on a multidimensional time series matrix."""
    phases_matrix=np.angle(signal.hilbert(data))
    ro,co=np.shape(phases_matrix)
    synch_matrix=np.zeros((ro, ro-1, co))  
    for i in range(ro):
        l=0
        for j in range(ro):
            if i!=j:
                synch_matrix[i,l]=self.compute_synchrony(phases_matrix[i],phases_matrix[j])
                l+=1
    
    return synch_matrix

def compute_lempel_ziv_complexity(data=None):
    """Computes LZc and use shuffled result as normalization."""
    if not data:
        print("Error: data is empty")

    data = _detrending_normalization(data)
    binary_str = _binary_matrix_to_string(data)
    m = list(binary_str)
    shuffle(m)
    w = ''
    for i in range(len(m)):
        w += m[i]

    result = _lempel_ziv_welch_compression(binary_str)/float(_lempel_ziv_welch_compression(w))

    return result


def compute_amplitude_coalition_entropy(data):
    """Computes Amplitude Coalition Entropy (ACE), using shuffled result as normalization."""
    if not data:
        print("Error: data is empty")

    data=_detrending_normalization(data, first_mean=False)
    ro,co=np.shape(data)
    data=_binarize_matrix(data)
    entropy=_compute_entropy(data)
    # shuffle the data for normalization
    for i in range(ro):
        shuffle(data[i])
    
    shuffle_entropy=_compute_entropy(data)

    result= entropy/float(shuffle_entropy)

    return result

def compute_synchrony_coalition_entropy(data):
    """Computes Synchrony Coalition Entropy (SCE), using shuffled result as normalization."""
    if not data:
        print("Error: data is empty")

    data = _detrending_normalization(data)
    ro,co=np.shape(data)
    data=_compute_synchrony_matrix(data)
    ce=np.zeros(ro)
    norm=_compute_entropy(_map_matrix_to_integer(_create_random_binary_matrix(ro-1,co)))
    for i in range(ro):
        c=_map_matrix_to_integer(data[i])
        ce[i]=_compute_entropy(c)

    result = np.mean(ce)/norm

    return result
    #TODO: see if it should return ce/norm


