"""Complexity metrics based on compressibility."""

# This code was created based on Michael Schartner's code.
# Author: Michael Schartner, m.schartner@sussex.ac.uk
#         Christian ferreyra, chrisferreyra13@gmail.com
# Date: 09.12.14 - 2022
# License : BSD-3-Clause


import numpy as np
from random import shuffle

from .utils import detrending_normalization, binary_matrix_to_string


def _lempel_ziv_welch_compression(binary_string):
    """Returns the size of the dictionary of binary words after Lempel-Ziv-Welch compression.

    Parameters
    ----------
    binary_string : str
        Binary string to be compressed.

    Returns
    -------
    int
        Size of the dictionary of binary words.

    """
    dictionary = {}
    word = ""
    for c in binary_string:
        wc = word + c
        if wc in dictionary:
            word = wc
        else:
            dictionary[wc] = wc
            word = c

    return len(dictionary)


def compute_lempel_ziv_complexity(data):
    """Computes LZc and use shuffled result as normalization.

    Parameters
    ----------
    data : ndarray, (n_channels, n_times)
        Mulidimensional time series matrix.

    Returns
    -------
    float
        Lempel-Ziv complexity value (between 0 and 1).
    """

    if not isinstance(data, np.ndarray):
        TypeError("Data matrix should be a ndarray of float values.")

    data = detrending_normalization(data)
    binary_str = binary_matrix_to_string(data)
    random_list = list(binary_str)
    shuffle(random_list)
    random_str = ""
    for c in random_list:
        random_str += c

    lzc_value = _lempel_ziv_welch_compression(binary_str) / float(
        _lempel_ziv_welch_compression(random_str)
    )

    return lzc_value