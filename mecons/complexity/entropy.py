"""Complexity metrics based on entropy."""

# This code was created based on Michael Schartner's code.
# Author: Michael Schartner, m.schartner@sussex.ac.uk
#         Christian ferreyra, chrisferreyra13@gmail.com
# Date: 09.12.14 - 2022
# License : BSD-3-Clause

import numpy as np
from random import shuffle

from .utils import (
    detrending_normalization,
    binarize_matrix,
    compute_synchrony_matrix,
    create_random_binary_matrix,
    map_matrix_to_integer,
)


def _compute_entropy_string(string):
    """Computes the Shannon entropy of a string."""
    string = list(string)
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    entropy = -sum([p * np.log2(p) for p in prob])

    return entropy


def compute_amplitude_coalition_entropy(data):
    """Computes Amplitude Coalition Entropy (ACE), using shuffled result as normalization.

    Parameters
    ----------
    data : Numpy array
        Data matrix with rows as channels and columns as values.

    Returns
    -------
    float
        Amplitude coalition entropy value (between 0 and 1).

    """
    if not isinstance(data, np.ndarray):
        TypeError("Data matrix should be a Numpy array of float values.")

    data = detrending_normalization(data, first_mean=False)
    ro, co = np.shape(data)
    data = binarize_matrix(data)
    entropy = _compute_entropy_string(data)
    # shuffle the data for normalization
    for i in range(ro):
        shuffle(data[i])

    shuffle_entropy = _compute_entropy_string(data)

    result = entropy / float(shuffle_entropy)

    return result


def compute_synchrony_coalition_entropy(data):
    """Computes Synchrony Coalition Entropy (SCE), using shuffled result as normalization.

    Parameters
    ----------
    data : Numpy array
        Data matrix with rows as channels and columns as values.

    Returns
    -------
    float
        Synchrony coalition entropy value (between 0 and 1).
    """
    if not isinstance(data, np.ndarray):
        TypeError("Data matrix should be a Numpy array of float values.")

    data = detrending_normalization(data)
    n_channels, n_values = np.shape(data)
    data = compute_synchrony_matrix(data)
    ce = np.zeros(n_channels)
    normalization_value = _compute_entropy_string(
        map_matrix_to_integer(create_random_binary_matrix(n_channels - 1, n_values))
    )
    for channel in range(n_channels):
        c = map_matrix_to_integer(data[channel])
        ce[channel] = _compute_entropy_string(c)

    synchrony_coalition_entropy_total = np.mean(ce) / normalization_value

    return synchrony_coalition_entropy_total
    # TODO: see if it should return ce/norm
