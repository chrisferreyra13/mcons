"""Complexity metrics based on entropy."""

# This code was created based on Michael Schartner's code.
# Author: Michael Schartner, michael.schartner@internationalbrainlab.org
#         Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 09.12.14 - 2022
# License : BSD-3-Clause

from random import shuffle
import numpy as np

from ..utils.preprocessing import detrending_normalization
from ..utils.binary import (
    binarize_matrix,
    compute_synchrony_matrix,
    create_random_binary_matrix,
    map_matrix_to_integer,
)


def _compute_entropy(data):
    """Compute the Shannon entropy of a list of numbers/strings.

    Parameters
    ----------
    data : list or ndarray, shape (n_values,)
        List of numbers/strings to compute Shannon entropy.

    Returns
    -------
    float
        Entropy value.
    """
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        raise TypeError("The input should be a list or a numpy array.")

    if isinstance(data, list):
        data = np.array(data)

    # get frequency of each item in list
    _, counts = np.unique(data, return_counts=True)

    # compute probability of each unique item
    probabilities = [float(c)/len(data) for c in counts]
    entropy = -sum([p * np.log2(p) for p in probabilities])

    return entropy


def amplitude_coalition_entropy(data):
    """Compute Amplitude Coalition Entropy (ACE).

    Note: The shuffled result is used as normalization.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Mulidimensional time series matrix.

    Returns
    -------
    float
        Amplitude coalition entropy value (between 0 and 1).
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Data matrix should be a ndarray of float values.")

    data = detrending_normalization(data)
    data = binarize_matrix(data)

    try:
        col_map = map_matrix_to_integer(data)
    except ValueError as ex:
        raise ex

    # compute ace value (not normalized)
    ace_value = _compute_entropy(col_map)

    # shuffle the data for normalization
    n_channels, _ = np.shape(data)
    for ch_idx in range(n_channels):
        shuffle(data[ch_idx])

    shuffled_ace_value = _compute_entropy(map_matrix_to_integer(data))

    # normalize
    ace_value_normalized = ace_value / float(shuffled_ace_value)

    return ace_value_normalized


def synchrony_coalition_entropy(data, per_channel=False):
    """Compute Synchrony Coalition Entropy (SCE).

    Note: The shuffled result is used as normalization.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Mulidimensional time series matrix.
    per_channel : bool, optional (default False)
        If True, also returns SCE value per channel.

    Returns
    -------
    float
        Synchrony coalition entropy value (between 0 and 1).
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Data matrix should be a ndarray of float values.")

    data = detrending_normalization(data)
    n_channels, n_values = np.shape(data)
    data = compute_synchrony_matrix(data)

    # compute sce value (not normalized)
    channel_sce_value = np.zeros(n_channels)
    for ch_idx in range(n_channels):
        try:
            col_map = map_matrix_to_integer(data[ch_idx])
        except ValueError as ex:
            raise ex
        channel_sce_value[ch_idx] = _compute_entropy(col_map)

    # create random matrix for normalization
    col_map = map_matrix_to_integer(
        create_random_binary_matrix(n_channels - 1, n_values)
    )
    normalization_value = _compute_entropy(col_map)

    # normalize
    sce_total = np.mean(channel_sce_value) / normalization_value

    if per_channel:
        return sce_total, channel_sce_value/normalization_value
    else:
        return sce_total
