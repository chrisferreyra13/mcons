"""Usuful functions for preprocessing."""

# This code was created based on Michael Schartner's code.
# Author: Michael Schartner, m.schartner@sussex.ac.uk
#         Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 09.12.14 - 2022
# License : BSD-3-Clause

import numpy as np
from scipy import signal


def detrending_normalization(data):
    """Detrend and subtract the mean on input data.

    Parameters
    ----------
    data : ndarray, (n_channels, n_times)
        Multidimensional time series matrix.

    Returns
    -------
    ndarray
        Data matrix after detrending and subtracting the mean.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("The input matrix 'data' should be ndarray.")

    n_channels, n_times = np.shape(data)
    data_processed = np.zeros((n_channels, n_times))

    for ch_idx in range(n_channels):
        data_processed[ch_idx, :] = signal.detrend(
            data[ch_idx, :] - np.mean(data[ch_idx, :]), axis=0
        )

    return data_processed
