"""Usuful functions for complexity metrics."""

# This code was created based on Michael Schartner's code.
# Author: Michael Schartner, m.schartner@sussex.ac.uk
#         Christian ferreyra, chrisferreyra13@gmail.com
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


def binarize_matrix(data, thr_method="mean"):
    """Binarize the input multidimensional time series based on
    the instantaneous amplitude of the analytic signal.

    Parameters
    ----------
    data : ndarray, (n_channels, n_times)
        Multidimensional time series matrix.

    thr_method : str
        If 'mean', the mean value of the Hilbert amplitude time series is used.
        If 'median' the median value is used. By default 'mean'.

    Returns
    -------
    ndarray
        Binarized data matrix.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("The input matrix 'data' should be ndarray.")

    if thr_method not in ["mean", "median"]:
        raise ValueError(
            "The parameter thr_method should be 'mean' or 'median'.")

    n_channels, n_times = np.shape(data)
    threshold = 0
    hilbert_amplitude_matrix = np.zeros((n_channels, n_times))
    binary_matrix = np.zeros((n_channels, n_times), dtype=np.int8)

    for ch_idx in range(n_channels):
        # get Hilbert amplitude time series
        hilbert_amplitude_matrix[ch_idx, :] = abs(
            signal.hilbert(data[ch_idx, :]))

        # get threshold
        if thr_method == "mean":
            threshold = np.mean(hilbert_amplitude_matrix[ch_idx, :])
        else:
            threshold = np.median(hilbert_amplitude_matrix[ch_idx, :])

        # binarize time series
        for t in range(n_times):
            if hilbert_amplitude_matrix[ch_idx, t] >= threshold:
                binary_matrix[ch_idx, t] = 1

    return binary_matrix


def binary_matrix_to_string(binary_matrix):
    """Create one string being the binarized input matrix,
    concatenated column-by-column.

    Parameters
    ----------
    binary_matrix : ndarray
        Data matrix with binary values.

    Returns
    -------
    str
        Binary string.
    """
    if not isinstance(binary_matrix, np.ndarray):
        raise TypeError("The input matrix 'data' should be ndarray.")

    n_rows, n_cols = np.shape(binary_matrix)
    binary_str = str()
    for j in range(n_cols):
        for i in range(n_rows):
            if binary_matrix[i, j] == 1:
                binary_str += "1"
            elif binary_matrix[i, j] == 0:
                binary_str += "0"
            else:
                raise ValueError("The input matrix 'data' should be binary.")

    return binary_str


def map_matrix_to_integer(binary_matrix):
    """Map each binary column of binary matrix onto an integer.

    Parameters
    ----------
    binary_matrix : ndarray, (n_rows, n_columns)
        Data matrix with binary values.

    Returns
    -------
    ndarray, (1,n_columns)
        Array with integers.
    """
    if not isinstance(binary_matrix, np.ndarray):
        raise TypeError("The input matrix 'data' should be ndarray.")

    n_rows, n_cols = np.shape(binary_matrix)
    col_map = np.zeros(n_cols, dtype=int)
    for col_idx in range(n_cols):
        for row_idx in range(n_rows):
            if (binary_matrix[row_idx, col_idx] == 0
                    or binary_matrix[row_idx, col_idx] == 1):
                col_map[col_idx] = col_map[col_idx] + \
                    binary_matrix[row_idx, col_idx] * (2**row_idx)
            else:
                raise ValueError("The input matrix 'data' should be binary.")

    return col_map


def _compute_synchrony(p_1, p_2, threshold=0.8):
    """Compute a binary synchrony time series between two phase time series.

    Parameters
    ----------
    p_1 : ndarray, (1, n_times)
        Phase time series.
    p_2 : ndarray, (1, n_times)
        Phase time series.
    threshold : float, optional
        Threshold to define "synchronized" (1) and "not synchronized" (0).
        By default 0.8.

    Returns
    -------
    ndarray, (1, n_times)
        Binary synchrony time series.
    """
    if not isinstance(p_1, np.ndarray) or not isinstance(p_2, np.ndarray):
        raise TypeError("The parameters p_1 and p_2 should be ndarray.")

    if len(p_1) != len(p_2):
        raise ValueError(
            "The parameters p_1 and p_2 don't have the same length.")

    differences = np.array(abs(p_1 - p_2))
    sync_time_series = np.zeros(len(differences))
    for i, difference in enumerate(differences):
        # center the difference between 0 and pi
        if difference > np.pi:
            difference = 2 * np.pi - difference

        if difference < threshold:
            # the time series are synchronized
            sync_time_series[i] = 1

    return sync_time_series


def compute_synchrony_matrix(data):
    """Compute binary synchrony matrix based on
    a multidimensional time series matrix.

    Parameters
    ----------
    data : ndarray, (n_channels, n_times)
        Multidimensional time series matrix.

    Returns
    -------
    ndarray, (n_channels, n_channels - 1, n_times)
        Synchrony matrix.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("The input matrix 'data' should be ndarray.")

    # get phase time series from Hilbert transform
    phases_matrix = np.angle(signal.hilbert(data))
    n_channels, n_times = np.shape(phases_matrix)
    synch_matrix = np.zeros((n_channels, n_channels - 1, n_times))
    for i in range(n_channels):
        k = 0
        for j in range(n_channels):
            # ignore the same channel
            if i != k:
                synch_matrix[i, k] = _compute_synchrony(
                    phases_matrix[i], phases_matrix[j])
                k += 1

    return synch_matrix


def create_random_binary_matrix(n_rows, n_columns):
    """Create a random binary matrix with uniform distribution.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_columns : int
        Number of colums

    Returns
    -------
    ndarray, (n_rows, n_columns)
        Random binary matrix.
    """
    if not isinstance(n_rows, int) or not isinstance(n_rows, int):
        raise TypeError("The number of rows and columns must be integer.")

    binary_matrix = np.random.rand(n_rows, n_columns)
    for i in range(n_rows):
        for j in range(n_columns):
            if binary_matrix[i, j] > 0.5:
                binary_matrix[i, j] = 1
            else:
                binary_matrix[i, j] = 0

    return binary_matrix
