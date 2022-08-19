"""Usuful functions for complexity metrics."""

# This code was created based on Michael Schartner's code.
# Author: Michael Schartner, m.schartner@sussex.ac.uk
#         Christian ferreyra, chrisferreyra13@gmail.com
# Date: 09.12.14 - 2022
# License : BSD-3-Clause

import numpy as np
from scipy import signal


def detrending_normalization(data, first_mean=True):
    """Detrends and subtracts the baseline on input data.

    Parameters
    ----------
    data : Numpy array
        Data matrix with rows as channels and columns as values.
    first_mean : bool, optional
        If true, the mean value is subtracted before detrending. By default True.

    Returns
    -------
    Numpy array
        Data matrix after detrending and subtracting the baseline.
    """
    # TODO: think about the first_mean
    n_channels, n_values = np.shape(data)
    data_processed = np.zeros((n_channels, n_values))

    for channel in range(n_channels):
        if first_mean:
            data_processed[channel, :] = signal.detrend(
                data[channel, :] - np.mean(data[channel, :]), axis=0
            )
        else:
            data_processed[channel, :] = signal.detrend(data[channel, :], axis=0)
            data_processed[channel, :] = data_processed[channel, :] - np.mean(
                data[channel, :]
            )

    return data_processed


def binarize_matrix(data):
    """Binarizes the input multidimensional time series based on Hilbert transform amplitude."""
    ro, co = np.shape(data)
    th = 0
    m = np.zeros((ro, co))
    binary_matrix = np.zeros((ro, co))
    for i in range(ro):
        m[i, :] = abs(signal.hilbert(data[i, :]))
        th = np.mean(m[i, :])
        for j in range(co):
            if m[i, :] >= th:
                binary_matrix[i, j] = 1

    return binary_matrix


def binary_matrix_to_string(binary_matrix):
    """Creates one string being the binarized input matrix concatenated comlumn-by-column."""
    ro, co = np.shape(binary_matrix)
    binary_str = ""
    for j in range(co):
        for i in range(ro):
            if binary_matrix[i, j] == 1:
                binary_str += "1"
            else:
                binary_str += "0"

    return binary_str


def map_matrix_to_integer(binary_matrix):
    """Bijection, mapping each binary column of binary matrix psi onto an integer."""
    ro, co = np.shape(binary_matrix)
    c = np.zeros(co)
    for t in range(co):
        for j in range(ro):
            c[t] = c[t] + binary_matrix[j, t] * (2**j)

    return c


def compute_synchrony(p1, p2, threshold=0.8):
    """Computes a binary synchrony time series between two phase time series."""
    differences = np.array(abs(p1 - p2))
    sync_time_series = np.zeros(len(differences))
    for i in range(len(differences)):
        if differences[i] > np.pi:
            differences[i] = 2 * np.pi - differences[i]
        if differences[i] < threshold:
            sync_time_series[i] = 1

    return sync_time_series


def compute_synchrony_matrix(data):
    """Computes binary synchrony matrix (by column) based on a multidimensional time series matrix."""
    phases_matrix = np.angle(signal.hilbert(data))
    ro, co = np.shape(phases_matrix)
    synch_matrix = np.zeros((ro, ro - 1, co))
    for i in range(ro):
        l = 0
        for j in range(ro):
            if i != j:
                synch_matrix[i, l] = compute_synchrony(
                    phases_matrix[i], phases_matrix[j]
                )
                l += 1

    return synch_matrix


def create_random_binary_matrix(n_rows, n_columns):
    """Creates a random binary matrix."""
    binary_matrix = np.random.rand(n_rows, n_columns)
    for i in range(n_rows):
        for j in range(n_columns):
            if binary_matrix[i, j] > 0.5:
                binary_matrix[i, j] = 1
            else:
                binary_matrix[i, j] = 0

    return binary_matrix
