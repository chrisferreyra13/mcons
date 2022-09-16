"""Tests for usuful functions in complexity"""

# Author: Christian ferreyra, chrisferreyra13@gmail.com

import numpy as np
import pytest

from mecons.complexity._utils import (
    binary_matrix_to_string,
    detrending_normalization,
    map_matrix_to_integer,
    binarize_matrix,
    compute_synchrony_matrix,
    create_random_binary_matrix,
    _compute_synchrony
)
from scipy import signal


def test_detrending_normalization():
    """Test detrending and normalization."""
    n_points = 1000
    # input data
    x = np.linspace(0, n_points, n_points)
    data = np.array([x, x+1, x+2])

    # theoretical result
    x_true = np.zeros(np.shape(x))
    data_true_processed = np.array([x_true, x_true, x_true])

    data_processed = detrending_normalization(data)

    for ch_idx in range(3):
        # the results are not exactly zero, so we need to test almost equal
        np.testing.assert_almost_equal(
            data_processed[ch_idx, :], data_true_processed[ch_idx, :]
        )

    # testing argument checker
    data = [[]]
    with pytest.raises(TypeError) as exc_info:
        data_processed = detrending_normalization(data)
    assert exc_info.type == TypeError


# def test_binarize_matrix():
#     """Test matrix binarization."""
    # TODO: finish it
    # n_points = 1000
    # # input data
    # x = np.linspace(0, n_points, n_points)
    # data = np.array([x, x+1, x+2])
    # data_processed = binarize_matrix(data)


def test_binary_matrix_to_string():
    """Test the conversion from binary matrix to binary string."""
    # testing correct operation
    data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    data_true_processed = "100010001"
    data_processed = binary_matrix_to_string(data)

    assert data_processed == data_true_processed

    # testing argument checker
    data = [[]]
    with pytest.raises(TypeError) as exc_info:
        data_processed = binary_matrix_to_string(data)
    assert exc_info.type == TypeError

    data = np.array([[1, 2, 1], [0, 1, 0]])
    with pytest.raises(ValueError) as exc_info:
        data_processed = binary_matrix_to_string(data)
    assert exc_info.type == ValueError

    data = np.array([[1, 1.5, 1], [0, 1, 0]])
    with pytest.raises(ValueError) as exc_info:
        data_processed = binary_matrix_to_string(data)
    assert exc_info.type == ValueError


def test_map_matrix_to_integer():
    """Test mapping binary matrix columns to integers."""
    # testing correct operation
    binary_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    true_mapping = [1, 2, 4]
    mapping = map_matrix_to_integer(binary_matrix)

    assert mapping == true_mapping

    # testing argument checker
    binary_matrix = [[]]
    with pytest.raises(TypeError) as exc_info:
        mapping = binary_matrix_to_string(binary_matrix)
    assert exc_info.type == TypeError

    no_binary_matrix = np.array([[1, 2, 1], [0, 1, 0]])
    with pytest.raises(ValueError) as exc_info:
        mapping = binary_matrix_to_string(no_binary_matrix)
    assert exc_info.type == ValueError


def test_compute_synchrony():
    """Test compute synchrony between two phase time series."""
    # testing correct operation
    n_points = 100
    t = np.linspace(0, 1, n_points)
    p_1 = t
    p_2 = t[:n_points/2]-t[n_points/2:]

    
