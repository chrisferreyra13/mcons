"""Tests for usuful functions in complexity"""

# Author: Christian ferreyra, chrisferreyra13@gmail.com

import numpy as np
from scipy.signal import hilbert
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


def test_binarize_matrix():
    """Test matrix binarization."""
    # NOTE: this test should be improved
    # testing correct operation
    f = 1
    n_points = 100
    t = np.linspace(0, 1, n_points)
    x1 = np.sin(t*2*np.pi*f)
    x2 = np.cos(t*2*np.pi*f)
    data = np.array([x1, x2])

    amp1 = abs(hilbert(x1))
    amp2 = abs(hilbert(x2))
    amplitudes = np.array([amp1, amp2])
    means = [np.mean(amp1), np.mean(amp2)]

    # amplitudes of the analytic signals are not exactly 1
    # so the means are not exactly 1 -> we need to check it manually
    binary_matrix = binarize_matrix(data)

    for ch in range(2):
        is_ok = [False]*n_points
        for i in range(n_points):
            if amplitudes[ch, i] >= means[ch] and binary_matrix[ch, i] == 1:
                is_ok[i] = True
            elif amplitudes[ch, i] < means[ch] and binary_matrix[ch, i] == 0:
                is_ok[i] = True

        assert all(is_ok)

    # testing argument checker
    data = [[]]
    with pytest.raises(TypeError) as exc_info:
        data_processed = binarize_matrix(data)
    assert exc_info.type == TypeError


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
    true_mapping = np.array([1, 2, 4])
    mapping = map_matrix_to_integer(binary_matrix)

    np.testing.assert_array_equal(mapping, true_mapping)

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
    # synchrony threshold (default)
    delta = 0.8
    p1 = np.array([0, 1, 2, 3])
    p2 = np.array([0-delta, 1-0.5*delta, 0, 2+delta])
    true_synchrony = np.array([0, 1, 0, 1])
    binary_synchrony = _compute_synchrony(p1, p2)

    np.testing.assert_array_equal(binary_synchrony, true_synchrony)

    # more conservative synchrony threshold
    binary_synchrony = _compute_synchrony(p1, p2, threshold=0.3)

    # if there are not equal, np raises AssertionError
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        binary_synchrony,
        true_synchrony
    )

    # testing argument checker
    p1 = []
    with pytest.raises(TypeError) as exc_info:
        binary_synchrony = _compute_synchrony(p1, p2)
    assert exc_info.type == TypeError

    p1 = np.array([0, 1, 3])
    with pytest.raises(ValueError) as exc_info:
        binary_synchrony = _compute_synchrony(p1, p2)
    assert exc_info.type == ValueError


def test_compute_synchrony_matrix():
    """Test compute binary synchrony matrix."""
    # testing correct operation
    # synchronized in every point
    f = 1
    n_points = 100
    delta_t = 0.25
    t = np.linspace(0, 1, n_points)
    x1 = np.sin(t*2*np.pi*f)
    x2 = np.cos((t-delta_t)*2*np.pi*f)
    data = np.array([x1, x2])
    # shape = (n_channels, n_channels-1, n_points)
    true_synchrony_matrix = np.ones(
        (2, 1, n_points),
        dtype=np.int8)

    data_processed = compute_synchrony_matrix(data)

    np.testing.assert_array_equal(data_processed, true_synchrony_matrix)

    # not synchronized in every point
    delta_t = 0
    x1 = np.sin(t*2*np.pi*f)
    x2 = np.cos((t-delta_t)*2*np.pi*f)
    data = np.array([x1, x2])
    # shape = (n_channels, n_channels-1, n_points)
    true_synchrony_matrix = np.ones(
        (2, 1, n_points),
        dtype=np.int8)

    data_processed = compute_synchrony_matrix(data)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        data_processed,
        true_synchrony_matrix
    )

    # testing argument checker
    data = [[]]
    with pytest.raises(TypeError) as exc_info:
        data_processed = compute_synchrony_matrix(data)
    assert exc_info.type == TypeError


def test_create_random_binary_matrix():
    """Test creating of random binary matrix."""
    # testing correct operation
    n_rows = 2
    n_columns = 10
    binary_matrix = create_random_binary_matrix(n_rows, n_columns)

    # check dimensions
    assert binary_matrix.shape == (n_rows, n_columns)
    # check if there are only zeros and ones
    np.testing.assert_array_equal(np.unique(binary_matrix), np.array([0, 1]))

    # testing argument checker
    n_rows = 2.0
    n_columns = 10
    with pytest.raises(TypeError) as exc_info:
        binary_matrix = create_random_binary_matrix(n_rows, n_columns)
    assert exc_info.type == TypeError
