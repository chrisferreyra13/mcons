"""Tests for usuful functions in complexity"""

# Author: Christian ferreyra, chrisferreyra13@gmail.com

from matplotlib.pyplot import xticks
import numpy as np

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
    n_points = 1000
    x = np.linspace(0, n_points, n_points)
    x_true = np.zeros(np.shape(x))
    data = np.array([x, x+1, x+2])
    data_processed = detrending_normalization(data)
    data_true_processed = np.array([x_true, x_true, x_true])
    for ch_idx in range(3):
        np.testing.assert_almost_equal(
            data_processed[ch_idx, :], data_true_processed[ch_idx, :]
        )
