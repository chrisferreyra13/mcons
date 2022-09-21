"""Tests for preprocessing usuful functions."""

# Author: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 09.2022
# License : BSD-3-Clause

import numpy as np
import pytest

from mecons.utils.preprocessing import detrending_normalization


def test_detrending_normalization():
    """Test detrending and normalization."""
    # testing correct operation
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
