"""Tests for complexity metrics based on compressibility."""

# Author: Christian ferreyra, chrisferreyra13@gmail.com
# Date: 09.2022
# License : BSD-3-Clause

import numpy as np
from numpy.random import default_rng
import pytest

from mecons.complexity.compressibility import (
    _lempel_ziv_welch_compression,
    compute_lempel_ziv_complexity
)


def test_lempel_ziv_welch_compression():
    """Test Lempel-Ziv-Welch compression."""
    # testing correct operation
    binary_str = "0101"
    dict_len = _lempel_ziv_welch_compression(binary_str)

    assert dict_len == 3

    # testeting argument checker
    no_binary_str = 100101000
    with pytest.raises(TypeError) as exc_info:
        dict_len = _lempel_ziv_welch_compression(no_binary_str)
    assert exc_info.type == TypeError


def test_compute_lempel_ziv_complexity():
    """Test computation of Lempel Ziv complexity metric."""
    # testing correct operation
    # two simple sinusoidal signals should have
    # lower LZc than more complex ones
    f = 1
    n_points = 100
    t = np.linspace(0, 1, n_points)
    x1 = np.sin(t*2*np.pi*f)
    x2 = np.cos(t*2*np.pi*f)
    data = np.array([x1, x2])
    lzc_1 = compute_lempel_ziv_complexity(data)

    rgn = default_rng()
    noise = 0.3*rgn.standard_normal(n_points)
    x1 = x1+noise
    x2 = x2+noise
    data = np.array([x1, x2])
    lzc_2 = compute_lempel_ziv_complexity(data)

    assert lzc_1 < lzc_2

    # testing argument checker
    data = [[]]
    with pytest.raises(TypeError) as exc_info:
        lzc = compute_lempel_ziv_complexity(data)
    assert exc_info.type == TypeError
