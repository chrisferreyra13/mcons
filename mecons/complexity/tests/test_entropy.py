"""Tests for complexity metrics based on entropy."""

# Author: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 09.2022
# License : BSD-3-Clause

import numpy as np
from numpy.random import default_rng
import pytest

from mecons.complexity.entropy import (
    amplitude_coalition_entropy,
    synchrony_coalition_entropy,
    _compute_entropy
)


def test_compute_entropy_string():
    """Test compute entropy of a list of numbers/strings."""
    # test correct operation
    col_map = [1, 5, 5, 1, 1, 5]
    # binary_str = str(col_map)
    entropy_value = _compute_entropy(col_map)

    assert entropy_value == 1

    binary_string_list = np.array(["1", "0", "1", "1", "0", "0"])
    # binary_str = str(col_map)
    entropy_value = _compute_entropy(binary_string_list)

    assert entropy_value == 1


def test_amplitude_coalition_entropy():
    """Test computation of amplitude coalition entropy metric."""
    # test correct operation
    # two simple sinusoidal signals should have
    # lower ACE than more complex ones
    f = 1
    n_points = 100
    t = np.linspace(0, 1, n_points)
    x1 = np.sin(t*2*np.pi*f)
    x2 = np.cos(t*2*np.pi*f)
    data = np.array([x1, x2])
    ace_1 = amplitude_coalition_entropy(data)

    rgn = default_rng()
    noise = 0.3*rgn.standard_normal(n_points)
    x1 = x1+noise
    x2 = x2+noise
    data = np.array([x1, x2])
    ace_2 = amplitude_coalition_entropy(data)

    assert ace_1 < ace_2

    # testing argument checker
    data = [[]]
    with pytest.raises(TypeError) as exc_info:
        ace = amplitude_coalition_entropy(data)
    assert exc_info.type == TypeError


def test_synchrony_coalition_entropy():
    """Test computation of synchrony coalition entropy metric."""
    # test correct operation
    # two simple sinusoidal signals should have
    # lower SCE than more complex ones
    f = 1
    n_points = 100
    t = np.linspace(0, 1, n_points)
    x1 = np.sin(t*2*np.pi*f)
    x2 = np.cos(t*2*np.pi*f)
    data = np.array([x1, x2])
    sce_1 = synchrony_coalition_entropy(data)

    rgn = default_rng()
    noise = 0.3*rgn.standard_normal(n_points)
    x1 = x1+noise
    x2 = x2+noise
    data = np.array([x1, x2])
    sce_2 = synchrony_coalition_entropy(data)

    assert sce_1 < sce_2

    # testing argument checker
    data = [[]]
    with pytest.raises(TypeError) as exc_info:
        sce = synchrony_coalition_entropy(data)
    assert exc_info.type == TypeError
