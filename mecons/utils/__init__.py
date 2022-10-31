"""Usuful functions for MeCons metrics."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# License: BSD-3-Clause

from .binary import (binarize_hilbert_amplitude, binary_matrix_to_string,
                     map_matrix_to_integer, compute_synchrony_matrix,
                     create_random_binary_matrix)

from .preprocessing import detrending_normalization
