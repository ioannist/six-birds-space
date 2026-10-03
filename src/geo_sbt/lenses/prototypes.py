"""Prototype constructions for lenses."""

from __future__ import annotations

import numpy as np

from ..packaging import assert_labels_valid
from ..metrics_tv import is_distribution


def prototypes_uniform(labels: np.ndarray, m: int | None = None) -> np.ndarray:
    """Uniform-on-block prototypes."""
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be 1D")
    n = labels_arr.shape[0]
    if m is None:
        m = int(labels_arr.max()) + 1 if n > 0 else 0
    assert_labels_valid(labels_arr, m)
    labels_arr = labels_arr.astype(int)
    U = np.zeros((m, n), dtype=np.float64)
    for x in range(m):
        idx = np.where(labels_arr == x)[0]
        if idx.size == 0:
            raise ValueError(f"empty block for label {x}")
        U[x, idx] = 1.0 / float(idx.size)
    return U


def prototypes_stationary_conditional(
    labels: np.ndarray,
    pi: np.ndarray,
    m: int | None = None,
) -> np.ndarray:
    """Stationary-conditional prototypes."""
    labels_arr = np.asarray(labels)
    pi_arr = np.asarray(pi, dtype=np.float64)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be 1D")
    if pi_arr.ndim != 1:
        raise ValueError("pi must be 1D")
    if not is_distribution(pi_arr) or np.any(pi_arr < 0):
        raise ValueError("pi must be a probability distribution")
    if labels_arr.shape[0] != pi_arr.shape[0]:
        raise ValueError("labels and pi must have same length")
    n = labels_arr.shape[0]
    if m is None:
        m = int(labels_arr.max()) + 1 if n > 0 else 0
    assert_labels_valid(labels_arr, m)
    labels_arr = labels_arr.astype(int)
    U = np.zeros((m, n), dtype=np.float64)
    for x in range(m):
        idx = np.where(labels_arr == x)[0]
        if idx.size == 0:
            raise ValueError(f"empty block for label {x}")
        weight = pi_arr[idx].sum()
        if weight <= 0.0:
            raise ValueError(f"block {x} has zero stationary mass")
        U[x, idx] = pi_arr[idx] / weight
    return U
