"""Total variation distance utilities."""

from __future__ import annotations

import numpy as np


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two 1D distributions."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.ndim != 1 or q_arr.ndim != 1:
        raise ValueError("tv expects 1D arrays")
    if p_arr.shape != q_arr.shape:
        raise ValueError("tv expects arrays of the same shape")
    return 0.5 * float(np.sum(np.abs(p_arr - q_arr)))


def tv_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Row-wise total variation distance for 2D arrays."""
    P_arr = np.asarray(P, dtype=np.float64)
    Q_arr = np.asarray(Q, dtype=np.float64)
    if P_arr.ndim != 2 or Q_arr.ndim != 2:
        raise ValueError("tv_rows expects 2D arrays")
    if P_arr.shape != Q_arr.shape:
        raise ValueError("tv_rows expects arrays of the same shape")
    return 0.5 * np.sum(np.abs(P_arr - Q_arr), axis=1)


def is_distribution(p: np.ndarray, tol: float = 1e-9) -> bool:
    """Return True if p is (approximately) a distribution."""
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim != 1:
        return False
    if np.any(p_arr < -tol):
        return False
    return abs(float(np.sum(p_arr)) - 1.0) <= tol
