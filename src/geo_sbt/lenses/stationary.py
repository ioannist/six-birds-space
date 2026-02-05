"""Stationary distribution computation."""

from __future__ import annotations

import numpy as np


def stationary_distribution(
    P: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 20000,
) -> np.ndarray:
    """Compute stationary distribution by power iteration."""
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square matrix")
    n = P_arr.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    mu = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(max_iter):
        mu_next = mu @ P_arr
        diff = np.sum(np.abs(mu_next - mu))
        mu = mu_next
        if diff <= tol:
            break
    total = float(mu.sum())
    if total > 0:
        mu = mu / total
    mu = np.maximum(mu, 0.0)
    mu = mu / mu.sum() if mu.sum() > 0 else mu
    return mu
