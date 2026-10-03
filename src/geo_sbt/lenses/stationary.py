"""Stationary distribution computation."""

from __future__ import annotations

import numpy as np

from ..packaging import assert_row_stochastic


def stationary_distribution(
    P: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 20000,
) -> np.ndarray:
    """Compute a stationary distribution using lazy power iteration.

    Lazification preserves stationary distributions and removes periodic
    oscillation. Exhaustion raises rather than returning a nonstationary vector.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square matrix")
    assert_row_stochastic(P_arr)
    if not np.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("tol and max_iter must be positive")
    n = P_arr.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    mu = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(max_iter):
        evolved = mu @ P_arr
        if np.sum(np.abs(evolved - mu)) <= tol:
            return mu / mu.sum()
        mu = .5 * (mu + evolved)
    mu = mu / mu.sum()
    if np.sum(np.abs(mu @ P_arr - mu)) > tol:
        raise RuntimeError("stationary iteration did not converge within max_iter")
    return mu
