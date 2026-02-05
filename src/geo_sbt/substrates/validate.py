"""Validation utilities for Markov kernels."""

from __future__ import annotations

from collections import deque

import numpy as np


def validate_kernel(
    P: np.ndarray,
    *,
    tol: float = 1e-9,
    eps_edge: float = 1e-12,
    check_connected: bool = True,
) -> dict:
    """Validate that P is a row-stochastic Markov kernel.

    Returns a dict with diagnostics.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square matrix")
    n = P_arr.shape[0]

    row_sums = P_arr.sum(axis=1)
    max_abs_row_sum_err = float(np.max(np.abs(row_sums - 1.0))) if n > 0 else 0.0
    min_entry = float(np.min(P_arr)) if n > 0 else 0.0
    nnz = int(np.sum(P_arr > eps_edge))
    density = float(nnz) / float(n * n) if n > 0 else 0.0

    connected = True
    if check_connected and n > 0:
        adj = [set() for _ in range(n)]
        mask = P_arr > eps_edge
        for i in range(n):
            for j in range(n):
                if mask[i, j] or mask[j, i]:
                    adj[i].add(j)
                    adj[j].add(i)
        visited = set([0])
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        connected = len(visited) == n

    return {
        "n": n,
        "max_abs_row_sum_err": max_abs_row_sum_err,
        "min_entry": min_entry,
        "nnz": nnz,
        "density": density,
        "connected": connected,
    }
