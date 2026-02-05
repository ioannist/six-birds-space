"""Grid substrates."""

from __future__ import annotations

import numpy as np


def grid_2d(n_side: int, lazy: float = 0.0) -> np.ndarray:
    """2D n_side x n_side grid random walk kernel.

    lazy is probability of staying in place (0<=lazy<1).
    Remaining probability distributed uniformly over valid von Neumann neighbors.
    """
    if n_side <= 0:
        raise ValueError("n_side must be positive")
    if lazy < 0.0 or lazy >= 1.0:
        raise ValueError("lazy must be in [0, 1)")
    n = n_side * n_side
    if n == 1:
        return np.array([[1.0]], dtype=np.float64)
    P = np.zeros((n, n), dtype=np.float64)

    def idx(r: int, c: int) -> int:
        return r * n_side + c

    for r in range(n_side):
        for c in range(n_side):
            i = idx(r, c)
            neighbors = []
            if r > 0:
                neighbors.append(idx(r - 1, c))
            if r < n_side - 1:
                neighbors.append(idx(r + 1, c))
            if c > 0:
                neighbors.append(idx(r, c - 1))
            if c < n_side - 1:
                neighbors.append(idx(r, c + 1))
            deg = len(neighbors)
            if deg == 0:
                P[i, i] = 1.0
                continue
            P[i, i] = lazy
            share = (1.0 - lazy) / float(deg)
            for j in neighbors:
                P[i, j] = share
    return P
