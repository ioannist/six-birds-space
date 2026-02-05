"""Constraint modifiers for kernels."""

from __future__ import annotations

import math
import numpy as np


def anisotropic_gate(P: np.ndarray, direction: str, strength: float) -> np.ndarray:
    """Apply a directional feasibility constraint to a grid kernel.

    Assumes states are ordered row-major on an n_side x n_side grid.
    direction in {'east','west','north','south'}.
    strength in [0,1]: 0=no change, 1=fully suppress moves opposite to direction.
    """
    if strength < 0.0 or strength > 1.0:
        raise ValueError("strength must be in [0, 1]")
    direction = direction.lower()
    if direction not in {"east", "west", "north", "south"}:
        raise ValueError("direction must be one of east, west, north, south")

    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square matrix")

    n = P_arr.shape[0]
    n_side = int(round(math.sqrt(n)))
    if n_side * n_side != n:
        raise ValueError("P size does not correspond to a square grid")

    def idx(r: int, c: int) -> int:
        return r * n_side + c

    def coords(i: int) -> tuple[int, int]:
        return divmod(i, n_side)

    newP = P_arr.copy()

    for i in range(n):
        r, c = coords(i)
        opp = None
        if direction == "east" and c > 0:
            opp = idx(r, c - 1)
        elif direction == "west" and c < n_side - 1:
            opp = idx(r, c + 1)
        elif direction == "north" and r < n_side - 1:
            opp = idx(r + 1, c)
        elif direction == "south" and r > 0:
            opp = idx(r - 1, c)
        if opp is not None:
            newP[i, opp] *= (1.0 - strength)

        row_sum = newP[i].sum()
        if row_sum <= 0.0:
            newP[i, :] = 0.0
            newP[i, i] = 1.0
        else:
            newP[i, :] /= row_sum

    return newP
