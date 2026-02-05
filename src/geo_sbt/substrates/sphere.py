"""Sphere point sampling utilities."""

from __future__ import annotations

import numpy as np


def sphere_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points approximately uniformly on the unit sphere S^2."""
    if n <= 0:
        raise ValueError("n must be positive")
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1)
    while np.any(norms == 0.0):
        mask = norms == 0.0
        pts[mask] = rng.normal(size=(mask.sum(), 3))
        norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, None]
