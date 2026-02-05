"""kNN-based kernels from point clouds."""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None


def _pairwise_sq_dists(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sum(diff * diff, axis=2)


def knn_points(
    points: np.ndarray,
    k: int,
    sigma: float,
    *,
    self_loop: float = 1e-6,
    symmetrize: bool = True,
) -> np.ndarray:
    """Build a Markov kernel from kNN Gaussian weights."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array")
    n = pts.shape[0]
    if n == 0:
        raise ValueError("points must be non-empty")
    if n == 1:
        return np.array([[1.0]], dtype=np.float64)
    if k <= 0:
        raise ValueError("k must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    k_eff = min(k, n - 1)
    W = np.zeros((n, n), dtype=np.float64)

    if cKDTree is not None:
        tree = cKDTree(pts)
        dists, idxs = tree.query(pts, k=k_eff + 1)
        if k_eff + 1 == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        for i in range(n):
            neighbors = idxs[i][1:]
            dist_sq = dists[i][1:] ** 2
            weights = np.exp(-dist_sq / (2.0 * sigma * sigma))
            W[i, neighbors] = weights
    else:
        d2 = _pairwise_sq_dists(pts)
        for i in range(n):
            d2[i, i] = np.inf
            idx = np.argpartition(d2[i], k_eff)[:k_eff]
            weights = np.exp(-d2[i, idx] / (2.0 * sigma * sigma))
            W[i, idx] = weights

    if symmetrize:
        W = W + W.T

    W[np.diag_indices_from(W)] += self_loop

    row_sums = W.sum(axis=1)
    P = np.zeros_like(W)
    for i in range(n):
        if row_sums[i] <= 0.0:
            P[i, i] = 1.0
        else:
            P[i, :] = W[i, :] / row_sums[i]
    return P
