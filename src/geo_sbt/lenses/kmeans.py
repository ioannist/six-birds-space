"""Deterministic k-means clustering."""

from __future__ import annotations

import numpy as np


def _kmeans_single(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = X.shape[0]
    if k > n:
        raise ValueError("k must be <= number of points")
    centroids = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                centroids[j] = X[rng.integers(0, n)]
            else:
                centroids[j] = X[mask].mean(axis=0)
    inertia = float(np.sum((X - centroids[labels]) ** 2))
    return labels, centroids, inertia


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 3,
    max_iter: int = 100,
) -> np.ndarray:
    """Simple deterministic k-means clustering.

    Returns labels in 0..k-1. Empty clusters are re-seeded deterministically.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2D")
    n = X_arr.shape[0]
    if n == 0:
        raise ValueError("X must be non-empty")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        raise ValueError("k must be <= number of points")
    if n_init <= 0:
        raise ValueError("n_init must be positive")

    rng = np.random.default_rng(seed)
    best_labels = None
    best_inertia = None
    for _ in range(n_init):
        labels, _, inertia = _kmeans_single(X_arr, k, rng, max_iter)
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels
