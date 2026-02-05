"""Hierarchical diffusion partition (lens ladder)."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..packaging import Q_f, U_f, make_C
from .diffusion import diffusion_coordinates
from .kmeans import kmeans


def _normalize_levels(levels, n: int) -> List[int]:
    if isinstance(levels, int):
        L = levels
        if L <= 0:
            raise ValueError("levels must be positive")
        if L == 1:
            return [min(max(1, n), n)]
        m_min = 2
        m_max = min(n, 2 ** (L - 1))
        if m_max < m_min:
            m_max = m_min
        counts = []
        for i in range(L):
            t = i / (L - 1)
            val = int(round(m_min * (m_max / m_min) ** t))
            counts.append(val)
        counts[0] = max(1, counts[0])
        for i in range(1, L):
            if counts[i] <= counts[i - 1]:
                counts[i] = counts[i - 1] + 1
        counts[-1] = min(counts[-1], n)
        for i in range(L - 2, -1, -1):
            if counts[i] >= counts[i + 1]:
                counts[i] = counts[i + 1] - 1
        if counts[0] <= 0:
            counts[0] = 1
        if counts[-1] > n:
            counts[-1] = n
        return counts

    if isinstance(levels, Iterable):
        counts = [int(x) for x in levels]
        if len(counts) == 0:
            raise ValueError("levels list must be non-empty")
        if any(c <= 0 for c in counts):
            raise ValueError("cluster counts must be positive")
        if any(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)):
            raise ValueError("cluster counts must be strictly increasing")
        if counts[-1] > n:
            raise ValueError("finest cluster count must be <= n")
        return counts

    raise ValueError("levels must be int or list[int]")


def _centroids_from_labels(coords: np.ndarray, labels: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    centroids = np.zeros((m, coords.shape[1]), dtype=np.float64)
    for i in range(m):
        mask = labels == i
        if not np.any(mask):
            centroids[i] = coords[rng.integers(0, coords.shape[0])]
        else:
            centroids[i] = coords[mask].mean(axis=0)
    return centroids


def check_refinement_consistency(
    labels_coarse: np.ndarray,
    labels_fine: np.ndarray,
    refine_map: np.ndarray,
) -> bool:
    """Check labels_coarse[z] == refine_map[labels_fine[z]] for all z."""
    labels_coarse = np.asarray(labels_coarse, dtype=int)
    labels_fine = np.asarray(labels_fine, dtype=int)
    refine_map = np.asarray(refine_map, dtype=int)
    if labels_coarse.shape != labels_fine.shape:
        return False
    return np.all(labels_coarse == refine_map[labels_fine])


def hierarchical_diffusion_partition(
    P: np.ndarray,
    levels,
    n_eigs: int,
    method: str = "kmeans_merge",
    *,
    seed: int = 0,
) -> dict:
    """Build a staged lens ladder from diffusion coordinates."""
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square matrix")
    n = P_arr.shape[0]

    ms = _normalize_levels(levels, n)
    if method != "kmeans_merge":
        raise ValueError("only method='kmeans_merge' is implemented")

    coords = diffusion_coordinates(P_arr, n_eigs)
    if coords.shape[0] != n:
        raise ValueError("diffusion coordinates shape mismatch")

    labels_finest = kmeans(coords, ms[-1], seed=seed)
    labels_by_level = {ms[-1]: labels_finest}

    rng = np.random.default_rng(seed + 1)
    maps = []
    current_labels = labels_finest
    current_m = ms[-1]

    for level_index, m_coarse in enumerate(reversed(ms[:-1])):
        centroids = _centroids_from_labels(coords, current_labels, current_m, rng)
        map_current_to_coarse = kmeans(centroids, m_coarse, seed=seed + 2 + level_index)
        labels_coarse = map_current_to_coarse[current_labels]
        maps.append(map_current_to_coarse)
        labels_by_level[m_coarse] = labels_coarse
        current_labels = labels_coarse
        current_m = m_coarse

    labels_list = [labels_by_level[m] for m in ms]
    refine_maps = list(reversed(maps))

    return {
        "cluster_counts": ms,
        "labels_list": labels_list,
        "refine_maps": refine_maps,
    }


def lens_C_U_from_labels(labels: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build C from labels and return (C, U)."""
    C = make_C(labels)
    return C, np.asarray(U, dtype=np.float64)


def max_macro_identity_deviation(C: np.ndarray, U: np.ndarray) -> float:
    """Compute max_x max_abs(Q_f(U_f(e_x)) - e_x)."""
    U_arr = np.asarray(U, dtype=np.float64)
    m = U_arr.shape[0]
    max_dev = 0.0
    for x in range(m):
        nu = np.zeros(m, dtype=np.float64)
        nu[x] = 1.0
        mu = U_f(nu, U_arr)
        nu_back = Q_f(mu, C)
        dev = float(np.max(np.abs(nu_back - nu)))
        if dev > max_dev:
            max_dev = dev
    return max_dev
