"""Holonomy diagnostics from macro metrics."""

from __future__ import annotations

import heapq
from typing import List, Tuple

import numpy as np


def metric_knn(d: np.ndarray, k: int) -> list[np.ndarray]:
    """For each i, return k nearest neighbors by distance (excluding i)."""
    d_arr = np.asarray(d, dtype=np.float64)
    if d_arr.ndim != 2 or d_arr.shape[0] != d_arr.shape[1]:
        raise ValueError("d must be square")
    if k <= 0:
        raise ValueError("k must be positive")
    n = d_arr.shape[0]
    neighbors: list[np.ndarray] = []
    for i in range(n):
        row = d_arr[i].copy()
        row[i] = np.inf
        finite = np.isfinite(row)
        idx = np.where(finite)[0]
        if idx.size < k:
            raise ValueError("not enough finite neighbors for kNN")
        dvals = row[idx]
        order = np.lexsort((idx, dvals))
        chosen = idx[order][:k]
        neighbors.append(chosen.astype(int))
    return neighbors


def classical_mds(D: np.ndarray, dim: int = 2) -> np.ndarray:
    """Classical MDS embedding from distance matrix."""
    D_arr = np.asarray(D, dtype=np.float64)
    if D_arr.ndim != 2 or D_arr.shape[0] != D_arr.shape[1]:
        raise ValueError("D must be square")
    n = D_arr.shape[0]
    if dim <= 0 or dim >= n:
        raise ValueError("dim must be in [1, n-1]")

    D2 = D_arr ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    pos = evals > 0
    if np.sum(pos) < dim:
        raise ValueError("not enough positive eigenvalues for embedding")
    vals = evals[pos][:dim]
    vecs = evecs[:, pos][:, :dim]
    coords = vecs * np.sqrt(vals)
    return coords


def procrustes_rotation(A: np.ndarray, B: np.ndarray, *, enforce_proper: bool = True) -> np.ndarray:
    """Compute orthogonal rotation aligning A to B."""
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    if A_arr.shape != B_arr.shape:
        raise ValueError("A and B must have same shape")
    if A_arr.ndim != 2:
        raise ValueError("A and B must be 2D")

    A_center = A_arr - A_arr.mean(axis=0, keepdims=True)
    B_center = B_arr - B_arr.mean(axis=0, keepdims=True)
    M = A_center.T @ B_center
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if enforce_proper and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def local_embeddings_from_metric(
    d: np.ndarray,
    knn: list[np.ndarray],
    dim: int = 2,
    *,
    expand_hops: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute local embeddings for each node and its neighbors."""
    d_arr = np.asarray(d, dtype=np.float64)
    n = d_arr.shape[0]
    neighborhoods: list[np.ndarray] = []
    coords_list: list[np.ndarray] = []
    for i in range(n):
        if expand_hops <= 0:
            neigh = np.concatenate(([i], knn[i]))
        else:
            seen = {i}
            frontier = {int(x) for x in knn[i]}
            seen.update(frontier)
            for _ in range(expand_hops):
                next_frontier = set()
                for node in frontier:
                    next_frontier.update(int(x) for x in knn[node])
                next_frontier -= seen
                seen.update(next_frontier)
                frontier = next_frontier
            neigh = np.array(sorted(seen), dtype=int)
        D_sub = d_arr[np.ix_(neigh, neigh)]
        coords = classical_mds(D_sub, dim=dim)
        neighborhoods.append(neigh)
        coords_list.append(coords)
    return neighborhoods, coords_list


def transport_rotation(
    i: int,
    j: int,
    neighborhoods: list[np.ndarray],
    coords_list: list[np.ndarray],
    *,
    min_overlap: int = 4,
) -> np.ndarray | None:
    """Compute transport rotation between local embeddings at i and j."""
    Ni = neighborhoods[i]
    Nj = neighborhoods[j]
    set_i = {int(x) for x in Ni}
    set_j = {int(x) for x in Nj}
    overlap = sorted(set_i & set_j)
    if len(overlap) < min_overlap:
        return None

    idx_i = {int(node): idx for idx, node in enumerate(Ni)}
    idx_j = {int(node): idx for idx, node in enumerate(Nj)}

    A = np.array([coords_list[i][idx_i[node]] for node in overlap], dtype=np.float64)
    B = np.array([coords_list[j][idx_j[node]] for node in overlap], dtype=np.float64)

    if np.linalg.matrix_rank(A - A.mean(axis=0)) < 2:
        return None
    if np.linalg.matrix_rank(B - B.mean(axis=0)) < 2:
        return None

    return procrustes_rotation(A, B, enforce_proper=True)


def sample_triangles_from_knn(
    knn: list[np.ndarray],
    *,
    max_loops: int = 500,
    seed: int = 0,
) -> list[tuple[int, int, int]]:
    """Sample triangles from kNN adjacency."""
    n = len(knn)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in knn[i]:
            adj[i].add(int(j))
            adj[j].add(int(i))

    triangles: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in sorted(adj[i]):
            if j <= i:
                continue
            common = adj[i] & adj[j]
            for k in sorted(common):
                if k <= j:
                    continue
                triangles.append((i, j, k))

    rng = np.random.default_rng(seed)
    rng.shuffle(triangles)
    return triangles[:max_loops]


def rotation_angle(R: np.ndarray) -> float:
    """Extract rotation angle magnitude from a 2x2 rotation-like matrix."""
    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.shape != (2, 2):
        raise ValueError("R must be 2x2")
    angle = float(np.arctan2(R_arr[1, 0], R_arr[0, 0]))
    angle = abs(angle)
    if angle > np.pi:
        angle = 2 * np.pi - angle
    return angle


def holonomy_angles_for_triangles(
    triangles: list[tuple[int, int, int]],
    neighborhoods: list[np.ndarray],
    coords_list: list[np.ndarray],
    *,
    min_overlap: int = 4,
) -> np.ndarray:
    """Compute holonomy angles for triangle loops."""
    angles = []
    for x, y, z in triangles:
        R_xy = transport_rotation(x, y, neighborhoods, coords_list, min_overlap=min_overlap)
        R_yz = transport_rotation(y, z, neighborhoods, coords_list, min_overlap=min_overlap)
        R_zx = transport_rotation(z, x, neighborhoods, coords_list, min_overlap=min_overlap)
        if R_xy is None or R_yz is None or R_zx is None:
            continue
        H = R_xy @ R_yz @ R_zx
        angles.append(rotation_angle(H))
    return np.asarray(angles, dtype=np.float64)
