"""Macro dynamics and emergent metric pipeline."""

from __future__ import annotations

import heapq
from typing import List, Tuple

import numpy as np

from ..packaging import markov_power

try:  # optional
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra  # type: ignore
except Exception:  # pragma: no cover
    sp_dijkstra = None


def macro_kernel(P: np.ndarray, tau: int, C: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Return macro kernel P_hat = U @ (P^tau) @ C."""
    P_tau = markov_power(P, tau)
    U_arr = np.asarray(U, dtype=np.float64)
    C_arr = np.asarray(C, dtype=np.float64)
    return U_arr @ P_tau @ C_arr


def cost_matrix_from_kernel(
    P_hat: np.ndarray,
    *,
    eta: float = 1e-12,
    eps_edge: float = 0.0,
    symmetrize: str = "none",
) -> np.ndarray:
    """Build a cost matrix from a kernel."""
    P_arr = np.asarray(P_hat, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P_hat must be a square matrix")

    if symmetrize == "none":
        W = P_arr
    elif symmetrize == "weight_avg":
        W = 0.5 * (P_arr + P_arr.T)
    elif symmetrize in {"min_cost", "avg_cost"}:
        W = P_arr
    else:
        raise ValueError("symmetrize must be one of: none, weight_avg, min_cost, avg_cost")

    cost = np.full_like(W, np.inf, dtype=np.float64)
    mask = W > eps_edge
    cost[mask] = -np.log(np.maximum(W[mask], eta))
    np.fill_diagonal(cost, 0.0)

    if symmetrize == "min_cost":
        cost = np.minimum(cost, cost.T)
        np.fill_diagonal(cost, 0.0)
    elif symmetrize == "avg_cost":
        both = np.isfinite(cost) & np.isfinite(cost.T)
        avg = np.full_like(cost, np.inf)
        avg[both] = 0.5 * (cost[both] + cost.T[both])
        only = np.isfinite(cost) & ~np.isfinite(cost.T)
        avg[only] = cost[only]
        only_t = np.isfinite(cost.T) & ~np.isfinite(cost)
        avg[only_t] = cost.T[only_t]
        cost = avg
        np.fill_diagonal(cost, 0.0)

    return cost


def adjacency_from_cost(cost: np.ndarray) -> List[List[Tuple[int, float]]]:
    """Build adjacency list from finite costs."""
    cost_arr = np.asarray(cost, dtype=np.float64)
    if cost_arr.ndim != 2 or cost_arr.shape[0] != cost_arr.shape[1]:
        raise ValueError("cost must be a square matrix")
    n = cost_arr.shape[0]
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = cost_arr[i, j]
            if np.isfinite(w):
                adj[i].append((j, float(w)))
    return adj


def _dijkstra_from_source(adj: List[List[Tuple[int, float]]], src: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[src] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def all_pairs_shortest_path(cost: np.ndarray) -> np.ndarray:
    """Compute all-pairs shortest path distances using Dijkstra."""
    cost_arr = np.asarray(cost, dtype=np.float64)
    if cost_arr.ndim != 2 or cost_arr.shape[0] != cost_arr.shape[1]:
        raise ValueError("cost must be a square matrix")

    if sp_dijkstra is not None:
        try:
            return sp_dijkstra(cost_arr, directed=True, unweighted=False)
        except Exception:
            pass

    adj = adjacency_from_cost(cost_arr)
    n = cost_arr.shape[0]
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dists[i] = _dijkstra_from_source(adj, i)
    return dists


def distortion_between_scales(
    d_fine: np.ndarray,
    d_coarse: np.ndarray,
    r_fine_to_coarse: np.ndarray,
    *,
    rescale: str | None = None,
) -> dict:
    """Compute distortion between fine distances and projected coarse distances."""
    d_f = np.asarray(d_fine, dtype=np.float64)
    d_c = np.asarray(d_coarse, dtype=np.float64)
    r = np.asarray(r_fine_to_coarse, dtype=int)
    if d_f.ndim != 2 or d_c.ndim != 2:
        raise ValueError("d_fine and d_coarse must be 2D")
    if d_f.shape[0] != d_f.shape[1] or d_c.shape[0] != d_c.shape[1]:
        raise ValueError("distance matrices must be square")
    if r.ndim != 1 or r.shape[0] != d_f.shape[0]:
        raise ValueError("refinement map length mismatch")

    proj = d_c[r[:, None], r[None, :]]
    mask = np.isfinite(d_f) & np.isfinite(proj)
    if not np.any(mask):
        return {"max_abs_diff": float("inf"), "alpha": 1.0, "finite_pairs": 0}

    alpha = 1.0
    if rescale == "lstsq":
        num = float(np.sum(d_f[mask] * proj[mask]))
        den = float(np.sum(proj[mask] ** 2))
        alpha = num / den if den > 0.0 else 1.0
    elif rescale == "median":
        valid = mask & (proj > 0)
        ratios = d_f[valid] / proj[valid]
        alpha = float(np.median(ratios)) if ratios.size else 1.0
    elif rescale is None:
        alpha = 1.0
    else:
        raise ValueError("rescale must be None, 'lstsq', or 'median'")

    diff = np.abs(d_f - alpha * proj)
    max_abs_diff = float(np.max(diff[mask]))
    return {
        "max_abs_diff": max_abs_diff,
        "alpha": alpha,
        "finite_pairs": int(np.sum(mask)),
    }
