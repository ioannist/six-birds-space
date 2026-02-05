"""Dimension and scaling diagnostics."""

from __future__ import annotations

import numpy as np

from ..packaging import Q_f


def shannon_entropy(p: np.ndarray, *, base: float = np.e) -> float:
    """Shannon entropy H(p) with 0 log 0 = 0."""
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim != 1:
        raise ValueError("p must be 1D")
    mask = p_arr > 0.0
    if not np.any(mask):
        return 0.0
    logs = np.log(p_arr[mask])
    if base != np.e:
        logs = logs / np.log(base)
    return float(-np.sum(p_arr[mask] * logs))


def entropy_at_scale(mu: np.ndarray, C: np.ndarray, *, base: float = np.e) -> float:
    """Entropy of the macro distribution Q_f(mu) = mu @ C."""
    nu = Q_f(np.asarray(mu, dtype=np.float64), np.asarray(C, dtype=np.float64))
    return shannon_entropy(nu, base=base)


def typical_spacing(d: np.ndarray) -> float:
    """Median of nearest-neighbor distances."""
    d_arr = np.asarray(d, dtype=np.float64)
    if d_arr.ndim != 2 or d_arr.shape[0] != d_arr.shape[1]:
        raise ValueError("d must be square")
    n = d_arr.shape[0]
    if n == 0:
        raise ValueError("empty distance matrix")

    mins = []
    for i in range(n):
        row = d_arr[i].copy()
        row[i] = np.inf
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        mins.append(float(np.min(row)))
    if not mins:
        raise ValueError("no finite off-diagonal distances")
    eps = float(np.median(mins))
    if eps <= 0.0:
        raise ValueError("non-positive typical spacing")
    return eps


def information_dimension_slope(
    entropies: np.ndarray,
    epsilons: np.ndarray,
    *,
    fit_slice: slice | None = None,
) -> dict:
    """Fit H ≈ a * log(1/epsilon) + b and return slope/intercept/r2."""
    H = np.asarray(entropies, dtype=np.float64)
    eps = np.asarray(epsilons, dtype=np.float64)
    if H.ndim != 1 or eps.ndim != 1:
        raise ValueError("entropies and epsilons must be 1D")
    if H.shape[0] != eps.shape[0]:
        raise ValueError("entropies and epsilons length mismatch")
    if fit_slice is not None:
        H = H[fit_slice]
        eps = eps[fit_slice]
    if H.size == 0:
        raise ValueError("no points for fit")

    x = np.log(1.0 / eps)
    y = H
    coeffs = np.polyfit(x, y, deg=1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    r2 = float("nan")
    if x.size >= 2:
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {"slope": slope, "intercept": intercept, "r2": r2}


def ball_growth_curve(
    d: np.ndarray,
    radii: np.ndarray,
    *,
    centers: np.ndarray | None = None,
) -> dict:
    """Compute mean ball counts over radii."""
    d_arr = np.asarray(d, dtype=np.float64)
    if d_arr.ndim != 2 or d_arr.shape[0] != d_arr.shape[1]:
        raise ValueError("d must be square")
    n = d_arr.shape[0]

    if centers is None:
        centers_idx = np.arange(n)
    else:
        centers_idx = np.asarray(centers, dtype=int)

    radii_arr = np.asarray(radii, dtype=np.float64)
    mean_counts = []
    counts_per_center = []
    for r in radii_arr:
        counts = []
        for c in centers_idx:
            row = d_arr[c]
            counts.append(int(np.sum(row <= r)))
        counts_arr = np.asarray(counts, dtype=np.float64)
        mean_counts.append(float(np.mean(counts_arr)))
        counts_per_center.append(counts_arr)

    return {
        "radii": radii_arr,
        "mean_counts": np.asarray(mean_counts, dtype=np.float64),
        "counts_per_center": counts_per_center,
    }


def ball_growth_dimension(
    d: np.ndarray,
    *,
    n_radii: int = 12,
    centers: int = 32,
    r_min_quantile: float = 0.05,
    r_max_quantile: float = 0.5,
) -> dict:
    """Estimate ball-growth dimension from a distance matrix."""
    d_arr = np.asarray(d, dtype=np.float64)
    if d_arr.ndim != 2 or d_arr.shape[0] != d_arr.shape[1]:
        raise ValueError("d must be square")
    n = d_arr.shape[0]
    if n == 0:
        raise ValueError("empty distance matrix")

    mask = np.isfinite(d_arr)
    mask &= ~np.eye(n, dtype=bool)
    vals = d_arr[mask]
    if vals.size == 0:
        raise ValueError("no finite off-diagonal distances")

    r_min = float(np.quantile(vals, r_min_quantile))
    r_max = float(np.quantile(vals, r_max_quantile))
    if r_min <= 0.0:
        r_min = float(np.min(vals[vals > 0]))
    if r_max <= r_min:
        r_max = float(np.max(vals))

    radii = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_radii))

    rng = np.random.default_rng(0)
    if centers >= n:
        center_idx = np.arange(n)
    else:
        center_idx = rng.choice(n, size=centers, replace=False)

    curve = ball_growth_curve(d_arr, radii, centers=center_idx)
    mean_counts = curve["mean_counts"]

    valid = (mean_counts >= 2.0) & (mean_counts <= n / 2.0)
    x = np.log(radii[valid])
    y = np.log(mean_counts[valid])

    if x.size < 2:
        x = np.log(radii)
        y = np.log(mean_counts)

    slope = 0.0
    if x.size >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        slope = float(coeffs[0])

    return {
        "slope": slope,
        "used_points": int(x.size),
        "radii": radii,
        "mean_counts": mean_counts,
    }
