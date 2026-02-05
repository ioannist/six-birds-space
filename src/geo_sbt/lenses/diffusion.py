"""Diffusion / spectral coordinates."""

from __future__ import annotations

import numpy as np

try:  # optional
    from scipy.sparse.linalg import eigsh  # type: ignore
except Exception:  # pragma: no cover
    eigsh = None


def diffusion_coordinates(P: np.ndarray, n_eigs: int) -> np.ndarray:
    """Compute diffusion/spectral coordinates for a kernel P.

    Uses symmetric normalized Laplacian constructed from W = (P + P.T) / 2.
    Returns coordinates with shape (n, n_eigs) excluding the trivial eigenvector.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square matrix")
    n = P_arr.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    k = min(max(n_eigs, 0), max(n - 1, 0))
    if k == 0:
        return np.zeros((n, 0), dtype=np.float64)

    W = 0.5 * (P_arr + P_arr.T)
    d = W.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_d = np.where(d > 0.0, 1.0 / np.sqrt(d), 0.0)
    A = (W * inv_sqrt_d[None, :]) * inv_sqrt_d[:, None]
    L = np.eye(n, dtype=np.float64) - A

    if eigsh is not None and n > k + 1:
        try:
            evals, evecs = eigsh(L, k=min(k + 1, n - 1), which="SM")
        except Exception:
            evals, evecs = np.linalg.eigh(L)
    else:
        evals, evecs = np.linalg.eigh(L)

    order = np.argsort(evals)
    evecs = evecs[:, order]

    coords = evecs[:, 1 : 1 + k]
    norms = np.linalg.norm(coords, axis=1)
    nonzero = norms > 0
    coords[nonzero] = coords[nonzero] / norms[nonzero, None]
    return coords
