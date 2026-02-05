"""Core SBT packaging operators for finite Markov chains."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .metrics_tv import tv_rows


def make_C(labels: np.ndarray, m: Optional[int] = None) -> np.ndarray:
    """Build the coarse map matrix C with shape (n, m)."""
    labels_arr = np.asarray(labels, dtype=int)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be a 1D array")
    n = labels_arr.shape[0]
    if m is None:
        m = int(labels_arr.max()) + 1 if n > 0 else 0
    assert_labels_valid(labels_arr, m)
    C = np.zeros((n, m), dtype=np.float64)
    if n == 0 or m == 0:
        return C
    C[np.arange(n), labels_arr] = 1.0
    return C


def Q_f(mu: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Coarse map Q_f applied to micro distribution mu."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    return mu_arr @ np.asarray(C, dtype=np.float64)


def U_f(nu: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Lift map U_f applied to macro distribution nu."""
    nu_arr = np.asarray(nu, dtype=np.float64)
    return nu_arr @ np.asarray(U, dtype=np.float64)


def markov_power(P: np.ndarray, tau: int) -> np.ndarray:
    """Compute P^tau using matrix_power."""
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square matrix")
    return np.linalg.matrix_power(P_arr, tau)


def E_matrix(P: np.ndarray, tau: int, C: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Compute E = P^tau @ C @ U."""
    P_tau = markov_power(P, tau)
    C_arr = np.asarray(C, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)
    return P_tau @ C_arr @ U_arr


def E_apply(mu: np.ndarray, P: np.ndarray, tau: int, C: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply E_{tau,f} to mu."""
    E = E_matrix(P, tau, C, U)
    return np.asarray(mu, dtype=np.float64) @ E


def idempotence_defect_delta(P: np.ndarray, tau: int, C: np.ndarray, U: np.ndarray) -> float:
    """Idempotence defect delta via extreme-point formula."""
    E = E_matrix(P, tau, C, U)
    diff = E @ E - E
    defects = 0.5 * np.sum(np.abs(diff), axis=1)
    return float(np.max(defects)) if defects.size else 0.0


def prototype_stabilities(P: np.ndarray, tau: int, C: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Prototype stability vector s(x) = TV(E(u_x), u_x)."""
    U_arr = np.asarray(U, dtype=np.float64)
    E = E_matrix(P, tau, C, U_arr)
    UE = U_arr @ E
    return tv_rows(UE, U_arr)


def assert_row_stochastic(P: np.ndarray, tol: float = 1e-9) -> None:
    """Raise if P is not row-stochastic within tolerance."""
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square matrix")
    if np.any(P_arr < -tol):
        raise ValueError("P has negative entries")
    row_sums = P_arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError("P rows do not sum to 1 within tolerance")


def assert_labels_valid(labels: np.ndarray, m: int) -> None:
    """Raise if labels are not integers in [0, m-1]."""
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be 1D")
    if labels_arr.size == 0:
        return
    if labels_arr.min() < 0 or labels_arr.max() >= m:
        raise ValueError("labels must be in range [0, m-1]")


def assert_prototypes_valid(U: np.ndarray, tol: float = 1e-9) -> None:
    """Raise if prototypes are not valid distributions per row."""
    U_arr = np.asarray(U, dtype=np.float64)
    if U_arr.ndim != 2:
        raise ValueError("U must be 2D")
    if np.any(U_arr < -tol):
        raise ValueError("U has negative entries")
    row_sums = U_arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError("U rows do not sum to 1 within tolerance")
