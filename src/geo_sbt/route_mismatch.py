"""Route mismatch (RM) machinery for SBT-style operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .packaging import E_matrix

PackKind = Literal["operator", "kernel", "metric"]


@dataclass(frozen=True)
class Pack:
    """Lightweight wrapper around a linear operator or metric."""

    kind: PackKind
    data: np.ndarray
    name: str = ""

    def validate(self) -> None:
        """Validate shape based on kind."""
        data = np.asarray(self.data, dtype=np.float64)
        if self.kind in {"operator", "kernel", "metric"}:
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError("Pack data must be a square 2D matrix")

    def then(self, other: "Pack") -> "Pack":
        """Return the Pack representing self applied then other."""
        if self.kind != other.kind:
            raise ValueError("Pack kinds must match for composition")
        if self.kind == "metric":
            raise NotImplementedError("Metric composition is not defined")
        a = np.asarray(self.data, dtype=np.float64)
        b = np.asarray(other.data, dtype=np.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("Pack data shapes are not compatible for composition")
        return Pack(kind=self.kind, data=a @ b, name=self.name or other.name)


def d_tv_sup(A: np.ndarray, B: np.ndarray) -> float:
    """Operator distance: max TV over extreme points."""
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    if A_arr.shape != B_arr.shape:
        raise ValueError("A and B must have the same shape")
    if A_arr.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    row_l1 = np.sum(np.abs(A_arr - B_arr), axis=1)
    return 0.5 * float(np.max(row_l1)) if row_l1.size else 0.0


def d_fro(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius distance between matrices."""
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    if A_arr.shape != B_arr.shape:
        raise ValueError("A and B must have the same shape")
    if A_arr.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    return float(np.linalg.norm(A_arr - B_arr, ord="fro"))


def route_mismatch(
    pack_direct: Pack,
    pack_first: Pack,
    pack_second: Pack,
    distance: str = "tv_sup",
) -> float:
    """Compute RM between direct pack and indirect (first then second)."""
    for pack in (pack_direct, pack_first, pack_second):
        pack.validate()
    if pack_direct.kind != pack_first.kind or pack_direct.kind != pack_second.kind:
        raise ValueError("All Pack kinds must match")

    indirect = pack_first.then(pack_second)
    if distance == "tv_sup":
        return d_tv_sup(pack_direct.data, indirect.data)
    if distance == "fro":
        return d_fro(pack_direct.data, indirect.data)
    raise ValueError(f"Unknown distance: {distance}")


def closure_pack(
    P: np.ndarray,
    tau: int,
    C: np.ndarray,
    U: np.ndarray,
    *,
    name: str = "",
) -> Pack:
    """Build a closure pack from E_matrix."""
    E = E_matrix(P, tau, C, U)
    return Pack(kind="operator", data=E, name=name)


def rm_closure_two_step(
    P: np.ndarray,
    tau1: int,
    tau2: int,
    C_coarse: np.ndarray,
    U_coarse: np.ndarray,
    C_mid: np.ndarray,
    U_mid: np.ndarray,
    *,
    distance: str = "tv_sup",
) -> float:
    """Route mismatch for direct two-step closure vs two one-step closures."""
    direct = closure_pack(P, tau1 + tau2, C_coarse, U_coarse, name="direct")
    first = closure_pack(P, tau2, C_mid, U_mid, name="first")
    second = closure_pack(P, tau1, C_coarse, U_coarse, name="second")
    return route_mismatch(direct, first, second, distance=distance)
