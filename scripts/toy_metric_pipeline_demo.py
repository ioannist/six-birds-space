from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geo_sbt.geometry.metric import (
    all_pairs_shortest_path,
    cost_matrix_from_kernel,
    distortion_between_scales,
    macro_kernel,
)
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition
from geo_sbt.lenses.prototypes import prototypes_uniform
from geo_sbt.packaging import make_C
from geo_sbt.substrates.grid import grid_2d


def _mean_finite_offdiag(d: np.ndarray) -> float:
    n = d.shape[0]
    mask = np.isfinite(d)
    mask &= ~np.eye(n, dtype=bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(d[mask]))


def main() -> None:
    P = grid_2d(n_side=20, lazy=0.5)
    tau = 5
    ladder = hierarchical_diffusion_partition(
        P,
        levels=[4, 8, 16, 32, 64, 128],
        n_eigs=6,
        method="kmeans_merge",
        seed=0,
    )

    labels_list = ladder["labels_list"]
    refine_maps = ladder["refine_maps"]
    counts = ladder["cluster_counts"]

    dists = []
    for idx, labels in enumerate(labels_list):
        U = prototypes_uniform(labels)
        C = make_C(labels)
        P_hat = macro_kernel(P, tau, C, U)
        cost = cost_matrix_from_kernel(P_hat, symmetrize="weight_avg", eps_edge=1e-15)
        d = all_pairs_shortest_path(cost)
        dists.append(d)

        mean_dist = _mean_finite_offdiag(d)
        inf_count = int(np.sum(~np.isfinite(d)))
        print(
            f"level {idx} | m={counts[idx]} | tau={tau} | "
            f"mean_finite_dist={mean_dist:.6f} | inf_count={inf_count}"
        )

    for i in range(len(refine_maps)):
        r = refine_maps[i]
        d_f = dists[i + 1]
        d_c = dists[i]
        raw = distortion_between_scales(d_f, d_c, r, rescale=None)
        fit = distortion_between_scales(d_f, d_c, r, rescale="lstsq")
        print(
            f"distortion level {i+1}->{i} | raw={raw['max_abs_diff']:.6f} | "
            f"lstsq={fit['max_abs_diff']:.6f} | alpha={fit['alpha']:.6f}"
        )


if __name__ == "__main__":
    main()
