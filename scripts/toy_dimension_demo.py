from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geo_sbt.geometry.dimension import (  # noqa: E402
    ball_growth_dimension,
    entropy_at_scale,
    information_dimension_slope,
    typical_spacing,
)
from geo_sbt.geometry.metric import (  # noqa: E402
    all_pairs_shortest_path,
    cost_matrix_from_kernel,
    macro_kernel,
)
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition  # noqa: E402
from geo_sbt.lenses.prototypes import prototypes_uniform  # noqa: E402
from geo_sbt.lenses.stationary import stationary_distribution  # noqa: E402
from geo_sbt.packaging import make_C  # noqa: E402
from geo_sbt.substrates.grid import grid_2d  # noqa: E402
from geo_sbt.substrates.sierpinski import sierpinski  # noqa: E402


def _compute_dimensions(P: np.ndarray, *, also_sqrt: bool) -> dict:
    tau = 5
    ladder = hierarchical_diffusion_partition(
        P,
        levels=[4, 8, 16, 32, 64, 128],
        n_eigs=6,
        method="kmeans_merge",
        seed=0,
    )
    labels_list = ladder["labels_list"]

    mu = stationary_distribution(P)
    entropies = []
    epsilons_cost = []
    epsilons_sqrt = []

    d_cost_finest = None
    d_len_finest = None
    for labels in labels_list:
        C = make_C(labels)
        U = prototypes_uniform(labels)
        P_hat = macro_kernel(P, tau, C, U)
        cost = cost_matrix_from_kernel(P_hat, symmetrize="weight_avg", eps_edge=1e-15)
        d_cost = all_pairs_shortest_path(cost)
        d_cost_finest = d_cost
        entropies.append(entropy_at_scale(mu, C))
        epsilons_cost.append(typical_spacing(d_cost))
        if also_sqrt:
            d_len = np.sqrt(d_cost)
            d_len_finest = d_len
            epsilons_sqrt.append(typical_spacing(d_len))

    fit_slice = slice(1, -1) if len(entropies) > 2 else None
    info_cost = information_dimension_slope(
        np.array(entropies), np.array(epsilons_cost), fit_slice=fit_slice
    )
    ball_cost = ball_growth_dimension(d_cost_finest)

    info_sqrt = None
    ball_sqrt = None
    if also_sqrt:
        info_sqrt = information_dimension_slope(
            np.array(entropies), np.array(epsilons_sqrt), fit_slice=fit_slice
        )
        ball_sqrt = ball_growth_dimension(d_len_finest)

    return {
        "info_cost": float(info_cost["slope"]),
        "ball_cost": float(ball_cost["slope"]),
        "info_sqrt": None if info_sqrt is None else float(info_sqrt["slope"]),
        "ball_sqrt": None if ball_sqrt is None else float(ball_sqrt["slope"]),
        "levels_used": len(entropies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy dimension diagnostics demo.")
    parser.add_argument(
        "--also-sqrt",
        action="store_true",
        help="Also print diagnostics using sqrt(d_cost).",
    )
    args = parser.parse_args()

    P_grid = grid_2d(n_side=20, lazy=0.5)
    grid = _compute_dimensions(P_grid, also_sqrt=args.also_sqrt)
    print(
        f"grid | info_dim_slope={grid['info_cost']:.3f} | ball_dim={grid['ball_cost']:.3f} | "
        f"levels_used={grid['levels_used']}"
    )

    _, P_sier = sierpinski(level=5, lazy=0.5)
    sier = _compute_dimensions(P_sier, also_sqrt=args.also_sqrt)
    print(
        f"sierpinski | info_dim_slope={sier['info_cost']:.3f} | ball_dim={sier['ball_cost']:.3f} | "
        f"levels_used={sier['levels_used']}"
    )

    if args.also_sqrt:
        print(
            f"grid | info_dim_slope_sqrt={grid['info_sqrt']:.3f} | ball_dim_sqrt={grid['ball_sqrt']:.3f}"
        )
        print(
            f"sierpinski | info_dim_slope_sqrt={sier['info_sqrt']:.3f} | "
            f"ball_dim_sqrt={sier['ball_sqrt']:.3f}"
        )


if __name__ == "__main__":
    main()
