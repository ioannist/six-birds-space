import numpy as np

from geo_sbt.geometry.dimension import (
    ball_growth_dimension,
    entropy_at_scale,
    information_dimension_slope,
    shannon_entropy,
    typical_spacing,
)
from geo_sbt.geometry.metric import (
    all_pairs_shortest_path,
    cost_matrix_from_kernel,
    macro_kernel,
)
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition
from geo_sbt.lenses.prototypes import prototypes_uniform
from geo_sbt.lenses.stationary import stationary_distribution
from geo_sbt.packaging import make_C
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.sierpinski import sierpinski


def test_entropy_sanity():
    p = np.full(8, 1.0 / 8.0)
    assert abs(shannon_entropy(p) - np.log(8.0)) <= 1e-12
    delta = np.array([1.0, 0.0, 0.0])
    assert shannon_entropy(delta) == 0.0


def test_dimension_estimators_finite():
    P = grid_2d(10, lazy=0.5)
    ladder = hierarchical_diffusion_partition(P, levels=[4, 8, 16, 32], n_eigs=4, seed=0)
    labels_list = ladder["labels_list"]
    tau = 3

    mu = stationary_distribution(P)
    entropies = []
    epsilons = []

    for labels in labels_list:
        C = make_C(labels)
        U = prototypes_uniform(labels)
        P_hat = macro_kernel(P, tau, C, U)
        cost = cost_matrix_from_kernel(P_hat, symmetrize="weight_avg", eps_edge=1e-15)
        d_cost = all_pairs_shortest_path(cost)
        entropies.append(entropy_at_scale(mu, C))
        epsilons.append(typical_spacing(d_cost))

    info = information_dimension_slope(
        np.array(entropies),
        np.array(epsilons),
        fit_slice=slice(1, -1),
    )
    assert np.isfinite(info["slope"])
    assert 0.1 <= info["slope"] <= 12.0

    ball = ball_growth_dimension(d_cost)
    assert np.isfinite(ball["slope"])
    assert 0.8 <= ball["slope"] <= 3.5


def test_ball_growth_separation_grid_vs_sierpinski():
    tau = 3
    levels = [4, 8, 16]

    def _ball_dim(P: np.ndarray) -> float:
        ladder = hierarchical_diffusion_partition(P, levels=levels, n_eigs=4, seed=0)
        labels = ladder["labels_list"][-1]
        C = make_C(labels)
        U = prototypes_uniform(labels)
        P_hat = macro_kernel(P, tau, C, U)
        cost = cost_matrix_from_kernel(P_hat, symmetrize="weight_avg", eps_edge=1e-15)
        d_cost = all_pairs_shortest_path(cost)
        return float(ball_growth_dimension(d_cost)["slope"])

    P_grid = grid_2d(8, lazy=0.5)
    _, P_sier = sierpinski(level=3, lazy=0.5)
    dim_grid = _ball_dim(P_grid)
    dim_sier = _ball_dim(P_sier)

    assert np.isfinite(dim_grid)
    assert np.isfinite(dim_sier)
    assert abs(dim_grid - dim_sier) >= 0.1
