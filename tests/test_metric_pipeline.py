import numpy as np

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


def test_macro_kernel_shape_and_stochasticity():
    P = grid_2d(6, lazy=0.5)
    ladder = hierarchical_diffusion_partition(P, levels=[2, 4], n_eigs=3, seed=0)
    labels = ladder["labels_list"][0]
    C = make_C(labels)
    U = prototypes_uniform(labels)
    P_hat = macro_kernel(P, tau=3, C=C, U=U)
    assert P_hat.shape == (U.shape[0], U.shape[0])
    assert np.max(np.abs(P_hat.sum(axis=1) - 1.0)) < 1e-9
    assert np.min(P_hat) >= -1e-12


def test_shortest_path_finite():
    P = grid_2d(6, lazy=0.5)
    ladder = hierarchical_diffusion_partition(P, levels=[2, 4], n_eigs=3, seed=0)
    labels = ladder["labels_list"][0]
    C = make_C(labels)
    U = prototypes_uniform(labels)
    P_hat = macro_kernel(P, tau=3, C=C, U=U)
    cost = cost_matrix_from_kernel(P_hat, symmetrize="weight_avg", eps_edge=1e-15)
    d = all_pairs_shortest_path(cost)
    assert np.isfinite(d).all()


def test_distortion_finite():
    P = grid_2d(6, lazy=0.5)
    ladder = hierarchical_diffusion_partition(P, levels=[2, 4], n_eigs=3, seed=0)
    labels_coarse = ladder["labels_list"][0]
    labels_fine = ladder["labels_list"][1]
    r = ladder["refine_maps"][0]

    Cc = make_C(labels_coarse)
    Uc = prototypes_uniform(labels_coarse)
    Cf = make_C(labels_fine)
    Uf = prototypes_uniform(labels_fine)

    P_hat_c = macro_kernel(P, tau=3, C=Cc, U=Uc)
    P_hat_f = macro_kernel(P, tau=3, C=Cf, U=Uf)

    cost_c = cost_matrix_from_kernel(P_hat_c, symmetrize="weight_avg", eps_edge=1e-15)
    cost_f = cost_matrix_from_kernel(P_hat_f, symmetrize="weight_avg", eps_edge=1e-15)

    d_c = all_pairs_shortest_path(cost_c)
    d_f = all_pairs_shortest_path(cost_f)

    out = distortion_between_scales(d_f, d_c, r, rescale="lstsq")
    assert np.isfinite(out["max_abs_diff"])
    assert out["finite_pairs"] > 0


def test_zero_cost_edges_preserved_in_both_backends(monkeypatch):
    import geo_sbt.geometry.metric as metric
    costs = np.array([[0., 0., np.inf], [0., 0., 2.], [np.inf, 2., 0.]])
    expected = np.array([[0., 0., 2.], [0., 0., 2.], [2., 2., 0.]])
    assert np.array_equal(metric.all_pairs_shortest_path(costs), expected)
    monkeypatch.setattr(metric, 'sp_dijkstra', None)
    assert np.array_equal(metric.all_pairs_shortest_path(costs), expected)


def test_invalid_costs_rejected_before_dijkstra():
    import pytest
    for invalid in [-1., np.nan, -np.inf]:
        with pytest.raises(ValueError):
            all_pairs_shortest_path(np.array([[0., invalid], [1., 0.]]))
    for eta in [0., 2., np.nan]:
        with pytest.raises(ValueError):
            cost_matrix_from_kernel(np.eye(2), eta=eta)


def test_unmatched_reachability_has_infinite_distortion():
    fine = np.array([[0., 1.], [1., 0.]])
    coarse = np.array([[0., np.inf], [np.inf, 0.]])
    result = distortion_between_scales(fine, coarse, np.arange(2), rescale='lstsq')
    assert result['max_abs_diff'] == np.inf
    assert result['unmatched_pairs'] == 2


def test_global_distortion_audit_rejects_shared_disconnection():
    d = np.array([[0., np.inf], [np.inf, 0.]])
    result = distortion_between_scales(d, d, np.arange(2))
    assert result['max_abs_diff'] == np.inf
    assert result['finite_max_abs_diff'] == 0.
    assert result['disconnected']


def test_threshold_can_genuinely_disconnect_macro_graph():
    costs = cost_matrix_from_kernel(np.full((2, 2), .5), eta=1e-12, eps_edge=.6)
    distances = all_pairs_shortest_path(costs)
    assert distances[0, 1] == np.inf
    assert np.all(np.diag(distances) == 0.)
