import numpy as np

from geo_sbt.lenses.ladder import (
    check_refinement_consistency,
    hierarchical_diffusion_partition,
    max_macro_identity_deviation,
)
from geo_sbt.lenses.prototypes import (
    prototypes_stationary_conditional,
    prototypes_uniform,
)
from geo_sbt.lenses.stationary import stationary_distribution
from geo_sbt.packaging import Q_f, U_f, make_C
from geo_sbt.substrates.grid import grid_2d


def test_grid_ladder_scales():
    P = grid_2d(n_side=10, lazy=0.5)
    ladder = hierarchical_diffusion_partition(
        P, levels=[2, 4, 8, 16], n_eigs=4, method="kmeans_merge", seed=0
    )
    assert len(ladder["labels_list"]) == 4
    assert ladder["cluster_counts"] == [2, 4, 8, 16]


def test_refinement_consistency():
    P = grid_2d(n_side=10, lazy=0.5)
    ladder = hierarchical_diffusion_partition(
        P, levels=[2, 4, 8, 16], n_eigs=4, method="kmeans_merge", seed=0
    )
    labels_list = ladder["labels_list"]
    refine_maps = ladder["refine_maps"]
    for i in range(len(refine_maps)):
        assert check_refinement_consistency(labels_list[i], labels_list[i + 1], refine_maps[i])


def test_uniform_prototypes_valid():
    P = grid_2d(n_side=10, lazy=0.5)
    ladder = hierarchical_diffusion_partition(
        P, levels=[2, 4, 8, 16], n_eigs=4, method="kmeans_merge", seed=0
    )
    for labels in ladder["labels_list"]:
        U = prototypes_uniform(labels)
        assert np.allclose(U.sum(axis=1), 1.0, atol=1e-12)
        assert np.min(U) >= -1e-15
        for x in range(U.shape[0]):
            mask = labels != x
            assert np.all(U[x, mask] == 0.0)


def test_stationary_prototypes_valid():
    P = grid_2d(n_side=10, lazy=0.5)
    ladder = hierarchical_diffusion_partition(
        P, levels=[2, 4, 8, 16], n_eigs=4, method="kmeans_merge", seed=0
    )
    pi = stationary_distribution(P)
    for labels in ladder["labels_list"]:
        U = prototypes_stationary_conditional(labels, pi)
        assert np.allclose(U.sum(axis=1), 1.0, atol=1e-12)
        assert np.min(U) >= -1e-15
        for x in range(U.shape[0]):
            mask = labels != x
            assert np.all(U[x, mask] == 0.0)


def test_macro_identity_deviation():
    P = grid_2d(n_side=10, lazy=0.5)
    ladder = hierarchical_diffusion_partition(
        P, levels=[2, 4, 8, 16], n_eigs=4, method="kmeans_merge", seed=0
    )
    for labels in ladder["labels_list"]:
        C = make_C(labels)
        U = prototypes_uniform(labels)
        max_dev = max_macro_identity_deviation(C, U)
        assert max_dev <= 1e-12

        for x in range(U.shape[0]):
            nu = np.zeros(U.shape[0])
            nu[x] = 1.0
            mu = U_f(nu, U)
            nu_back = Q_f(mu, C)
            assert np.allclose(nu_back, nu, atol=1e-12)


def test_repeated_coordinates_still_make_nonempty_cells():
    from geo_sbt.lenses.kmeans import kmeans
    labels = kmeans(np.zeros((8, 2)), 4)
    assert np.array_equal(np.unique(labels), np.arange(4))
    assert np.array_equal(labels, kmeans(np.zeros((8, 2)), 4))


def test_periodic_chain_stationarity_and_nonconvergence():
    import pytest
    P = np.array([[0., .5, .5], [1., 0., 0.], [1., 0., 0.]])
    pi = stationary_distribution(P)
    assert np.allclose(pi, [.5, .25, .25], atol=1e-12)
    assert np.sum(np.abs(pi @ P - pi)) <= 1e-12
    with pytest.raises(RuntimeError):
        stationary_distribution(np.array([[.9, .1], [.2, .8]]), max_iter=1)


def test_refinement_audit_does_not_silently_truncate_invalid_labels():
    assert not check_refinement_consistency([.5], [.5], [.5])
    assert not check_refinement_consistency([0], [-1], [0])
    assert not check_refinement_consistency([0], [1], [0])
    assert not check_refinement_consistency([0], [0], [np.nan])
    assert not check_refinement_consistency([[0]], [[0]], [0])
    assert check_refinement_consistency([4, 7], [1, 0], [7, 4])


def test_lens_parameters_and_diffusion_kernel_are_validated():
    import pytest
    from geo_sbt.lenses.diffusion import diffusion_coordinates
    P = grid_2d(3, .5)
    for levels in [[2.5, 4], [True, 4], ['2', '4']]:
        with pytest.raises(ValueError, match='integers'):
            hierarchical_diffusion_partition(P, levels, 2)
    for n_eigs in [1.5, np.nan, -1, True]:
        with pytest.raises(ValueError, match='nonnegative integer'):
            diffusion_coordinates(P, n_eigs)
    with pytest.raises(ValueError, match='finite'):
        diffusion_coordinates(np.array([[np.nan]]), 1)
