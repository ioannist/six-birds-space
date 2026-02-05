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
