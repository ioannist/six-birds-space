"""Check the boundary-sensitive constructive control against the actual grid."""
from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geo_sbt.packaging import make_C, idempotence_defect_from_factors
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('block_control', ROOT / 'scripts/audit_block_lens_coherence.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


@pytest.mark.parametrize('M,b', [(2, 2), (3, 2), (3, 4), (4, 3)])
def test_formula_and_sparse_moves_agree_with_original_dense_grid(M, b):
    P = grid_2d(M*b, lazy=.5)
    assert np.allclose(control.sparse_grid(M*b).toarray(), P, atol=1e-15, rtol=0)
    C = make_C(control.block_labels(M, b), M*M)
    U = C.T/(b*b)
    B = P @ C
    K = U @ B
    oracle = np.zeros_like(K)
    rows = control.verify_counts(M, b)
    for x, row in enumerate(rows):
        for y, weight in row.items():
            oracle[x, y] = float(weight)
    assert np.allclose(K, oracle, atol=1e-13, rtol=0)
    assert idempotence_defect_from_factors(B, U) <= 1/(2*b)+1e-12
    assert np.max(1-np.diag(K)) <= 1/(2*b)+1e-12


def test_open_boundary_correction_cannot_be_omitted():
    rows = control.verify_counts(3, 4)
    # Top-row horizontal crossing has degree-three endpoints; the central
    # row crossing has only degree-four endpoints. A periodic formula fails.
    assert rows[0][1] == Fraction(1, 32)+Fraction(1, 384)
    assert rows[3][4] == Fraction(1, 32)
    assert 1-rows[4][4] == Fraction(1, 8)


def test_block_domain_and_refinement_maps_are_explicit():
    for M, b in [(1, 2), (2, 1), (2.5, 2), (2, True)]:
        with pytest.raises(ValueError):
            control.block_labels(M, b)
    fine = control.block_labels(4, 2)
    coarse = control.block_labels(2, 4)
    r, c = np.divmod(fine, 4)
    assert np.array_equal(coarse, (r//2)*2+c//2)


def test_clipping_probability_floor_invalidates_the_readout_bound():
    M, b = 3, 4
    P = grid_2d(M*b, .5)
    C = make_C(control.block_labels(M, b), M*M)
    K = (C.T/(b*b)) @ P @ C
    reference = control.reference_distance(M)
    for eta, should_pass in [(1/(16*b), True), (.5, False)]:
        d = all_pairs_shortest_path(cost_matrix_from_kernel(K, eta=eta, symmetrize='weight_avg'))
        discrepancy = np.max(np.abs(d/(M*np.log(8*b))-reference))
        assert bool(discrepancy <= control.metric_error_bound(b)+1e-12) == should_pass
