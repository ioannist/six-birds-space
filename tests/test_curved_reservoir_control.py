from fractions import Fraction as F
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel, macro_kernel
from geo_sbt.geometry.spherical_transport import sphere_distances, sphere_embedding_from_landmarks

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('curved_reservoir', ROOT/'scripts/audit_curved_reservoir_coherence.py')
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


@pytest.mark.parametrize('volume,tau', [(4, 5), (5, 5), (8, 5), (4, 10), (8, 10)])
def test_polynomial_equals_actual_public_staged_macro_kernel(volume, tau):
    points, _ = control.cubical_points(0)
    a, _, _ = control.contact_generator(points, 1.)
    p = control.sparse_micro_kernel(a, volume).toarray()
    c = np.eye(len(points))[np.arange(len(p))//volume]
    u = c.T/volume
    original = macro_kernel(p, tau, c, u)
    polynomial = control.reservoir_macro_kernel(a, volume, tau)
    assert np.allclose(original, polynomial, atol=3e-14, rtol=0)
    assert np.min(np.diag(p)) >= .5
    assert np.allclose(p, p.T, atol=0, rtol=0)
    assert np.allclose(p.sum(axis=1), 1, atol=2e-14, rtol=0)


def test_exact_stage_five_coefficients_and_moment_constant():
    expected = [F(5), F(7345, 1024), F(109, 16), F(31, 8), F(1)]
    for volume in [4, 5, 8, 16]:
        assert control.exact_macro_coefficients(volume, 5) == expected
    assert control.moment_constant(5) == F(32805, 32768)


def test_nested_cubical_lenses_preserve_orthogonal_landmarks_and_geometry():
    fine, landmarks = control.cubical_points(3)
    assert np.allclose(fine[list(landmarks)], np.eye(3), atol=0, rtol=0)
    for level in [0, 1, 2]:
        coarse, coarse_landmarks = control.cubical_points(level)
        parents = control.parent_labels(3, level)
        assert np.array_equal(parents[list(landmarks)], coarse_landmarks)
        assert np.all(np.bincount(parents) == 4**(3-level))
        displacement = np.arctan2(np.linalg.norm(np.cross(fine, coarse[parents]), axis=1),
                                  np.sum(fine*coarse[parents], axis=1))
        assert displacement.max() <= min(np.pi, 2*np.pi*np.sqrt(2)*2.**(-level))
    with pytest.raises(ValueError):
        control.parent_labels(1, 2)
    with pytest.raises(ValueError):
        control.cubical_points(1.5)


def test_all_gates_hold_together_for_actual_micro_kernel():
    result = control.finite_ladder(0, 8, .25)
    assert result['staging'] == 5
    for layer in result['levels']:
        bound = float(F(layer['closure_and_prototype_bound']))
        assert max(layer['closure_all_microstate_rows'], layer['worst_prototype_defect']) <= bound+2e-13
        assert layer['connected_all_pairs']
        assert layer['uniform_normalized_metric_error'] <= layer['metric_error_bound']
        assert layer['normalized_metric_diameter'] >= np.pi-2e-13
    assert result['equal_time_prototype_route_mismatch'] <= float(F(result['prototype_route_bound']))
    assert result['equal_time_all_microstate_route_mismatch'] <= result['all_microstate_route_bound']


def test_true_markov_metric_supplies_landmark_reconstruction_error_premise():
    points, landmarks = control.cubical_points(0)
    beta, volume = 128., 8
    a, geometry, _ = control.contact_generator(points, beta)
    kernel = control.reservoir_macro_kernel(a, volume)
    eta = 2.**(-np.ceil(beta*np.pi))/(256*len(points)*volume)
    metric = all_pairs_shortest_path(cost_matrix_from_kernel(kernel, eta=eta, symmetrize='weight_avg'))/(beta*np.log(2))
    _, epsilon = control.metric_error_bound(beta, len(points), volume, 1, 0)
    assert np.max(np.abs(metric-geometry)) <= epsilon
    # Clamping is a reconstruction step only. It cannot increase the uniform
    # error from a true spherical metric; the actual path metric stays intact.
    candidate = np.minimum(metric, np.pi)
    recovered, residuals = sphere_embedding_from_landmarks(
        candidate, landmarks, recognition_tolerance=(1+2*np.pi*np.sqrt(3))*epsilon)
    assert np.max(np.linalg.norm(recovered-points, axis=1)) <= 2*np.sqrt(3)*epsilon
    assert residuals['metric_reconstruction_error'] <= (1+2*np.pi*np.sqrt(3))*epsilon


def test_fixed_clipping_floor_destroys_the_geometric_lower_bound():
    points, _ = control.cubical_points(0)
    a, geometry, _ = control.contact_generator(points, 4.)
    kernel = control.reservoir_macro_kernel(a, 8)
    wrong = all_pairs_shortest_path(cost_matrix_from_kernel(kernel, eta=.9, symmetrize='weight_avg'))/(4*np.log(2))
    assert np.min(wrong-geometry) < -3


def test_removing_geometric_contact_weights_collapses_the_normalized_geometry():
    m, volume = 6, 8
    a = np.full((m, m), 1/(16*m))
    np.fill_diagonal(a, -(m-1)/(16*m))
    kernel = control.reservoir_macro_kernel(a, volume)
    metric = all_pairs_shortest_path(cost_matrix_from_kernel(kernel, eta=1e-8, symmetrize='weight_avg'))
    assert metric.max()/(128*np.log(2)) < .1
    # No corresponding unit-sphere diameter survives when beta grows.
    assert metric.max()/np.log(2) > 1


def test_baseline_and_finite_representation_domains_are_enforced():
    with pytest.raises(ValueError, match='baseline'):
        control.metric_error_bound(5, 6, 4, 4, 1)
    with pytest.raises(ValueError, match='integer'):
        control.exact_macro_coefficients(4.5, 5)
    points, _ = control.cubical_points(0)
    with pytest.raises(ValueError, match='represent'):
        control.contact_generator(points, 1000.)
    with pytest.raises(ValueError, match='gateway'):
        control.sparse_micro_kernel(np.eye(3), 4)


def test_universal_rate_is_stronger_than_shrinking_loop_area():
    early, late = control.universal_parameter_bounds(64), control.universal_parameter_bounds(128)
    assert late['metric_error_divided_by_loop_scale_squared'] < 1e-6
    assert late['metric_error_divided_by_loop_scale_squared'] < early['metric_error_divided_by_loop_scale_squared']/10000
    assert late['all_microstate_route_bound'] < 1e-30
    assert late['baseline_margin'] > 0


def test_actual_staged_markov_metric_to_transport_control():
    result = control.finite_transport_control(256.)
    assert result['actual_Markov_metric_error'] <= result['derived_uniform_metric_error_bound']
    assert result['actual_angle_error'] <= result['proved_angle_error_bound']
    assert abs(result['area_normalized_angle_from_Markov_metric']-1) < .04


def test_irregular_contact_rows_against_dense_micro_dynamics():
    points = np.random.default_rng(30).normal(size=(7, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    volume, beta = 5, 2.
    a, reference, _ = control.contact_generator(points, beta)
    assert np.ptp(np.diag(a)) > .001
    p = control.sparse_micro_kernel(a, volume).toarray()
    c = np.eye(len(points))[np.arange(len(p))//volume]
    k = macro_kernel(p, 5, c, c.T/volume)
    assert np.allclose(k, control.reservoir_macro_kernel(a, volume), atol=2e-14, rtol=0)
    eta = 2.**(-np.ceil(beta*np.pi))/(256*len(points)*volume)
    distance = all_pairs_shortest_path(cost_matrix_from_kernel(k, eta=eta, symmetrize='weight_avg'))/(beta*np.log(2))
    _, epsilon = control.metric_error_bound(beta, len(points), volume, 1, 0)
    assert np.min(distance-reference) >= -2e-14
    assert np.max(distance-reference) <= epsilon


def test_evidence_verifier_rejects_numerical_and_exact_tampering():
    control.compare_evidence({'coefficient': '5', 'error': .2}, {'coefficient': '5', 'error': .2})
    with pytest.raises(ValueError, match='exact'):
        control.compare_evidence({'coefficient': '6'}, {'coefficient': '5'})
    with pytest.raises(ValueError, match='numerical'):
        control.compare_evidence({'error': .3}, {'error': .2})
