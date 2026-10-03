from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geo_sbt.geometry.holonomy import rotation_angle
from geo_sbt.geometry.spherical_transport import (
    great_circle_rotation, sphere_distances, sphere_embedding_from_landmarks,
    spherical_transport_rotation, tangent_frame,
    landmark_transport_error_bound,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('holonomy_estimator_audit', ROOT/'scripts/audit_holonomy_estimator.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_exact_leading_certificate_has_nonvanishing_bias_and_positive_ranks():
    certificate = audit.exact_leading_certificate()
    ratio = Fraction(certificate['absolute_area_normalized_limit'])
    assert ratio > Fraction(6, 5)
    assert ratio < Fraction(121, 100)
    assert all(Fraction(v) > 0 for v in certificate['overlap_covariance_determinants'])
    assert all(Fraction(v) > 0 for v in certificate['squared_distance_boundary_gaps'])
    record = {'exact_leading_certificate': certificate}
    audit.verify(record)
    certificate['oriented_edge_angle_coefficients'][0] = '0'
    with pytest.raises(ValueError, match='reconstruction'):
        audit.verify(record)


def test_spherical_mds_refinement_approaches_wrong_exact_coefficient():
    exact = float(Fraction(audit.exact_leading_certificate()['absolute_area_normalized_limit']))
    errors = []
    for h in [.01, .005, .0025]:
        measured = audit.loop_audit(sphere_distances(audit.exponential_cloud(h)), 8, h)
        ratio = measured['angle_divided_by_planar_h_squared_area']
        assert ratio > 1.2
        errors.append(abs(ratio-exact))
        assert min(d['cross_singular_values_divided_by_h_squared'][-1]
                   for d in measured['rank_diagnostics']) > 1
    assert errors[-1] < errors[0]/12


def test_identical_spherical_neighborhoods_are_exactly_flat_connection():
    h = .02
    result = audit.loop_audit(sphere_distances(audit.exponential_cloud(h)), 11, h)
    assert result['principal_angle'] < 2e-13
    assert audit.triangle_area_quadrature(h) > .0007


def test_metric_only_spherical_reconstruction_and_transport_area():
    h = .08
    control = audit.spherical_control(audit.exponential_cloud(h))
    assert abs(control['principal_angle']-audit.triangle_area_quadrature(h)) < 1e-13
    assert max(control['metric_recognition'].values()) < 1e-13


def test_landmark_reconstruction_noise_bound():
    points = np.concatenate([audit.exponential_cloud(.1), np.eye(3)])
    distances = sphere_distances(points)
    epsilon = 1e-6
    noise = np.random.default_rng(22).uniform(-epsilon, epsilon, distances.shape)
    noise = (noise+noise.T)/2
    perturbed = np.clip(distances+noise, 0, np.pi)
    np.fill_diagonal(perturbed, 0)
    recovered, residuals = sphere_embedding_from_landmarks(
        perturbed, (12, 13, 14), recognition_tolerance=1e-4)
    assert np.max(np.linalg.norm(recovered-points, axis=1)) <= 2*np.sqrt(3)*epsilon
    assert residuals['metric_reconstruction_error'] < 1e-4
    with pytest.raises(ValueError, match='recognition'):
        sphere_embedding_from_landmarks(perturbed, (12, 13, 14), recognition_tolerance=1e-10)


def test_false_sphere_model_is_rejected():
    # Equilateral Euclidean data can supply the marked right angles, but four
    # equally spaced points at pi/2 cannot all lie in R^3 on the unit sphere.
    distances = (np.ones((4, 4))-np.eye(4))*np.pi/2
    with pytest.raises(ValueError, match='degenerate'):
        sphere_embedding_from_landmarks(distances, (0, 1, 2))
    with pytest.raises(ValueError, match='integer'):
        sphere_embedding_from_landmarks(distances, (0, 1, 2.0))


def test_conditioned_transport_error_bound_on_noisy_spherical_metric():
    points = np.concatenate([audit.exponential_cloud(.1), np.eye(3)])
    distances = sphere_distances(points)
    epsilon = 2e-5
    # A metric perturbation with a known uniform error, generated independently
    # of the transport implementation. Model applicability is supplied here.
    perturbed = distances*(1-epsilon/np.pi)
    recovered, _ = sphere_embedding_from_landmarks(
        perturbed, (12, 13, 14), recognition_tolerance=.001)
    bound = landmark_transport_error_bound(epsilon, frame_margin=.8, arc_margin=1.8)
    actual, estimated = [], []
    for i, j in audit.EDGES:
        for cloud, output in [(points, actual), (recovered, estimated)]:
            assert np.linalg.norm(np.array([1., 0., 0.])-cloud[i, 0]*cloud[i]) >= .8
            assert 1+np.dot(cloud[i], cloud[j]) >= 1.8
            output.append(spherical_transport_rotation(
                cloud[i], cloud[j], tangent_frame(cloud[i]), tangent_frame(cloud[j])))
    assert max(np.linalg.norm(a-b, 2) for a, b in zip(actual, estimated)) <= bound['edge_operator_error']
    h, g = actual[0]@actual[1]@actual[2], estimated[0]@estimated[1]@estimated[2]
    assert np.linalg.norm(h-g, 2) <= bound['triangle_operator_error']
    assert abs(rotation_angle(h)-rotation_angle(g)) <= bound['triangle_principal_angle_error']
    with pytest.raises(ValueError, match='domain'):
        landmark_transport_error_bound(.4, frame_margin=.8, arc_margin=1.8)


def test_great_circle_transport_is_parallel_and_reciprocal():
    p = np.array([0., 0., 1.])
    q = np.array([np.sin(.3), 0., np.cos(.3)])
    rotation = great_circle_rotation(p, q)
    assert np.allclose(rotation@p, q, atol=1e-14)
    assert np.allclose(rotation@np.array([0., 1., 0.]), [0, 1, 0], atol=1e-14)
    f, g = tangent_frame(p), tangent_frame(q)
    forward = spherical_transport_rotation(p, q, f, g)
    reverse = spherical_transport_rotation(q, p, g, f)
    assert np.allclose(forward, reverse.T, atol=1e-14)
    assert rotation_angle(forward@reverse) < 1e-14
    with pytest.raises(ValueError, match='conditioned'):
        great_circle_rotation(p, -p)
    with pytest.raises(ValueError, match='singular'):
        tangent_frame(p, p)
    with pytest.raises(ValueError, match='tangent'):
        spherical_transport_rotation(p, q, np.eye(3)[:, :2], np.eye(3)[:, :2])
