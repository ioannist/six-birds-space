"""Metric reconstruction and transport under an explicit unit-sphere model.

This is a separate control, not the local planar MDS/Procrustes estimator.
Three marked orthogonal landmarks fix an ambient frame. Recognition residuals
must be checked before applying the model to a Markov-induced metric.
"""
from __future__ import annotations

import numpy as np


def sphere_distances(points: np.ndarray) -> np.ndarray:
    """Stable great-circle distances between unit vectors in R^3."""
    p = np.asarray(points, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or not np.allclose(np.linalg.norm(p, axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("points must be finite unit vectors in R^3")
    return np.arctan2(np.linalg.norm(np.cross(p[:, None], p[None, :]), axis=2), p @ p.T)


def sphere_embedding_from_landmarks(
    distances: np.ndarray, landmarks: tuple[int, int, int], *, recognition_tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Recover unit-sphere coordinates and check the full metric residual.

    Exact applicability requires orthogonal landmark points and a unit-sphere
    geodesic metric. A positive tolerance only accepts a finite residual; it
    does not prove the model or a continuum interpretation.
    """
    d = np.asarray(distances, dtype=float)
    if (d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 3
            or not np.isfinite(d).all() or np.any(d < 0) or np.any(d > np.pi)
            or not np.allclose(d, d.T, atol=1e-12, rtol=0)
            or not np.allclose(np.diag(d), 0, atol=1e-12, rtol=0)):
        raise ValueError("distances must be a finite symmetric unit-sphere candidate")
    if (len(landmarks) != 3 or any(type(i) is not int or not 0 <= i < len(d) for i in landmarks)
            or len(set(landmarks)) != 3):
        raise ValueError("three distinct integer landmark indices are required")
    if not np.isfinite(recognition_tolerance) or recognition_tolerance < 0:
        raise ValueError("recognition tolerance must be finite and nonnegative")
    landmark_distances = d[np.ix_(landmarks, landmarks)]
    expected = (np.ones((3, 3))-np.eye(3))*np.pi/2
    landmark_error = float(np.max(np.abs(landmark_distances-expected)))
    raw = np.cos(d[:, landmarks])
    norms = np.linalg.norm(raw, axis=1)
    if np.min(norms) < .5:
        raise ValueError("landmark reconstruction is degenerate")
    points = raw/norms[:, None]
    radial_error = float(np.max(np.abs(norms-1)))
    metric_error = float(np.max(np.abs(sphere_distances(points)-d)))
    diagnostics = {"landmark_orthogonality_error": landmark_error,
                   "radial_error": radial_error, "metric_reconstruction_error": metric_error}
    if max(diagnostics.values()) > recognition_tolerance:
        raise ValueError("metric does not satisfy the unit-sphere recognition tolerance")
    return points, diagnostics


def _unit_point(point: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    if (p.shape != (3,) or not np.isfinite(p).all()
            or not np.isclose(np.linalg.norm(p), 1, atol=1e-10, rtol=0)):
        raise ValueError("point must be a finite unit vector")
    return p


def tangent_frame(point: np.ndarray, reference: np.ndarray = np.array([1., 0., 0.]),
                  *, min_frame_norm: float = 1e-8) -> np.ndarray:
    """An oriented orthonormal tangent frame on a nonsingular reference patch."""
    p, reference = _unit_point(point), _unit_point(reference)
    if not np.isfinite(min_frame_norm) or not 0 < min_frame_norm <= 1:
        raise ValueError("frame threshold must be in (0,1]")
    projected = reference-np.dot(reference, p)*p
    norm = np.linalg.norm(projected)
    if norm < min_frame_norm:
        raise ValueError("reference tangent frame is singular")
    first = projected/norm
    return np.column_stack([first, np.cross(p, first)])


def great_circle_rotation(source: np.ndarray, target: np.ndarray,
                          *, min_denominator: float = 1e-8) -> np.ndarray:
    """Ambient rotation implementing parallel transport along the shorter arc."""
    p, q = _unit_point(source), _unit_point(target)
    if not np.isfinite(min_denominator) or not 0 < min_denominator <= 2:
        raise ValueError("antipodal threshold must be in (0,2]")
    denominator = 1+np.dot(p, q)
    if denominator < min_denominator:
        raise ValueError("shorter great-circle arc is not uniquely conditioned")
    x, y, z = np.cross(p, q)
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+skew+skew@skew/denominator


def spherical_transport_rotation(source: np.ndarray, target: np.ndarray,
                                 source_frame: np.ndarray, target_frame: np.ndarray,
                                 *, min_denominator: float = 1e-8) -> np.ndarray:
    """Row-coordinate transport, accepting independent O(2) tangent gauges."""
    p, q = _unit_point(source), _unit_point(target)
    frames = [np.asarray(source_frame, dtype=float), np.asarray(target_frame, dtype=float)]
    for point, frame in zip([p, q], frames):
        if (frame.shape != (3, 2) or not np.isfinite(frame).all()
                or not np.allclose(frame.T@frame, np.eye(2), atol=1e-10, rtol=0)
                or not np.allclose(point@frame, 0, atol=1e-10, rtol=0)):
            raise ValueError("frames must be orthonormal and tangent at their points")
    rotation = great_circle_rotation(p, q, min_denominator=min_denominator)
    return frames[0].T@rotation.T@frames[1]


def landmark_transport_error_bound(metric_error: float, *, frame_margin: float,
                                   arc_margin: float) -> dict:
    """Written-proof bounds under uniform metric error from a true sphere.

    The true landmarks must be orthogonal. Both true and reconstructed frames
    and arcs must satisfy the supplied margins. Recognition residuals alone
    do not establish these hypotheses or the uniform error from a true model.
    """
    if (not all(np.isfinite(x) for x in [metric_error, frame_margin, arc_margin])
            or not 0 <= metric_error <= 1/(2*np.sqrt(3))
            or not 0 < frame_margin <= 1 or not 0 < arc_margin <= 2):
        raise ValueError("error and conditioning margins are outside the proved domain")
    point_error = 2*np.sqrt(3)*metric_error
    ambient_constant = 2+4/arc_margin+2/arc_margin**2
    edge_constant = 2+16/frame_margin+ambient_constant
    edge_error = point_error*edge_constant
    return {"point_error": point_error, "edge_operator_error": edge_error,
            "triangle_operator_error": 3*edge_error,
            "triangle_principal_angle_error": 3*np.pi/2*edge_error}
