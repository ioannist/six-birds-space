"""Common-space connection operators and exact spherical parallel-transport law.

This supplies a controlled connection, not a consistency theorem for the local
MDS estimator or learned Markov lenses. The spherical area law is derived in
the companion note; matrix checks below use floating arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def validate_connection(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 4 or a.shape[-2:] != (2, 2) or b.shape != a.shape or min(a.shape[:2]) < 1:
        raise ValueError('connection must have shape (rows,columns,2,2) on a nonempty grid')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('connection must be finite')
    for field in [a, b]:
        if not np.allclose(field @ np.swapaxes(field, -1, -2), np.eye(2), atol=1e-12, rtol=0):
            raise ValueError('connection must be orthogonal')
    return a, b


def shift(a, axis, section):
    section = np.asarray(section, dtype=float)
    if axis not in [0, 1] or section.shape != a.shape[:2]+(2,):
        raise ValueError('section or translation axis does not match the connection')
    return np.einsum('...ij,...j->...i', a, np.roll(section, -1, axis=axis))


def route_data(a, b):
    a, b = validate_connection(a, b)
    first = a @ np.roll(b, -1, axis=0)
    second = b @ np.roll(a, -1, axis=1)
    holonomy = first @ np.swapaxes(second, -1, -2)
    return first-second, holonomy, second


def gauge_connection(q, a, b):
    validate_connection(a, b)
    validate_connection(q, q)
    if q.shape != a.shape:
        raise ValueError('gauge does not match connection shape')
    return (q @ a @ np.swapaxes(np.roll(q, -1, axis=0), -1, -2),
            q @ b @ np.swapaxes(np.roll(q, -1, axis=1), -1, -2))


def commutator_matrix(a, b):
    """Construct both operators on the same finite vector-section space."""
    a, b = validate_connection(a, b)
    section_shape = a.shape[:2]+(2,)
    size = math.prod(section_shape)
    result = np.empty((size, size))
    for column in range(size):
        section = np.zeros(size)
        section[column] = 1
        section = section.reshape(section_shape)
        result[:, column] = (shift(a, 0, shift(b, 1, section))
                             - shift(b, 1, shift(a, 0, section))).ravel()
    return result


def sphere_rectangle(low, high, width):
    if not all(math.isfinite(v) for v in [low, high, width]) or not -math.pi/2 < low < high < math.pi/2 or not 0 < width < math.pi:
        raise ValueError('rectangle must lie in one nonsingular spherical coordinate patch')
    a = np.broadcast_to(np.eye(2), (2, 2, 2, 2)).copy()
    b = np.empty_like(a)
    for i, latitude in enumerate([low, high]):
        for j, step in enumerate([width, -width]):
            b[i, j] = rotation(-math.sin(latitude)*step)
    area = width*(math.sin(high)-math.sin(low))
    return a, b, area


def check_rectangle(low, high, width, rng):
    a, b, area = sphere_rectangle(low, high, width)
    if area >= math.pi:
        raise ValueError('principal-angle audit requires spherical area below pi')
    coefficient, holonomy, second = route_data(a, b)
    matrix = commutator_matrix(a, b)
    expected = 2*math.sin(area/2)
    measured = float(np.linalg.norm(matrix, 2))
    if not np.allclose(coefficient, (holonomy-np.eye(2)) @ second, atol=1e-14, rtol=0):
        raise AssertionError('common-space operator bridge failed')
    if not np.allclose(holonomy[0, 0], rotation(-area), atol=1e-14, rtol=0):
        raise AssertionError('spherical area holonomy failed')
    if not math.isclose(measured, expected, abs_tol=1e-14, rel_tol=0):
        raise AssertionError('operator norm and holonomy angle disagree')
    q = np.stack([rotation(float(angle)) for angle in rng.uniform(-3, 3, 4)]).reshape(2, 2, 2, 2)
    q[0, 1] = q[0, 1] @ np.diag([1., -1.])
    q[1, 0] = q[1, 0] @ np.diag([1., -1.])
    ga, gb = gauge_connection(q, a, b)
    _, gh, _ = route_data(ga, gb)
    conjugated = q @ holonomy @ np.swapaxes(q, -1, -2)
    if not np.allclose(gh, conjugated, atol=1e-14, rtol=0):
        raise AssertionError('loop gauge law failed')
    gnorm = float(np.linalg.norm(commutator_matrix(ga, gb), 2))
    if not math.isclose(gnorm, measured, abs_tol=1e-14, rel_tol=0):
        raise AssertionError('common-space commutator is not gauge invariant')
    return {'latitude_low': low, 'latitude_high': high, 'longitude_width': width,
            'oriented_surface_area': area, 'principal_holonomy_angle_magnitude': area,
            'operator_commutator_norm': measured, 'analytic_norm': expected,
            'commutator_norm_after_independent_O2_gauges': gnorm,
            'normalized_by_coordinate_area_and_center_density': measured/(width*(high-low)*math.cos((low+high)/2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/connection_holonomy_review')
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    rational = check_rectangle(math.asin(1/4), math.asin(1/2), 1/4, rng)
    if not math.isclose(rational['oriented_surface_area'], 1/16, abs_tol=1e-15):
        raise AssertionError('rational-area calibration failed')
    refinements = [check_rectangle(.3-h/2, .3+h/2, h, rng) for h in [.4, .2, .1, .05, .025]]
    flat_a = np.broadcast_to(rotation(.2), (2, 2, 2, 2)).copy()
    flat_b = np.broadcast_to(rotation(-.7), flat_a.shape).copy()
    flat_norm = float(np.linalg.norm(commutator_matrix(flat_a, flat_b), 2))
    if flat_norm > 1e-14:
        raise AssertionError('constant flat connection has false holonomy')
    paths = [Path(__file__).resolve(), ROOT/'lean/GeoSBT/Connection.lean']
    record = {'scope': 'Supplied geometric connection and common-space operators; no MDS-estimator, Markov-metric, or learned-lens consistency theorem.',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'exact_rational_area_law': {'sin_latitude_low': '1/4', 'sin_latitude_high': '1/2',
                                         'longitude_width': '1/4', 'area_and_angle_magnitude': '1/16',
                                         'commutator_norm': '2 sin(1/32)'},
              'rational_area_floating_check': rational, 'shrinking_rectangles': refinements,
              'constant_flat_connection_commutator_norm': flat_norm}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evidence.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print('Exact area law: 1/16; floating commutator norm:', rational['operator_commutator_norm'])
    print('Shrinking-area normalization:', [r['normalized_by_coordinate_area_and_center_density'] for r in refinements])


if __name__ == '__main__':
    main()
