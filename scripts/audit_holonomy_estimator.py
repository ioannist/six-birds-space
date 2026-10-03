"""Exact leading MDS bias and exact-metric flat/spherical refinement controls.

The rational coefficient is exact arithmetic. Finite spectral calculations,
area quadrature, and spherical-model reconstruction use floating arithmetic.
The Taylor and transport arguments are in the companion review note.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from geo_sbt.geometry.holonomy import (metric_knn, local_embeddings_from_metric,
    transport_rotation, rotation_angle)
from geo_sbt.geometry.spherical_transport import (sphere_distances,
    sphere_embedding_from_landmarks, tangent_frame, spherical_transport_rotation,
    landmark_transport_error_bound)

CLOUD = [[0, 0], [2, 0], [0, 2], [1, -4], [5, 1], [-3, -3],
         [6, -4], [4, 5], [2, -3], [1, 3], [-4, 0], [1, 1]]
EDGES = [(0, 1), (1, 2), (2, 0)]


def center(a):
    return a-a.sum(axis=0)/len(a)


def inverse2(a):
    x, y, z, w = a.ravel()
    determinant = x*w-y*z
    if determinant <= 0:
        raise ValueError('rational chart covariance must be positive definite')
    return np.array([[w, -y], [-z, x]], dtype=object)/determinant


def exact_leading_certificate():
    q = np.array([[F(x) for x in point] for point in CLOUD], dtype=object)
    squared = np.array([[sum((a-b)**2) for b in q] for a in q], dtype=object)
    neighborhoods, gaps, derivatives, determinants = [], [], [], []
    for i in range(3):
        ordered = sorted((squared[i, j], j) for j in range(len(q)) if j != i)
        gap = ordered[8][0]-ordered[7][0]
        if gap <= 0:
            raise ValueError('kNN boundary has a tie')
        n = [i]+[j for _, j in ordered[:8]]
        neighborhoods.append(n); gaps.append(gap)
        z, count = q[n], len(n)
        x = center(z)
        gram = x.T@x
        gi = inverse2(gram)
        determinants.append(gram[0, 0]*gram[1, 1]-gram[0, 1]*gram[1, 0])
        j = np.eye(count, dtype=object)-F(1, count)
        w = np.array([[-(a[0]*b[1]-a[1]*b[0])**2/F(3) for b in z] for a in z], dtype=object)
        e = -F(1, 2)*(j@w@j)
        y = e@x@gi-F(1, 2)*x@gi@(x.T@e@x)@gi
        # Check the claimed spectral derivative, independently of eigenvectors.
        projector = x@gi@x.T
        expected = projector@e+e@projector-projector@e@projector
        if np.any(x@y.T+y@x.T != expected):
            raise AssertionError('rank-two Gram derivative identity failed')
        derivatives.append(y)
    coefficients, overlaps, overlap_determinants = [], [], []
    for i, j in EDGES:
        overlap = sorted(set(neighborhoods[i]) & set(neighborhoods[j]))
        a = center(q[overlap])
        yi = center(derivatives[i][[neighborhoods[i].index(v) for v in overlap]])
        yj = center(derivatives[j][[neighborhoods[j].index(v) for v in overlap]])
        gram = a.T@a
        inverse2(gram)
        determinant = gram[0, 0]*gram[1, 1]-gram[0, 1]*gram[1, 0]
        m = a.T@yj+yi.T@a
        coefficients.append((m[1, 0]-m[0, 1])/np.trace(gram))
        overlaps.append(overlap); overlap_determinants.append(str(determinant))
    total = sum(coefficients)
    return {'cloud_integer_normal_coordinates': CLOUD, 'knn': 8,
            'neighborhoods': neighborhoods, 'squared_distance_boundary_gaps': [str(v) for v in gaps],
            'chart_covariance_determinants': [str(v) for v in determinants], 'overlaps': overlaps,
            'overlap_covariance_determinants': overlap_determinants,
            'oriented_edge_angle_coefficients': [str(v) for v in coefficients],
            'oriented_loop_angle_coefficient': str(total), 'planar_triangle_area': '2',
            'absolute_area_normalized_limit': str(abs(total)/2),
            'limit_exceeds_one_by_at_least': '1/5'}


def exponential_cloud(h):
    q = np.array(CLOUD, dtype=float)
    radius = np.linalg.norm(q, axis=1)
    return np.column_stack([q*(h*np.sinc(h*radius/np.pi))[:, None], np.cos(h*radius)])


def triangle_area_quadrature(h, order=48):
    """Gnomonic Jacobian on the right triangle with side tan(2h)."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    side = np.tan(2*h)
    x = side*nodes[:, None]
    y = (side-x)*nodes[None, :]
    area = float(np.sum(weights[:, None]*weights[None, :]
                        *side*(side-x)/(1+x*x+y*y)**1.5))
    return area


def loop_audit(d, k, h):
    neighbors = metric_knn(d, k)
    neighborhoods, coords = local_embeddings_from_metric(d, neighbors)
    matrices, ranks = [], []
    for i, j in EDGES:
        rotation = transport_rotation(i, j, neighborhoods, coords)
        if rotation is None:
            raise AssertionError('control unexpectedly has missing or rank-deficient transport')
        matrices.append(rotation)
        overlap = sorted(set(neighborhoods[i]) & set(neighborhoods[j]))
        a = center(coords[i][[list(neighborhoods[i]).index(v) for v in overlap]])
        b = center(coords[j][[list(neighborhoods[j]).index(v) for v in overlap]])
        cross = a.T@b
        ranks.append({'overlap_count': len(overlap),
                      'cross_singular_values_divided_by_h_squared': (np.linalg.svd(cross, compute_uv=False)/h**2).tolist()})
        reverse = transport_rotation(j, i, neighborhoods, coords)
        if not np.allclose(rotation, reverse.T, atol=2e-13, rtol=0):
            raise AssertionError('edge reciprocity failed')
    holonomy = matrices[0]@matrices[1]@matrices[2]
    angle = rotation_angle(holonomy)
    gauges = []
    for i, coordinate in enumerate(coords):
        theta = .31*(i+1)
        c, s = np.cos(theta), np.sin(theta)
        gauge = np.array([[c, -s], [s, c]])@np.diag([1, (-1)**i])
        gauges.append(coordinate@gauge+np.array([i*.013, -i*.019]))
    g = [transport_rotation(i, j, neighborhoods, gauges) for i, j in EDGES]
    gauged_angle = rotation_angle(g[0]@g[1]@g[2])
    if abs(gauged_angle-angle) > 3e-12:
        raise AssertionError('independent O(2) gauge and translation invariance failed')
    return {'principal_angle': angle, 'angle_divided_by_planar_h_squared_area': angle/(2*h*h),
            'angle_after_independent_gauges': gauged_angle, 'rank_diagnostics': ranks,
            'loop_determinant': float(np.linalg.det(holonomy)),
            'center_neighborhoods': [n.tolist() for n in neighborhoods[:3]]}


def spherical_control(points):
    # Add the first two orthogonal landmarks; the north pole is already point 0.
    extended = np.concatenate([points, [[1., 0., 0.], [0., 1., 0.]]])
    distances = sphere_distances(extended)
    recovered, diagnostics = sphere_embedding_from_landmarks(distances, (len(points), len(points)+1, 0))
    frames = [tangent_frame(p) for p in recovered[:3]]
    rotations = [spherical_transport_rotation(recovered[i], recovered[j], frames[i], frames[j]) for i, j in EDGES]
    h = rotations[0]@rotations[1]@rotations[2]
    angle = rotation_angle(h)
    gauges = [np.eye(2), np.diag([1., -1.]), np.array([[0., -1.], [1., 0.]])]
    gf = [f@g for f, g in zip(frames, gauges)]
    gr = [spherical_transport_rotation(recovered[i], recovered[j], gf[i], gf[j]) for i, j in EDGES]
    gh = gr[0]@gr[1]@gr[2]
    if not np.allclose(gh, gauges[0].T@h@gauges[0], atol=2e-13, rtol=0):
        raise AssertionError('spherical transport gauge law failed')
    return {'metric_recognition': diagnostics, 'principal_angle': angle,
            'angle_after_independent_O2_gauges': rotation_angle(gh)}


def noisy_spherical_control(points, h, area):
    extended = np.concatenate([points, [[1., 0., 0.], [0., 1., 0.]]])
    distances = sphere_distances(extended)
    epsilon = h**4
    perturbed = distances*(1-epsilon/np.pi)
    recovered, residuals = sphere_embedding_from_landmarks(
        perturbed, (len(points), len(points)+1, 0), recognition_tolerance=10*epsilon+1e-12)
    bound = landmark_transport_error_bound(epsilon, frame_margin=.8, arc_margin=1.8)
    rotations = []
    for i, j in EDGES:
        for p in [extended, recovered]:
            if min(np.linalg.norm(np.array([1., 0., 0.])-p[v, 0]*p[v]) for v in [i, j]) < .8:
                raise AssertionError('frame margin failed')
            if 1+np.dot(p[i], p[j]) < 1.8:
                raise AssertionError('arc margin failed')
        rotations.append(spherical_transport_rotation(
            recovered[i], recovered[j], tangent_frame(recovered[i]), tangent_frame(recovered[j])))
    angle = rotation_angle(rotations[0]@rotations[1]@rotations[2])
    if abs(angle-area) > bound['triangle_principal_angle_error']+3e-13:
        raise AssertionError('conditioned spherical error bound failed')
    return {'supplied_true_metric_error_bound': epsilon, 'recognition_residuals': residuals,
            'proved_error_bounds_under_model_hypotheses': bound,
            'area_normalized_error_bound': bound['triangle_principal_angle_error']/area,
            'observed_area_normalized_angle': angle/area}


def verify(record):
    if record['exact_leading_certificate'] != exact_leading_certificate():
        raise ValueError('rational MDS certificate disagrees with reconstruction')
    ratio = F(record['exact_leading_certificate']['absolute_area_normalized_limit'])
    if ratio <= F(6, 5):
        raise ValueError('claimed strict normalized bias is absent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/holonomy_estimator_review')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(json.loads(args.verify.read_text()))
        print('Exact leading coefficient, positive ranks and strict bias verified. Taylor applicability is a written proof.')
        return
    certificate = exact_leading_certificate()
    cases = []
    for h in [.02, .01, .005, .0025, .00125]:
        points = exponential_cloud(h)
        distances = sphere_distances(points)
        measured = loop_audit(distances, 8, h)
        if [sorted(n) for n in measured['center_neighborhoods']] != [sorted(n) for n in certificate['neighborhoods']]:
            raise AssertionError('finite test left the certified kNN regime')
        flat_points = h*np.array(CLOUD, dtype=float)
        flat = loop_audit(np.linalg.norm(flat_points[:, None]-flat_points[None, :], axis=2), 8, h)
        common = loop_audit(distances, len(points)-1, h)
        if flat['principal_angle'] > 3e-12 or common['principal_angle'] > 3e-12:
            raise AssertionError('exact flat or common-chart connection has spurious holonomy')
        area = triangle_area_quadrature(h)
        if abs(area-triangle_area_quadrature(h, 24)) > 1e-14:
            raise AssertionError('area quadrature failed numerical crosscheck')
        side = np.tan(2*h)
        lower, upper = side**2/(2*(1+side**2)**1.5), side**2/2
        if not lower <= area <= upper:
            raise AssertionError('area quadrature escaped analytic bounds')
        control = spherical_control(points)
        if abs(control['principal_angle']-area) > 3e-13:
            raise AssertionError('metric-reconstructed spherical transport disagrees with area')
        cases.append({'h': h, 'spherical_triangle_area_quadrature': area,
                      'analytic_area_interval': [lower, upper],
                      'mds_angle_divided_by_area': measured['principal_angle']/area,
                      'spherical_transport_angle_divided_by_area': control['principal_angle']/area,
                      'original_MDS': measured, 'exact_flat': flat,
                      'identical_neighborhood_spherical_MDS': common, 'spherical_model_control': control,
                      'known_metric_error_spherical_control': noisy_spherical_control(points, h, area)})
    paths = [Path(__file__).resolve(), ROOT/'src/geo_sbt/geometry/holonomy.py',
             ROOT/'src/geo_sbt/geometry/spherical_transport.py', ROOT/'lean/GeoSBT/Connection.lean']
    record = {'scope': 'Exact rational leading MDS bias; floating exact-metric controls. Spherical repair assumes three orthogonal landmarks and a unit-sphere model; no learned Markov applicability claim.',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'exact_leading_certificate': certificate, 'refinement_cases': cases}
    verify(record)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evidence.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print('Exact normalized MDS limit:', certificate['absolute_area_normalized_limit'])
    print('Finite area-normalized MDS:', [c['mds_angle_divided_by_area'] for c in cases])
    print('Metric-reconstructed spherical transport:', [c['spherical_transport_angle_divided_by_area'] for c in cases])


if __name__ == '__main__':
    main()
