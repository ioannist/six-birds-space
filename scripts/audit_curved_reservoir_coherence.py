"""Coherent spherical reservoir control with fixed staging five.

The companion note proves the universal family, all coherence gates, and
transport applicability. Finite checks below use floating arithmetic; exact
reservoir polynomial coefficients use rational arithmetic. This changes the
original sphere kNN kernel and supplies the cell/landmark recognition.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel
from geo_sbt.geometry.spherical_transport import (sphere_distances, sphere_embedding_from_landmarks,
    tangent_frame, spherical_transport_rotation, landmark_transport_error_bound)
from geo_sbt.geometry.holonomy import rotation_angle


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('integer parameter outside its domain')
    return value


def cubical_points(level):
    """Dyadic cube-face cells with marked axis representatives on each face."""
    integer(level)
    side = 2**level
    grid = -1+(2*np.arange(side)+1)/side
    u, v = np.meshgrid(grid, grid, indexing='ij')
    points, landmarks = [], []
    for face in range(6):
        axis, sign = face//2, 1 if face % 2 == 0 else -1
        raw = np.empty((side*side, 3))
        raw[:, axis] = sign
        raw[:, (axis+1) % 3] = u.ravel()
        raw[:, (axis+2) % 3] = v.ravel()
        marked = (side//2)*side+side//2 if level else 0
        raw[marked] = np.eye(3)[axis]*sign
        if sign == 1:
            landmarks.append(face*side*side+marked)
        points.append(raw/np.linalg.norm(raw, axis=1)[:, None])
    return np.concatenate(points), tuple(landmarks)


def parent_labels(fine_level, coarse_level):
    integer(fine_level); integer(coarse_level)
    if fine_level < coarse_level:
        raise ValueError('parent level must not be finer')
    side, parent = 2**fine_level, 2**coarse_level
    index = np.arange(6*side*side)
    face, local = np.divmod(index, side*side)
    row, col = np.divmod(local, side)
    factor = side//parent
    return face*parent*parent+(row//factor)*parent+col//factor


def contact_generator(points, beta):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    distances = sphere_distances(points)
    exponents = np.ceil(beta*distances).astype(np.int64)
    if np.max(exponents) > 1000:
        raise ValueError('finite audit cannot represent these contact weights')
    contacts = np.exp2(-exponents)/(8*len(points))
    np.fill_diagonal(contacts, 0)
    generator = contacts.copy()
    np.fill_diagonal(generator, -contacts.sum(axis=1))
    return generator, distances, exponents


def validate_generator(generator):
    a = np.asarray(generator, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 2
            or not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-14, rtol=0)
            or not np.allclose(a.sum(axis=1), 0, atol=1e-14, rtol=0)
            or np.any(a[~np.eye(len(a), dtype=bool)] < 0)
            or np.any(np.diag(a) > 0) or np.any(-np.diag(a) > 1/8+1e-14)):
        raise ValueError('contact generator must be symmetric with lawful gateway rates')
    return a


def sparse_micro_kernel(generator, volume):
    a = validate_generator(generator)
    integer(volume, 4)
    count = len(a)*volume
    index = np.arange(count)
    local = index % volume
    base = index-local
    rows = [index, index, index]
    cols = [index, base+(local+1) % volume, base+(local-1) % volume]
    values = [np.full(count, 5/8), np.full(count, 3/16), np.full(count, 3/16)]
    x, y = np.where(a != 0)
    rows.append(x*volume); cols.append(y*volume); values.append(a[x, y])
    p = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(count, count))
    if (np.min(p.data) < 0 or not np.allclose(p.sum(axis=1).A.ravel(), 1, atol=2e-14, rtol=0)
            or np.min(p.diagonal()) < .5-1e-14 or (p-p.T).nnz):
        raise AssertionError('micro stochasticity, symmetry or laziness failed')
    return p


def exact_return_moments(volume, maximum):
    integer(volume, 4); integer(maximum)
    vector = [F(0) for _ in range(volume)]
    vector[0] = F(1)
    result = []
    for _ in range(maximum+1):
        result.append(vector[0])
        vector = [F(5, 8)*vector[j]+F(3, 16)*(vector[(j-1) % volume]+vector[(j+1) % volume])
                  for j in range(volume)]
    return result


def exact_macro_coefficients(volume, tau):
    """K_tau = I + sum(c_k A^k)/V; expand all B/gateway words."""
    integer(volume, 4); integer(tau, 1)
    moments = exact_return_moments(volume, tau-1)
    result = []
    convolution = [F(1)]
    for k in range(1, tau+1):
        maximum = tau-k
        result.append(sum(F(maximum-s+1)*convolution[s] for s in range(min(len(convolution), maximum+1))))
        next_convolution = [F(0) for _ in range(tau)]
        for i, x in enumerate(convolution):
            for j, y in enumerate(moments[:tau-i]):
                next_convolution[i+j] += x*y
        convolution = next_convolution
    return result


def reservoir_macro_kernel(generator, volume, tau=5):
    a = validate_generator(generator)
    coefficients = exact_macro_coefficients(volume, tau)
    identity = np.eye(len(a))
    polynomial = float(coefficients[-1])*identity
    for coefficient in reversed(coefficients[:-1]):
        polynomial = float(coefficient)*identity+a@polynomial
    kernel = identity+(a@polynomial)/volume
    off_diagonal = ~np.eye(len(a), dtype=bool)
    if (not np.isfinite(kernel).all() or np.any(kernel[off_diagonal] <= 0)
            or not np.allclose(kernel.sum(axis=1), 1, atol=3e-14, rtol=0)):
        raise ValueError('finite polynomial audit lost positive stochastic macro weights')
    return kernel


def moment_constant(tau):
    integer(tau, 1)
    return F(tau, 8)*F(9, 8)**(tau-1)


def metric_error_bound(beta, fine_count, volume, group_count, rho):
    t = beta*math.log(2)
    c = float(moment_constant(5))
    baseline = math.log(volume/c)-2*t*rho
    if baseline < 0:
        raise ValueError('probability baseline does not absorb coarsening displacement')
    error = 2*rho+math.log(256*fine_count*volume*group_count)/t
    return baseline, error


def finite_ladder(level, volume, beta):
    integer(level); integer(volume, 4)
    points, _ = cubical_points(level+2)
    a, reference, exponents = contact_generator(points, beta)
    p = sparse_micro_kernel(a, volume)
    m = len(points)
    labels_micro_fine = np.arange(m*volume)//volume
    stage = np.eye(m)[labels_micro_fine]
    for _ in range(5):
        stage = p@stage
    k5 = reservoir_macro_kernel(a, volume)
    sparse_k5 = stage.reshape(m, volume, m).mean(axis=1)
    if not np.allclose(k5, sparse_k5, atol=3e-14, rtol=0):
        raise AssertionError('polynomial kernel differs from staged micro dynamics')
    stage10 = stage.copy()
    for _ in range(5):
        stage10 = p@stage10
    k10 = reservoir_macro_kernel(a, volume, 10)
    if not np.allclose(k10, stage10.reshape(m, volume, m).mean(axis=1), atol=4e-14, rtol=0):
        raise AssertionError('stage-ten polynomial differs from staged micro dynamics')
    records, arrays = [], []
    maximum_groups = 16
    # Half the direct-transition lower bound, uniformly valid at all levels.
    eta = 2.**(-math.ceil(beta*math.pi))/(256*m*volume*maximum_groups)
    if eta <= 0:
        raise ValueError('finite audit floor underflow')
    for lens_level in [level, level+1, level+2]:
        representatives, landmarks = cubical_points(lens_level)
        labels = parent_labels(level+2, lens_level)
        restrict = np.eye(len(representatives))[labels]
        groups = 4**(level+2-lens_level)
        lift = restrict.T/groups
        kernel = lift@k5@restrict
        b = stage@restrict
        closure = float(.5*np.abs(b@kernel-b).sum(axis=1).max())
        escape = float(np.max(1-np.diag(kernel)))
        rho = 0 if lens_level == level+2 else min(math.pi, 2*math.pi*math.sqrt(2)*2**(-lens_level))
        actual_rho = float(np.max(np.arctan2(np.linalg.norm(np.cross(points, representatives[labels]), axis=1),
                                                     np.sum(points*representatives[labels], axis=1))))
        if actual_rho > rho+2e-14:
            raise AssertionError('geometric cell displacement bound failed')
        baseline, error = metric_error_bound(beta, m, volume, groups, rho)
        distances = all_pairs_shortest_path(cost_matrix_from_kernel(kernel, eta=eta, eps_edge=0, symmetrize='weight_avg'))
        normalized = distances/(beta*math.log(2))
        geometric = sphere_distances(representatives)
        actual_error = float(np.max(np.abs(normalized-geometric)))
        lower_gap = float(np.min(normalized-geometric))
        escape_bound = F(5, 8*volume)
        if (max(closure, escape) > float(escape_bound)+3e-13 or not np.isfinite(distances).all()
                or actual_error > error+3e-12 or lower_gap < -3e-12
                or np.min(kernel[~np.eye(len(kernel), dtype=bool)]) <= eta):
            raise AssertionError('one of the simultaneous curved coherence gates failed')
        off = ~np.eye(len(kernel), dtype=bool)
        probability_upper = float(moment_constant(5))/volume*np.exp(-beta*math.log(2)*(geometric-2*rho))
        probability_lower = np.exp(-beta*math.log(2)*(geometric+2*rho))/(256*m*volume*groups)
        if (np.any(kernel[off] > probability_upper[off]+3e-14)
                or np.any(kernel[off] < probability_lower[off]-3e-14)):
            raise AssertionError('derived all-pairs probability bracket failed')
        records.append({'lens_level': lens_level, 'macro_states': len(kernel), 'fine_reservoirs_per_fiber': groups,
                        'closure_all_microstate_rows': closure, 'worst_prototype_defect': escape,
                        'closure_and_prototype_bound': str(escape_bound), 'rho_bound': rho,
                        'actual_representative_displacement': actual_rho, 'edge_baseline_margin': baseline,
                        'uniform_normalized_metric_error': actual_error, 'metric_error_bound': error,
                        'minimum_reference_lower_gap': lower_gap, 'normalized_metric_diameter': float(normalized.max()),
                        'connected_all_pairs': True, 'floor_below_all_positive_off_diagonal_weights': True,
                        'landmark_indices': list(landmarks)})
        arrays.append((restrict, lift, normalized, kernel, b))
    distortions = []
    for i in range(2):
        projection = parent_labels(level+i+1, level+i)
        coarse, fine = arrays[i][2], arrays[i+1][2]
        measured = float(np.max(np.abs(fine-coarse[projection[:, None], projection])))
        bound = records[i]['metric_error_bound']+records[i+1]['metric_error_bound']+2*records[i]['rho_bound']
        if measured > bound+3e-12:
            raise AssertionError('normalized refinement distortion failed')
        distortions.append({'uniform_distortion': measured, 'analytic_bound': bound})
    coarse_r, middle_r, middle_l = arrays[0][0], arrays[1][0], arrays[1][1]
    direct = k10@coarse_r
    via = k5@middle_r@middle_l@k5@coarse_r
    prototype_route = float(.5*np.abs(direct-via).sum(axis=1).max())
    prototype_route_bound = F(5, 2*volume)
    direct_micro = stage10@coarse_r
    via_micro = stage@middle_r@middle_l@k5@coarse_r
    micro_route = float(.5*np.abs(direct_micro-via_micro).sum(axis=1).max())
    side = 2**(level+2)
    t = beta*math.log(2)
    gateway_bound = min(1/8, 25/2*(2/t**2+2/(side*t)+1/side**2))
    micro_route_bound = 15*gateway_bound+5/(8*volume)
    if (prototype_route > float(prototype_route_bound)+3e-13
            or micro_route > micro_route_bound+3e-13 or np.max(-np.diag(a)) > gateway_bound+3e-13):
        raise AssertionError('equal-time route or gateway bound failed')
    return {'coarsest_level': level, 'reservoir_volume': volume, 'beta': beta,
            'microstates': len(p.diagonal()), 'staging': 5, 'holding_probability_at_least': .5,
            'exact_stage_five_polynomial_coefficients': [str(c) for c in exact_macro_coefficients(volume, 5)],
            'contact_quantization_maximum_exponent': int(np.max(exponents)), 'eta': eta, 'eps_edge': 0,
            'fine_reference_contains_antipodal_axis_pairs': True, 'levels': records, 'distortions': distortions,
            'equal_time_prototype_route_mismatch': prototype_route, 'prototype_route_bound': str(prototype_route_bound),
            'equal_time_all_microstate_route_mismatch': micro_route, 'all_microstate_route_bound': micro_route_bound,
            'maximum_gateway_rate': float(np.max(-np.diag(a))), 'gateway_rate_bound': gateway_bound,
            'route_domain': 'Both finest prototypes and every microstate; total time ten versus five plus five.'}


def universal_parameter_bounds(level):
    integer(level, 5)
    rho = 2*math.pi*math.sqrt(2)*2.**(-level)
    beta = 2.**(level/2)
    volume = 2**level
    m = 6*4**(level+2)
    t = beta*math.log(2)
    error = 4*math.pi*math.sqrt(2)*2.**(-level)+(16+math.log2(6)+3*level)*2.**(-level/2)
    baseline = level*math.log(2)-math.log(float(moment_constant(5)))-2*t*rho
    side = 2.**(level+2)
    gateway = min(1/8, 25/2*(2/t**2+2/(side*t)+1/side**2))
    h = 2.**(-level/8)
    if baseline < 0:
        raise AssertionError('universal parameter regime failed its baseline gate')
    return {'level': level, 'finest_macro_states': m, 'reservoir_volume': volume, 'beta': beta,
            'metric_error_bound': error, 'cell_covering_radius_bound': rho,
            'closure_and_prototype_bound': 5/(8*volume), 'prototype_route_bound': 5/(2*volume),
            'all_microstate_route_bound': 15*gateway+5/(8*volume), 'baseline_margin': baseline,
            'curvature_loop_scale': h, 'metric_error_divided_by_loop_scale_squared': error/h**2,
            'floor_formula': '2^(-ceil(beta*pi))/(256*m*V*16); no fixed numerical floor in the limit',
            'scope': 'Values of proved formulae, not an executed exponentially large microstate audit.'}


def finite_transport_control(beta):
    """End-to-end reconstruction from a real staged Markov metric."""
    h, volume = .2, 8
    points = np.array([[0., 0., 1.], [h, 0., 1.], [0., h, 1.], [1., 0., 0.], [0., 1., 0.]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    a, reference, _ = contact_generator(points, beta)
    k5 = reservoir_macro_kernel(a, volume)
    eta = 2.**(-math.ceil(beta*math.pi))/(256*len(points)*volume)
    metric = all_pairs_shortest_path(cost_matrix_from_kernel(k5, eta=eta, symmetrize='weight_avg'))/(beta*math.log(2))
    _, epsilon = metric_error_bound(beta, len(points), volume, 1, 0)
    if np.max(np.abs(metric-reference)) > epsilon+3e-13:
        raise AssertionError('actual Markov metric did not supply the error premise')
    recovered, recognition = sphere_embedding_from_landmarks(
        np.minimum(metric, math.pi), (3, 4, 0), recognition_tolerance=(1+2*math.pi*math.sqrt(3))*epsilon)
    bound = landmark_transport_error_bound(epsilon, frame_margin=.8, arc_margin=1.8)
    edges = [(0, 1), (1, 2), (2, 0)]
    loop_angles = []
    for p in [points, recovered]:
        r = []
        for i, j in edges:
            if (min(np.linalg.norm(np.array([1., 0., 0.])-p[v, 0]*p[v]) for v in [i, j]) < .8
                    or 1+np.dot(p[i], p[j]) < 1.8):
                raise AssertionError('transport conditioning margins failed')
            r.append(spherical_transport_rotation(p[i], p[j], tangent_frame(p[i]), tangent_frame(p[j])))
        loop_angles.append(rotation_angle(r[0]@r[1]@r[2]))
    actual_error = abs(loop_angles[1]-loop_angles[0])
    if actual_error > bound['triangle_principal_angle_error']+3e-13:
        raise AssertionError('Markov-derived transport estimate exceeded proved bound')
    return {'beta': beta, 'tau': 5, 'reservoir_volume': volume, 'spherical_gnomonic_triangle_side': h,
            'actual_Markov_metric_error': float(np.max(np.abs(metric-reference))),
            'derived_uniform_metric_error_bound': epsilon, 'metric_recognition_residuals': recognition,
            'true_spherical_area': loop_angles[0], 'angle_from_Markov_metric_reconstruction': loop_angles[1],
            'area_normalized_angle_from_Markov_metric': loop_angles[1]/loop_angles[0],
            'actual_angle_error': actual_error, 'proved_angle_error_bound': float(bound['triangle_principal_angle_error']),
            'scope': 'Finite Markov-to-metric-to-transport control. Universal shrinking-loop theorem is written separately.'}


def build_evidence():
    cases = [finite_ladder(0, 8, .25), finite_ladder(1, 16, .5)]
    rates = [universal_parameter_bounds(r) for r in [5, 16, 32, 64, 128, 256]]
    paths = sorted((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve(),
             ROOT/'lean/GeoSBT/MarkovClosure.lean', ROOT/'lean/GeoSBT/CoherentMetric.lean',
             ROOT/'lean/GeoSBT/TransportError.lean']
    return {'scope': 'New spherical reservoir dynamics and supplied nested cell lenses; stage five retained. Universal proofs are written; finite matrices and parameter evaluations are floating controls.',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'finite_ladders': cases, 'universal_formula_evaluations': rates,
              'exact_moment_constant_stage_five': str(moment_constant(5)),
              'actual_Markov_metric_transport_controls': [finite_transport_control(beta) for beta in [64., 128., 256.]]}


def compare_evidence(saved, fresh, location='root'):
    """Recompute the matrices; tolerate rounding but reject schema/certificate changes."""
    if type(saved) is not type(fresh):
        raise ValueError('evidence type mismatch at '+location)
    if isinstance(fresh, dict):
        if saved.keys() != fresh.keys():
            raise ValueError('evidence keys mismatch at '+location)
        for key in fresh:
            compare_evidence(saved[key], fresh[key], location+'.'+key)
    elif isinstance(fresh, list):
        if len(saved) != len(fresh):
            raise ValueError('evidence length mismatch at '+location)
        for i, (x, y) in enumerate(zip(saved, fresh)):
            compare_evidence(x, y, location+'.'+str(i))
    elif isinstance(fresh, float):
        if not math.isfinite(saved) or not math.isclose(saved, fresh, rel_tol=1e-9, abs_tol=1e-11):
            raise ValueError('numerical reconstruction mismatch at '+location)
    elif saved != fresh:
        raise ValueError('exact reconstruction mismatch at '+location)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/curved_reservoir_review')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    record = build_evidence()
    if args.verify:
        compare_evidence(json.loads(args.verify.read_text()), record)
        print('Finite staged matrices, all coherence gates, Markov transport controls and exact coefficients rechecked. Universal limit proof is written separately.')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evidence.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for case in record['finite_ladders']:
        print('microstates', case['microstates'], 'worst fine persistence', case['levels'][-1]['worst_prototype_defect'],
              'all-microstate route', case['equal_time_all_microstate_route_mismatch'], flush=True)
    print('Universal epsilon/h^2 formula:', [r['metric_error_divided_by_loop_scale_squared'] for r in record['universal_formula_evaluations']])


if __name__ == '__main__':
    main()
