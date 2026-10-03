"""Audit the uniform quadratic-limit proof and a matched correlated-walk control.

The universal Fourier estimate is proved in the companion note. Finite Gaussian
comparisons below are floating checks, not certificates of that estimate.
The matched control uses exact integer recurrence and rational log enclosures;
verification reconstructs its counts rather than accepting arbitrary tables.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'src')]
from certify_local_nonembedding import log_enclosure
from experiments.runners.pythagoras_rw_grid import torus_rw_distribution

ISOTROPIC = [(0, 0, 4), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1)]
CORRELATED = [(0, 0, 8), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
              (1, 1, 2), (-1, -1, 2)]
CONTROL_STAGES = [16, 32, 64, 128]


def integer_counts(max_stage, steps=CORRELATED):
    if type(max_stage) is not int or max_stage < 1:
        raise ValueError('stage must be a positive integer')
    if (not steps or any(type(x) is not int or type(y) is not int
                         or type(w) is not int or w <= 0 for x, y, w in steps)):
        raise ValueError('integer recurrence needs integer displacements and positive integer weights')
    denominator = sum(w for _, _, w in steps)
    counts = {(0, 0): 1}
    for n in range(1, max_stage+1):
        updated = {}
        for (x, y), value in counts.items():
            for dx, dy, w in steps:
                key = x+dx, y+dy
                updated[key] = updated.get(key, 0)+w*value
        counts = updated
        if sum(counts.values()) != denominator**n:
            raise AssertionError('integer mass is not conserved')
        yield n, counts


def isotropic_bounds(n, radius):
    """Analytic formulas evaluated in floats; radius is in units sqrt(n).

    The statement is for every integer lattice point with Euclidean norm
    <= radius*sqrt(n). The written proof supplies the universal bound.
    """
    if type(n) is not int or n < 2 or not math.isfinite(radius) or radius < 0:
        raise ValueError('bound requires integer stage >=2 and finite radius >=0')
    a = 11/96
    absolute = (7/(768*math.pi*a**3))*n/(n-1)**3
    absolute += math.pi/(2*n)*math.exp(-n/(2*math.pi**2))
    absolute += 2/(math.pi*n)*math.exp(-n/8)
    relative = absolute/(2/(math.pi*n)*math.exp(-2*radius**2))
    cost = 2*relative/(1-relative) if relative < 1 else None
    return {'uniform_absolute_probability_bound': absolute,
            'uniform_relative_probability_bound': relative,
            'uniform_centered_cost_bound': cost,
            'uniform_axis_residual_bound': None if cost is None else 3*cost,
            'uniform_normalized_sqrt_readout_bound': None if cost is None else math.sqrt(cost/2)}


def floating_distribution(n, steps=ISOTROPIC):
    """Nonnegative convolution on the full lattice support, with no FFT floor."""
    if type(n) is not int or n < 0:
        raise ValueError('stage must be a nonnegative integer')
    denominator = sum(w for _, _, w in steps)
    data = np.ones((1, 1))
    for _ in range(n):
        updated = np.zeros((data.shape[0]+2, data.shape[1]+2))
        for dx, dy, w in steps:
            updated[1+dx:1+dx+data.shape[0], 1+dy:1+dy+data.shape[1]] += data*(w/denominator)
        data = updated
    if not np.isclose(data.sum(), 1, atol=1e-12, rtol=0) or (data < 0).any():
        raise AssertionError('floating convolution mass failed')
    return data


def floating_gaussian_check(n):
    p = floating_distribution(n)
    x = np.arange(-n, n+1)
    square_radius = x[:, None]**2+x[None, :]**2
    g = 2/(math.pi*n)*np.exp(-2*square_radius/n)
    bounds = isotropic_bounds(n, math.sqrt(2))
    # Beyond this box p is exactly zero; the largest Gaussian value there
    # is bounded by the value at displacement (n+1,0).
    outside = 2/(math.pi*n)*math.exp(-2*(n+1)**2/n)
    actual = max(float(np.abs(p-g).max()), outside)
    if actual > bounds['uniform_absolute_probability_bound']+1e-14:
        raise AssertionError('analytic uniform probability bound failed')
    D = math.isqrt(n)
    window = p[n-D:n+D+1, n-D:n+D+1]
    target = 2*(np.arange(-D, D+1)[:, None]**2+np.arange(-D, D+1)[None, :]**2)/n
    centered = np.log(p[n, n]/window)
    actual_cost = float(np.abs(centered-target).max())
    if (bounds['uniform_centered_cost_bound'] is not None
            and actual_cost > bounds['uniform_centered_cost_bound']+1e-12):
        raise AssertionError('analytic cost bridge failed')
    return {'stage': n, 'scope': 'floating exhaustive support-box comparison plus analytic outside-box bound',
            'measured_uniform_absolute_probability_error': actual,
            'measured_window_centered_cost_error': actual_cost,
            'window_D': D, **bounds}


def control_row(n, counts):
    k = math.isqrt(n)
    points = [(0, 0), (k, 0), (0, k), (k, k)]
    origin = counts[(0, 0)]
    probabilities = [counts[point] for point in points]
    if any(value <= 0 or value > origin for value in probabilities):
        raise AssertionError('control support or centered-cost sign failed')
    intervals = [log_enclosure(Fraction(origin, value)) for value in probabilities[1:]]
    ax, ay, diagonal = intervals
    residual = diagonal[0]-ax[1]-ay[1], diagonal[1]-ax[0]-ay[0]
    if residual[1] >= -1:
        raise AssertionError('the proposed matched control has no strong residual')
    return {'stage': n, 'k': k, 'points_and_counts': [[*point, value] for point, value in zip(points, probabilities)],
            'centered_cost_intervals': [[str(lo), str(hi)] for lo, hi in intervals],
            'exact_residual_interval': [str(value) for value in residual],
            'covariance_quadratic_residual': str(-Fraction(16, 5)*k*k/n),
            'residual_upper_bound_excluded': '-1'}


def generate_control():
    result = []
    for n, counts in integer_counts(max(CONTROL_STAGES)):
        if n in CONTROL_STAGES:
            result.append(control_row(n, counts))
    return {'steps': [list(step) for step in CORRELATED], 'denominator': 16,
            'covariance': [['3/8', '1/4'], ['1/4', '3/8']],
            'quadratic_cost': '(12/5)*(x*x+y*y)/n-(16/5)*x*y/n',
            'scope': 'Matched lazy finite-step dynamics; failure of the supplied coordinate-axis identity, with a weighted inner-product limit.',
            'rows': result}


def verify_control(record):
    expected_metadata = {'steps': [list(step) for step in CORRELATED], 'denominator': 16,
                         'covariance': [['3/8', '1/4'], ['1/4', '3/8']],
                         'quadratic_cost': '(12/5)*(x*x+y*y)/n-(16/5)*x*y/n'}
    for name, value in expected_metadata.items():
        if record[name] != value:
            raise ValueError('control dynamics or theoretical coefficient changed')
    for row in record['rows']:
        if (type(row['stage']) is not int or type(row['k']) is not int
                or len(row['points_and_counts']) != 4
                or any(len(point) != 3 or any(type(x) is not int for x in point)
                       for point in row['points_and_counts'])):
            raise ValueError('control stages, displacements and counts must be integers')
    if [row['stage'] for row in record['rows']] != CONTROL_STAGES:
        raise ValueError('missing, duplicated or changed control stages')
    rows = iter(record['rows'])
    for n, counts in integer_counts(max(CONTROL_STAGES)):
        if n in CONTROL_STAGES and next(rows) != control_row(n, counts):
            raise ValueError('control counts or intervals disagree with exact reconstruction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/quadratic_limit_review')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify_control(json.loads(args.verify.read_text())['matched_control'])
        print('Exact matched-control probabilities and residual exclusions verified.')
        return
    paths = [Path(__file__).resolve(), ROOT/'scripts/certify_local_nonembedding.py',
             ROOT/'experiments/runners/pythagoras_rw_grid.py', ROOT/'experiments/config_validation.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    checks = [floating_gaussian_check(n) for n in [16, 32, 64, 128, 256]]
    # Public torus implementation must agree with the infinite-lattice walk
    # when the declared no-aliasing condition holds.
    N, n = 33, 16
    torus = torus_rw_distribution(N, .5, n)
    infinite = floating_distribution(n)
    ix = np.arange(-n, n+1) % N
    if not np.allclose(torus[ix[:, None], ix], infinite, atol=1e-14, rtol=0):
        raise AssertionError('public torus walk differs from the audited lattice kernel')
    control = generate_control()
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise AssertionError('sources changed during audit')
    record = {'scope': 'Written universal Fourier proof; floating finite checks and exact matched-control certificates are separately labelled.',
              'source_sha256': hashes, 'finite_checks': checks, 'matched_control': control,
              'analytic_bounds_not_executions': [{'stage': n, 'radius_in_sqrt_stage_units': math.sqrt(2),
                                                **isotropic_bounds(n, math.sqrt(2))}
                                               for n in [512, 1024, 4096, 16384, 65536]]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evidence.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for row in control['rows']:
        print(row['stage'], 'control residual', [float(Fraction(x)) for x in row['exact_residual_interval']], flush=True)


if __name__ == '__main__':
    main()
