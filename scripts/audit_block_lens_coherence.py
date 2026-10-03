"""Constructive block-lens control for the existing open lazy grid.

This is a supplied coordinate-based lens, not the learned spectral pipeline.
Exact boundary counting and an analytic argument establish a joint size/lens
regime. Finite path metrics and route comparisons are floating-point audits.
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
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel
from geo_sbt.packaging import idempotence_defect_from_factors, make_C


def validate(M: int, b: int) -> None:
    if type(M) is not int or type(b) is not int or M < 2 or b < 2:
        raise ValueError('macro side and block width must be integers at least two')


def block_labels(M: int, b: int) -> np.ndarray:
    validate(M, b)
    N = M*b
    r, c = np.indices((N, N))
    return ((r // b)*M + c // b).ravel()


def neighbors(N: int, r: int, c: int):
    return [(x, y) for x, y in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if 0 <= x < N and 0 <= y < N]


def sparse_grid(N: int):
    """Same laziness and neighbor normalization as grid_2d(N, .5)."""
    if type(N) is not int or N < 2:
        raise ValueError('sparse control requires an integer micro side at least two')
    rows, cols, data = [], [], []
    for r in range(N):
        for c in range(N):
            z = r*N+c
            adjacent = neighbors(N, r, c)
            rows.append(z); cols.append(z); data.append(.5)
            for x, y in adjacent:
                rows.append(z); cols.append(x*N+y); data.append(.5/len(adjacent))
    return csr_matrix((data, (rows, cols)), shape=(N*N, N*N))


def formula_kernel(M: int, b: int) -> list[dict[int, Fraction]]:
    """Boundary-count formula, including degree-three substrate boundary nodes."""
    validate(M, b)
    result = []
    for r in range(M):
        for c in range(M):
            row = {}
            for x, y in neighbors(M, r, c):
                transverse = c if x != r else r
                boundary = int(transverse in [0, M-1])
                row[x*M+y] = Fraction(1, 8*b) + Fraction(boundary, 24*b*b)
            row[r*M+c] = 1 - sum(row.values())
            result.append(row)
    return result


def count_kernel(M: int, b: int) -> list[dict[int, Fraction]]:
    """Independent rational enumeration of U P C from the microstate moves."""
    validate(M, b)
    N = M*b
    result = [{} for _ in range(M*M)]
    def add(row, target, value):
        row[target] = row.get(target, Fraction(0)) + value
    for r in range(N):
        for c in range(N):
            label = (r//b)*M+c//b
            row = result[label]
            add(row, label, Fraction(1, 2*b*b))
            adjacent = neighbors(N, r, c)
            for x, y in adjacent:
                add(row, (x//b)*M+y//b, Fraction(1, 2*len(adjacent)*b*b))
    return result


def verify_counts(M: int, b: int) -> list[dict[int, Fraction]]:
    rows = count_kernel(M, b)
    if rows != formula_kernel(M, b):
        raise AssertionError('boundary-count formula disagrees with micro dynamics')
    for x, row in enumerate(rows):
        if sum(row.values()) != 1 or min(row.values()) <= 0:
            raise AssertionError('macro probability row is invalid')
        if 1-row[x] > Fraction(1, 2*b):
            raise AssertionError('escape bound failed')
        for y, weight in row.items():
            if rows[y].get(x) != weight:
                raise AssertionError('macro kernel is not symmetric')
            if x != y and not Fraction(1, 8*b) <= weight <= Fraction(1, 8*b)+Fraction(1, 24*b*b):
                raise AssertionError('edge probability bracket failed')
    return rows


def metric_error_bound(b: int) -> float:
    return 2*math.log1p(1/(3*b))/math.log(8*b)


def reference_distance(M: int) -> np.ndarray:
    r, c = np.divmod(np.arange(M*M), M)
    return (np.abs(r[:, None]-r) + np.abs(c[:, None]-c))/M


def finite_ladder(k: int) -> dict:
    # Fixed ratios across three levels; b_min and M_min grow together.
    t = 2**k
    N = 4*t*t
    blocks = [4*t, 2*t, t]
    P = sparse_grid(N)
    levels, arrays = [], []
    for b in blocks:
        M = N//b
        exact = verify_counts(M, b)
        labels = block_labels(M, b)
        C = make_C(labels, M*M)
        U = C.T/(b*b)
        B = P @ C
        K = U @ B
        oracle = np.zeros_like(K)
        for x, row in enumerate(exact):
            for y, weight in row.items():
                oracle[x, y] = float(weight)
        if not np.allclose(K, oracle, atol=1e-13, rtol=0):
            raise AssertionError('floating macro kernel disagrees with exact count')
        # The floor stays below every positive off-diagonal probability even
        # in the growing family. A fixed floor would eventually clip its edges.
        eta = Fraction(1, 16*b)
        d = all_pairs_shortest_path(cost_matrix_from_kernel(K, eta=float(eta), eps_edge=0, symmetrize='weight_avg'))
        normalized = d/(M*math.log(8*b))
        epsilon = metric_error_bound(b)
        readout_error = float(np.max(np.abs(normalized-reference_distance(M))))
        delta = idempotence_defect_from_factors(B, U)
        escape = float(np.max(1-np.diag(K)))
        if not np.isfinite(d).all() or max(delta, escape) > 1/(2*b)+1e-12 or readout_error > epsilon+1e-12:
            raise AssertionError('finite coherence audit failed its analytic bound')
        levels.append({'block_width': b, 'macro_side': M, 'macro_states': M*M,
                       'delta': delta, 'escape_max': escape, 'escape_and_delta_bound': str(Fraction(1, 2*b)),
                       'normalized_l1_readout_error': readout_error, 'normalized_l1_error_bound': epsilon,
                       'eta': str(eta), 'eps_edge': 0,
                       'inf_count': int((~np.isfinite(d)).sum())})
        arrays.append((C, U, B, normalized))
    distortions = []
    for i in range(2):
        coarse_M, fine_M = levels[i]['macro_side'], levels[i+1]['macro_side']
        r, c = np.divmod(np.arange(fine_M*fine_M), fine_M)
        projection = (r//2)*coarse_M+c//2
        actual = float(np.max(np.abs(arrays[i+1][3]-arrays[i][3][projection[:, None], projection])))
        # The exact per-axis rounding bound is 1/fine_M for ratio two.
        bound = 2/fine_M + levels[i]['normalized_l1_error_bound'] + levels[i+1]['normalized_l1_error_bound']
        if actual > bound+1e-12:
            raise AssertionError('normalized interscale distortion failed')
        distortions.append({'max_normalized_distortion': actual, 'analytic_bound': bound,
                            'raw_cost_rescaling_alpha': fine_M*math.log(8*levels[i+1]['block_width'])/(coarse_M*math.log(8*levels[i]['block_width'])),
                            'coarse_macro_side': coarse_M, 'fine_macro_side': fine_M})
    Cc, Uc, Bc, _ = arrays[0]
    Cm, Um, Bm, _ = arrays[1]
    Cf, Uf, Bf, _ = arrays[2]
    direct = Uf @ (P @ Bc)
    via = (Uf @ Bm) @ (Um @ Bc)
    mismatch = float(.5*np.abs(direct-via).sum(axis=1).max())
    route_bound = 8.5/blocks[2]+.5/blocks[1]
    if mismatch > route_bound+1e-12:
        raise AssertionError('prototype-input route bound failed')
    return {'k': k, 'micro_side': N, 'lazy': .5, 'tau': 1, 'levels': levels,
            'distortions': distortions, 'prototype_input_route_mismatch': mismatch,
            'prototype_input_route_bound': route_bound,
            'route_domain': 'all finest prototype rows; equal total staging two; not the microstate simplex'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/block_lens_review')
    args = parser.parse_args()
    sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    measured = [finite_ladder(k) for k in [1, 2]]
    family_bounds = []
    for k in range(1, 9):
        t = 2**k
        family_bounds.append({'k': k, 'micro_side': 4*t*t,
                              'worst_escape_and_delta_bound': str(Fraction(1, 2*t)),
                              'worst_normalized_distortion_bound': 1/t+metric_error_bound(t)+metric_error_bound(2*t),
                              'prototype_input_route_bound': 8.75/t,
                              'scope': 'analytic formula, not a run at this substrate size'})
    evidence = {'scope': 'Explicit block-lens control on the existing open grid; no learned-lens, Euclidean, curvature, or fractal claim.',
                'source_sha256': hashes, 'finite_ladders': measured, 'analytic_family_bounds': family_bounds}
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise AssertionError('sources changed during audit')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'evidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for row in measured:
        print('micro side', row['micro_side'], 'worst escape', max(x['escape_max'] for x in row['levels']),
              'route mismatch', row['prototype_input_route_mismatch'])


if __name__ == '__main__':
    main()
