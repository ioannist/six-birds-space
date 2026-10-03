"""Stage-five extension of the open-grid block control.

Uses unchanged micro dynamics and explicit block lenses. Bounds are proved
in the companion note; these finite audits test every macro pair and row.
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
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from audit_block_lens_coherence import block_labels, sparse_grid, reference_distance
from geo_sbt.packaging import make_C, idempotence_defect_from_factors
from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel


def layer(P, N, b):
    if type(b) is not int or b < 8 or N % b or N//b < 2:
        raise ValueError('stage-five blocks require integer width at least eight and macro side at least two')
    M = N//b
    labels = block_labels(M, b)
    C = make_C(labels, M*M)
    U = C.T/(b*b)
    B = C
    for _ in range(5):
        B = P @ B
    K = U @ B
    W = (K+K.T)/2
    row, col = np.divmod(np.arange(M*M), M)
    for x in range(M*M):
        for y in range(M*M):
            if x == y:
                continue
            dr, dc = abs(row[x]-row[y]), abs(col[x]-col[y])
            if max(dr, dc) > 1:
                if W[x, y] != 0:
                    raise AssertionError('unexpected multi-block jump')
                continue
            length = dr+dc
            lower = 1/(128*b) if length == 1 else 1/(512*b*b)
            upper = (5/b)**length
            if not lower-1e-14 <= W[x, y] <= upper+1e-14:
                raise AssertionError('macro edge probability bracket failed')
    eta = Fraction(1, 1024*b*b)
    d = all_pairs_shortest_path(cost_matrix_from_kernel(K, eta=float(eta), eps_edge=0, symmetrize='weight_avg'))
    c = math.log(128*b)
    normalized = d/(M*c)
    epsilon = 2*math.log(640)/c
    readout_error = float(np.abs(normalized-reference_distance(M)).max())
    delta = idempotence_defect_from_factors(B, U)
    escape = float(np.max(1-np.diag(K)))
    if not np.isfinite(d).all() or max(delta, escape) > 5/b+1e-12 or readout_error > epsilon+1e-12:
        raise AssertionError('stage-five coherence bound failed')
    record = {'block_width': b, 'macro_side': M, 'macro_states': M*M,
              'delta': delta, 'escape_max': escape, 'delta_and_escape_bound': str(Fraction(5, b)),
              'eta': str(eta), 'eps_edge': 0, 'normalized_l1_readout_error': readout_error,
              'normalized_l1_readout_error_bound': epsilon, 'inf_count': int((~np.isfinite(d)).sum())}
    return record, (C, U, B, normalized)


def finite_ladder(N, blocks):
    if len(blocks) != 3 or blocks[0] != 2*blocks[1] or blocks[1] != 2*blocks[2]:
        raise ValueError('three nested factor-two block widths are required')
    P = sparse_grid(N)
    records, arrays = [], []
    for b in blocks:
        record, values = layer(P, N, b)
        records.append(record); arrays.append(values)
    distortions = []
    for i in range(2):
        fine_M = records[i+1]['macro_side']; coarse_M = records[i]['macro_side']
        r, c = np.divmod(np.arange(fine_M*fine_M), fine_M)
        projection = (r//2)*coarse_M+c//2
        actual = float(np.abs(arrays[i+1][3]-arrays[i][3][projection[:, None], projection]).max())
        bound = 2/fine_M+records[i]['normalized_l1_readout_error_bound']+records[i+1]['normalized_l1_readout_error_bound']
        if actual > bound+1e-12:
            raise AssertionError('stage-five normalized distortion failed')
        distortions.append({'max_normalized_distortion': actual, 'analytic_bound': bound})
    Uf, Um, Bc, Bm = arrays[2][1], arrays[1][1], arrays[0][2], arrays[1][2]
    twice = Bc
    for _ in range(5):
        twice = P @ twice
    direct = Uf @ twice
    via = (Uf @ Bm) @ (Um @ Bc)
    mismatch = float(.5*np.abs(direct-via).sum(axis=1).max())
    route_bound = Fraction(15, blocks[2])+Fraction(5, blocks[1])
    if mismatch > float(route_bound)+1e-12:
        raise AssertionError('stage-five prototype-input route bound failed')
    return {'micro_side': N, 'lazy': '1/2', 'tau': 5, 'levels': records, 'distortions': distortions,
            'prototype_input_route_mismatch': mismatch, 'prototype_input_route_bound': str(route_bound),
            'route_domain': 'finest prototypes, equal total staging ten'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/staged_block_lens_review')
    args = parser.parse_args()
    sources = sorted((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve(), ROOT/'scripts/audit_block_lens_coherence.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    cases = [finite_ladder(64, [32,16,8]), finite_ladder(128, [64,32,16]), finite_ladder(128, [32,16,8])]
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise AssertionError('sources changed during audit')
    record = {'scope': 'Supplied block-lens positive control, unchanged open-grid dynamics, stage five; no learned-lens or Euclidean claim.',
              'source_sha256': hashes, 'finite_ladders': cases}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evidence.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for case in cases:
        print(case['micro_side'], case['levels'][-1]['block_width'], 'escape', case['levels'][-1]['escape_max'],
              'route mismatch', case['prototype_input_route_mismatch'], flush=True)


if __name__ == '__main__':
    main()
