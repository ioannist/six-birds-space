"""Exact optimization obstruction for canonical grid/gasket prototypes.

The partition is fixed by the previously recorded canonical certificate.
All fiber-supported stochastic prototype choices are allowed. Independent
CLI verification reconstructs the staged counts and the optimum.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from geo_sbt.substrates.sierpinski import sierpinski

PARTITIONS = ROOT/'docs/notes/local_nonembedding_review_20261003/certificate.json'


def staged_counts(kind, size, labels, tau=5):
    if type(size) is not int or size < 1 or type(tau) is not int or tau < 0:
        raise ValueError('size and stage require integer domains')
    if kind == 'grid' and size >= 2:
        adj = {i*size+j: {r*size+c for r, c in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                         if 0 <= r < size and 0 <= c < size}
               for i in range(size) for j in range(size)}
        denominator = 24
    elif kind == 'gasket':
        adj, _ = sierpinski(size, lazy=.5)
        denominator = 8
    else:
        raise ValueError('unsupported exact substrate')
    n = len(adj)
    if (len(labels) != n or any(type(x) is not int or x < 0 for x in labels)
            or set(labels) != set(range(max(labels)+1))):
        raise ValueError('labels must be integer, surjective and cover all microstates')
    mass = denominator**tau
    if n*mass >= 2**63:
        raise ValueError('exact int64 range exceeded')
    rows, cols, weights = [], [], []
    for z, adjacent in adj.items():
        if denominator % (2*len(adjacent)):
            raise ValueError('integer denominator does not represent micro moves')
        rows.append(z); cols.append(z); weights.append(denominator//2)
        for y in adjacent:
            rows.append(z); cols.append(y); weights.append(denominator//(2*len(adjacent)))
    P = csr_matrix((np.array(weights, dtype=np.int64), (rows, cols)), shape=(n, n))
    result = np.zeros((n, max(labels)+1), dtype=np.int64)
    result[np.arange(n), labels] = 1
    for stage in range(1, tau+1):
        result = P @ result
        if (result < 0).any() or not np.all(result.sum(axis=1) == denominator**stage):
            raise AssertionError('exact row mass failed')
    return result, mass


def analyze(kind, size, labels):
    counts, mass = staged_counts(kind, size, labels)
    optimum, uniform, safest = [], [], []
    for x in range(counts.shape[1]):
        fiber = [z for z, label in enumerate(labels) if label == x]
        values = [int(counts[z, x]) for z in fiber]
        best = max(values)
        safest.append(fiber[values.index(best)])
        optimum.append(Fraction(mass-best, mass))
        uniform.append(1-Fraction(sum(values), mass*len(fiber)))
    lower = max(optimum)
    x = optimum.index(lower)
    fiber = [z for z, label in enumerate(labels) if label == x]
    return {'kind': kind, 'size': size, 'tau': 5, 'lazy': '1/2', 'labels': labels,
            'scope': 'All fiber-supported stochastic prototypes, fixed recorded canonical partition.',
            'denominator': mass, 'optimal_escape_by_fiber': [str(v) for v in optimum],
            'uniform_escape_by_fiber': [str(v) for v in uniform],
            'optimal_point_prototype_microstates': safest,
            'optimal_worst_prototype_defect': str(lower), 'uniform_worst_prototype_defect': str(max(uniform)),
            'worst_fiber': x, 'worst_fiber_microstates': fiber,
            'worst_fiber_return_counts': [int(counts[z, x]) for z in fiber]}


def canonical_results():
    saved = json.loads(PARTITIONS.read_text())
    cases = [c for c in saved['cases'] if c['configuration_scope'] == 'canonical']
    if [(c['kind'], c['size']) for c in cases] != [('grid', 25), ('gasket', 5)]:
        raise ValueError('incorrect canonical partition inputs')
    if any(c['tau'] != 5 or c['lazy'] != '1/2' or c['prototypes'] != 'uniform'
           or c['n_eigs'] != 6 or c['seed'] != 0 or c['m'] != 128
           or c['lens_levels'] != [4, 8, 16, 32, 64, 128] for c in cases):
        raise ValueError('incorrect canonical partition metadata')
    return [analyze(c['kind'], c['size'], c['labels']) for c in cases]


def verify(record):
    for case in record['cases']:
        if (any(type(case[name]) is not int for name in ['size', 'tau', 'denominator', 'worst_fiber'])
                or any(type(value) is not int for name in ['labels', 'optimal_point_prototype_microstates',
                                                          'worst_fiber_microstates', 'worst_fiber_return_counts']
                       for value in case[name])):
            raise ValueError('prototype certificate counts and labels require integers')
    if record['cases'] != canonical_results():
        raise ValueError('saved prototype obstruction disagrees with exact reconstruction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/prototype_obstruction_review')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(json.loads(args.verify.read_text()))
        print('Exact canonical prototype optima and universal defect lower bounds verified.')
        return
    cases = canonical_results()
    for case in cases:
        if case['optimal_worst_prototype_defect'] != case['uniform_worst_prototype_defect']:
            raise AssertionError('uniform prototypes do not attain this proposed optimum')
    paths = [Path(__file__).resolve(), PARTITIONS, ROOT/'src/geo_sbt/substrates/sierpinski.py']
    record = {'scope': 'Exact fixed-partition prototype optimization; no efficient lens-discovery or whole-pipeline optimum claim.',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'cases': cases}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evidence.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for case in cases:
        print(case['kind'], 'best possible worst prototype defect', case['optimal_worst_prototype_defect'])


if __name__ == '__main__':
    main()
