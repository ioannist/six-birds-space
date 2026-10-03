"""Constructive recursive-cell gasket control, using the original graph.

No planar coordinates enter the cell lens. Finite audits check nested ownership,
exact transition counts, the cell-contact graph, metric bounds, and a strict
Hilbert obstruction. The universal argument is recorded in the companion note.
"""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel
from geo_sbt.substrates.sierpinski import _SierpinskiGraph, _build_level, _copy_graph, _union_find, _compress_graph
from certify_local_nonembedding import log_enclosure, check_gap

INTERFACE_FLUX = {1: Fraction(1, 4), 5: Fraction(11905, 16384)}


def cell_cover(s, m):
    if type(s) is not int or type(m) is not int or s < 1 or m < 0:
        raise ValueError('cell level must be positive and macro depth nonnegative integers')
    if m == 0:
        graph = _build_level(s)
        return graph, [set(graph.adj)]
    prev, cells = cell_cover(s, m-1)
    n = len(prev.adj)
    copies = [_copy_graph(prev, i*n) for i in range(3)]
    parent, find, union = _union_find(3*n)
    union(copies[0].corners[1], copies[1].corners[0])
    union(copies[0].corners[2], copies[2].corners[0])
    union(copies[1].corners[2], copies[2].corners[1])
    merged = {}
    for graph in copies:
        merged.update({z: set(adjacent) for z, adjacent in graph.adj.items()})
    roots = {}
    mapping = {}
    for z in merged:
        root = find(z)
        if root not in roots:
            roots[root] = len(roots)
        mapping[z] = roots[root]
    corners = (mapping[copies[0].corners[0]], mapping[copies[1].corners[1]], mapping[copies[2].corners[2]])
    graph = _SierpinskiGraph(_compress_graph(merged, parent), corners)
    lifted = [{mapping[z+i*n] for z in cell} for i in range(3) for cell in cells]
    return graph, lifted


def cell_labels(graph, cells, s):
    labels = np.full(len(graph.adj), -1, dtype=int)
    membership = np.zeros(len(graph.adj), dtype=int)
    V = (3**(s+1)+3)//2
    for i, cell in enumerate(cells):
        if len(cell) != V:
            raise AssertionError('incorrect recursive cell size')
        ix = np.array(sorted(cell), dtype=int)
        membership[ix] += 1
        labels[ix[labels[ix] < 0]] = i
    sizes = np.bincount(labels, minlength=len(cells))
    if (labels < 0).any() or membership.max() > 2 or sizes.min() < V-3 or sizes.max() > V:
        raise AssertionError('invalid disjoint ownership of the recursive cover')
    return labels, sizes


def contact_graph(m):
    if type(m) is not int or m < 0:
        raise ValueError('macro depth must be a nonnegative integer')
    if m == 0:
        return {0: set()}
    prev = contact_graph(m-1)
    q = len(prev)
    result = {z+i*q: {y+i*q for y in adjacent} for i in range(3) for z, adjacent in prev.items()}
    corners = (0, (q-1)//2, q-1)
    for x, y in [(corners[1], q+corners[0]), (corners[2], 2*q+corners[0]),
                 (q+corners[2], 2*q+corners[1])]:
        result[x].add(y); result[y].add(x)
    return result


def integer_distances(adj):
    n = len(adj)
    distances = np.full((n, n), -1, dtype=int)
    for source in range(n):
        distances[source, source] = 0
        queue = deque([source])
        while queue:
            z = queue.popleft()
            for y in adj[z]:
                if distances[source, y] < 0:
                    distances[source, y] = distances[source, z]+1
                    queue.append(y)
    if (distances < 0).any():
        raise AssertionError('cell graph disconnected')
    return distances


def sparse_transition(adj):
    rows, cols, data = [], [], []
    for z, adjacent in adj.items():
        if len(adjacent) not in [2, 4]:
            raise AssertionError('unexpected micro degree')
        rows.append(z); cols.append(z); data.append(4)
        for y in adjacent:
            rows.append(z); cols.append(y); data.append(8//(2*len(adjacent)))
    return csr_matrix((np.array(data, dtype=np.int64), (rows, cols)), shape=(len(adj), len(adj)))


def layer(s, m, expected_graph, tau=5):
    if type(s) is not int or type(m) is not int or s < 3 or m < 0:
        raise ValueError('certified layer requires cell level at least three and nonnegative macro depth')
    if tau not in INTERFACE_FLUX:
        raise ValueError('only stages one and five have an established interface law')
    flux = INTERFACE_FLUX[tau]
    graph, cells = cell_cover(s, m)
    if graph.adj != expected_graph.adj or graph.corners != expected_graph.corners:
        raise AssertionError('cell construction changed the original substrate')
    labels, sizes = cell_labels(graph, cells, s)
    n, count = len(labels), len(cells)
    C = csr_matrix((np.ones(n, dtype=np.int64), (np.arange(n), labels)), shape=(n, count))
    P8 = sparse_transition(graph.adj)
    B8 = C
    for _ in range(tau):
        B8 = P8 @ B8
    denominator = 8**tau
    if not np.all(np.asarray(B8.sum(axis=1)).ravel() == denominator):
        raise AssertionError('incorrect probability row mass')
    aggregate = (C.T @ B8).tocsr()
    macro = contact_graph(m)
    for x in range(count):
        actual = {int(y): int(v) for y, v in zip(aggregate[x].indices, aggregate[x].data) if y != x and v}
        if actual != {y: int(flux*denominator) for y in macro[x]}:
            raise AssertionError('cell-interface transition count failed')
    K = aggregate.astype(float).multiply((1/(denominator*sizes))[:, None]).tocsr()
    B = B8.astype(float)/denominator
    difference = B @ K-B
    difference.data = np.abs(difference.data)
    delta = float(.5*np.asarray(difference.sum(axis=1)).max())
    escape = np.array([len(macro[x])*float(flux)/int(sizes[x]) for x in range(count)])
    V = (3**(s+1)+3)//2
    bound = 3*flux/(V-3)
    if max(delta, float(escape.max())) > float(bound)+1e-12:
        raise AssertionError('closure or prototype bound failed')
    unit = integer_distances(macro)
    if unit.max() != 2**m-1:
        raise AssertionError('cell graph diameter law failed')
    eta = flux/(2*V)
    d = all_pairs_shortest_path(cost_matrix_from_kernel(K.toarray(), eta=float(eta), eps_edge=0, symmetrize='weight_avg'))
    c = math.log(float(V/flux))
    epsilon = math.log(V/(V-3))/c
    normalized = d/(2**m*c)
    readout_error = float(np.abs(normalized-unit/(2**m)).max())
    if readout_error > epsilon+1e-12:
        raise AssertionError('unit-graph readout bound failed')
    record = {'micro_cell_level': s, 'macro_depth': m, 'macro_states': count,
              'tau': tau, 'interface_flux': str(flux),
              'cell_size_before_ownership': V, 'minimum_fiber_size': int(sizes.min()),
              'maximum_fiber_size': int(sizes.max()), 'delta': delta, 'escape_max': float(escape.max()),
              'delta_and_escape_bound': str(bound), 'eta': str(eta), 'eps_edge': 0,
              'normalized_readout_error': readout_error, 'normalized_readout_error_bound': epsilon,
              'unit_graph_diameter': int(unit.max()), 'inf_count': int((~np.isfinite(d)).sum())}
    if m >= 2:
        vertices = [1, 7, 2, 5]
        lengths = {f'{i}{j}': int(unit[vertices[i], vertices[j]]) for i in range(4) for j in range(i+1, 4)}
        if lengths != {'01': 3, '02': 1, '03': 2, '12': 2, '13': 1, '23': 3}:
            raise AssertionError('isometric central-hexagon witness failed')
        lo, _ = log_enclosure((V-3)/flux)
        _, hi = log_enclosure(V/flux)
        bounds = {key: (length*lo, length*hi) for key, length in lengths.items()}
        gap = check_gap(bounds, hi/4)
        record['nonembedding'] = {'vertices': vertices, 'distance_bounds': {key: [str(x) for x in pair] for key, pair in bounds.items()},
                                  'absolute_fit_error_excluded': str(hi/4), 'strict_rational_gap': str(gap),
                                  'corner_patch_radius_upper_bound': str(3*hi),
                                  'relative_fit_error_lower_bound': '1/12',
                                  'normalized_corner_patch_radius_upper_bound': 3/(2**m)}
    return record, (labels, C, K, B, normalized, unit, P8)


def finite_ladder(s, m, tau=5):
    if s < 3 or m < 3:
        raise ValueError('audit ladder requires cell level and macro depth at least three')
    total = s+m
    original = _build_level(total)
    records, arrays = [], []
    for offset in [2, 1, 0]:
        record, values = layer(s+offset, m-offset, original, tau=tau)
        records.append(record); arrays.append(values)
    distortion = []
    for i in range(2):
        fine_labels = arrays[i+1][0]
        if not np.array_equal(arrays[i][0], fine_labels//3):
            raise AssertionError('ownership maps are not nested')
        count_fine = records[i+1]['macro_states']
        projection = np.arange(count_fine)//3
        unit_gap = arrays[i+1][5]-2*arrays[i][5][projection[:, None], projection]
        if np.abs(unit_gap).max() > 1:
            raise AssertionError('unit-metric refinement law failed')
        actual = float(np.abs(arrays[i+1][4]-arrays[i][4][projection[:, None], projection]).max())
        bound = 2**(-records[i+1]['macro_depth'])+records[i]['normalized_readout_error_bound']+records[i+1]['normalized_readout_error_bound']
        if actual > bound+1e-12:
            raise AssertionError('metric refinement bound failed')
        distortion.append({'max_normalized_distortion': actual, 'analytic_bound': bound})
    P = arrays[2][6].astype(float)/8
    Bc, Bmid = arrays[0][3], arrays[1][3]
    Cf, Cm = arrays[2][1], arrays[1][1]
    nf, nm = np.asarray(Cf.sum(axis=0)).ravel(), np.asarray(Cm.sum(axis=0)).ravel()
    Uf = Cf.T.astype(float).multiply((1/nf)[:, None]).tocsr()
    Um = Cm.T.astype(float).multiply((1/nm)[:, None]).tocsr()
    twice = Bc
    for _ in range(tau):
        twice = P @ twice
    direct = Uf @ twice
    via = (Uf @ Bmid) @ (Um @ Bc)
    diff = direct-via; diff.data = np.abs(diff.data)
    mismatch = float(.5*np.asarray(diff.sum(axis=1)).max())
    V = (3**(s+1)+3)//2
    Vm = (3**(s+2)+3)//2
    flux = INTERFACE_FLUX[tau]
    route_bound = Fraction(6*tau+2, 4*(V-3)-2)+3*flux/(V-3)+3*flux/(Vm-3)
    if mismatch > float(route_bound)+1e-12:
        raise AssertionError('prototype-input route bound failed')
    return {'micro_level': total, 'n_micro': len(original.adj), 'tau': tau, 'lazy': '1/2',
            'levels': records, 'distortions': distortion, 'prototype_input_route_mismatch': mismatch,
            'prototype_input_route_bound': str(route_bound), 'route_domain': f'finest macro prototypes, equal total staging {2*tau}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/recursive_gasket_review')
    args = parser.parse_args()
    sources = sorted((ROOT / 'src').rglob('*.py'))+[Path(__file__).resolve(), ROOT/'scripts/certify_local_nonembedding.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    calibration, _ = layer(3, 1, _build_level(4), tau=5)
    cases = [finite_ladder(3, 3), finite_ladder(4, 3), finite_ladder(5, 4)]
    limit_witness_checks = []
    for m in range(3, 7):
        q, p = 3**(m-1), 3**(m-2)
        vertices = [(q-1)//2, 2*q+(p-1)//2, 2*p-1, 2*q-1]
        unit = integer_distances(contact_graph(m))
        actual = {f'{i}{j}': int(unit[vertices[i], vertices[j]]) for i in range(4) for j in range(i+1, 4)}
        a = 2**(m-2)
        expected = {'01': 3*a-1, '02': a-1, '03': 2*a, '12': 2*a, '13': a+1, '23': 3*a-1}
        if actual != expected:
            raise AssertionError('address-prefix distance formula failed')
        limit_witness_checks.append({'macro_depth': m, 'vertices': vertices, 'integer_distances': actual,
                                     'scope': 'finite check of the separately proved prefix formula'})
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise AssertionError('sources changed during audit')
    evidence = {'scope': 'Recursive-cell positive control on the original gasket kernel; no learned-lens claim. Universal argument is in the companion proof note.',
                'source_sha256': hashes, 'flux_calibration': calibration,
                'finite_ladders': cases, 'limit_witness_prefix_checks': limit_witness_checks}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for case in cases:
        print('gasket level', case['micro_level'], 'states', case['n_micro'], 'escape', case['levels'][-1]['escape_max'],
              'route mismatch', case['prototype_input_route_mismatch'], flush=True)


if __name__ == '__main__':
    main()
