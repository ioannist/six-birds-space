"""Exact finite lower bounds on local Hilbert approximation for learned lenses.

Witness search is floating point. Verification reconstructs rational staged
macro weights, solves maximum-product paths exactly, bounds their logarithms
by rational series, and checks a strict four-point gap. No MDS fit is trusted.
These finite certificates do not prove an asymptotic non-smoothing limit.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from geo_sbt.geometry.holonomy import metric_knn
from geo_sbt.geometry.metric import all_pairs_shortest_path
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.sierpinski import sierpinski

DEGREE = 24
ETA = Fraction(1, 10**12)
EPS_EDGE = Fraction(1, 10**15)
CASES = [(kind, level, side) for level, side in [(3, 7), (4, 11), (5, 19), (6, 33)]
         for kind in ['grid', 'gasket']]
CANONICAL_CASES = [('grid', 5, 25), ('gasket', 5, 25)]
CANONICAL_LEVELS = [4, 8, 16, 32, 64, 128]


def exact_macro_rows(kind, size, labels, tau=5):
    if type(size) is not int or size < 1 or type(tau) is not int or tau < 0:
        raise ValueError('size and stage must be nonnegative integers with positive size')
    if kind == 'grid':
        N = size
        if N < 2:
            raise ValueError('exact grid checker requires side at least two')
        adj = {}
        for r in range(N):
            for c in range(N):
                adj[r*N+c] = [x*N+y for x, y in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                              if 0 <= x < N and 0 <= y < N]
        base = 24
    elif kind == 'gasket':
        adj, _ = sierpinski(size, lazy=.5)
        base = 8
    else:
        raise ValueError('unsupported substrate')
    n = len(adj)
    if len(labels) != n or any(type(x) is not int or x < 0 for x in labels):
        raise ValueError('labels must be nonnegative integers covering all microstates')
    m = max(labels)+1
    if m > n or set(labels) != set(range(m)):
        raise ValueError('prototype fibers must be nonempty')
    if n * base**tau >= 2**63:
        raise ValueError('integer count range would exceed the exact int64 domain')
    rows, cols, weights = [], [], []
    for z, adjacent in adj.items():
        degree = len(adjacent)
        if base % (2*degree):
            raise ValueError('kernel is not represented by this integer denominator')
        rows.append(z); cols.append(z); weights.append(base//2)
        for y in adjacent:
            rows.append(z); cols.append(y); weights.append(base//(2*degree))
    transition = csr_matrix((np.array(weights, dtype=np.int64), (rows, cols)), shape=(n, n))
    labels_array = np.array(labels, dtype=int)
    counts = np.zeros((n, m), dtype=np.int64)
    counts[np.arange(n), labels_array] = 1
    for stage in range(1, tau+1):
        counts = transition @ counts
        if (counts < 0).any() or not np.all(counts.sum(axis=1) == base**stage):
            raise AssertionError('staged integer probability mass is incorrect')
    aggregate = np.zeros((m, m), dtype=np.int64)
    np.add.at(aggregate, labels_array, counts)
    fiber = np.bincount(labels_array, minlength=m)
    denominators = [int(x)*base**tau for x in fiber]
    result = [{} for _ in range(m)]
    for x in range(m):
        for y in range(x+1, m):
            w = (Fraction(int(aggregate[x, y]), denominators[x])+
                 Fraction(int(aggregate[y, x]), denominators[y]))/2
            if w > EPS_EDGE:
                w = max(w, ETA)
                if not 0 < w <= 1:
                    raise AssertionError('invalid symmetric macro weight')
                result[x][y] = result[y][x] = w
    return result


def maximum_products(rows, source):
    """Exact multiplicative Dijkstra; products decrease along every edge."""
    if type(source) is not int or not 0 <= source < len(rows):
        raise ValueError('invalid source')
    if any(type(y) is not int or not 0 <= y < len(rows)
           or not (isinstance(w, Fraction) or type(w) is int) or not 0 < w <= 1
           for row in rows for y, w in row.items()):
        raise ValueError('path weights must be exact rationals in (0, 1] on valid vertices')
    best = [Fraction(0) for _ in rows]
    best[source] = Fraction(1)
    queue = [(-Fraction(1), source)]
    while queue:
        negative, z = heapq.heappop(queue)
        probability = -negative
        if probability != best[z]:
            continue
        for y, weight in rows[z].items():
            candidate = probability*weight
            if candidate > best[y]:
                best[y] = candidate
                heapq.heappush(queue, (-candidate, y))
    if any(x == 0 for x in best):
        raise ValueError('the exact macro graph is disconnected')
    return best


def log_unit_interval(t):
    """Bounds log(t), 1<=t<=2, via 2*atanh((t-1)/(t+1))."""
    t = Fraction(t)
    if not 1 <= t <= 2:
        raise ValueError('range reduction requires 1 <= t <= 2')
    u = (t-1)/(t+1)
    total = sum((2*u**(2*j+1)/Fraction(2*j+1) for j in range(DEGREE+1)), Fraction(0))
    tail = 2*u**(2*DEGREE+3)/(Fraction(2*DEGREE+3)*(1-u*u))
    return total, total+tail


@lru_cache(maxsize=4096)
def log_enclosure(t):
    t = Fraction(t)
    if t < 1:
        raise ValueError('distance logarithm requires ratio at least one')
    exponent = max(0, t.numerator.bit_length()-t.denominator.bit_length())
    if t < 2**exponent:
        exponent -= 1
    reduced = t/Fraction(2**exponent)
    lo, hi = log_unit_interval(reduced)
    lo2, hi2 = log_unit_interval(Fraction(2))
    lo, hi = lo+exponent*lo2, hi+exponent*hi2
    # Outward rational rounding keeps the saved witnesses small.
    scale = 10**12
    return Fraction((lo*scale).__floor__(), scale), Fraction((hi*scale).__ceil__(), scale)


def float_distances(rows):
    cost = np.full((len(rows), len(rows)), np.inf)
    np.fill_diagonal(cost, 0.)
    for x, row in enumerate(rows):
        for y, probability in row.items():
            cost[x, y] = -math.log(float(probability))
    return all_pairs_shortest_path(cost)


def best_quad(d, patch, radius):
    quad = np.array(list(itertools.combinations(patch, 4)))
    a, b, c, e = quad.T
    six = np.stack([d[a, b], d[a, c], d[a, e], d[b, c], d[b, e], d[c, e]], axis=1)
    sq = six*six
    match = np.stack([sq[:, 0]+sq[:, 5], sq[:, 1]+sq[:, 4], sq[:, 2]+sq[:, 3]], axis=1)
    which = match.argmax(axis=1)
    gap = np.maximum(2*match.max(axis=1)-sq.sum(axis=1), 0)
    total = six.sum(axis=1)
    lower = gap/(np.sqrt(total*total+2*gap)+total)/radius
    choice = int(lower.argmax())
    order = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]][which[choice]]
    return float(lower[choice]), quad[choice][order].tolist()


def check_gap(bounds, delta):
    if delta < 0:
        raise ValueError('negative fit error')
    la, lb = bounds['01'][0], bounds['23'][0]
    if min(la, lb) < delta:
        raise ValueError('diagonal lower bounds do not permit squaring')
    gap = (la-delta)**2+(lb-delta)**2-sum((bounds[key][1]+delta)**2 for key in ['02', '03', '12', '13'])
    if gap <= 0:
        raise ValueError('strict Hilbert quadrilateral obstruction failed')
    return gap


def witness_data(rows, center, k, vertices, relative):
    center_products = maximum_products(rows, center)
    nearest = sorted((x for x in range(len(rows)) if x != center), key=lambda x: (-center_products[x], x))[:k]
    if len(set(vertices)) != 4 or not set(vertices) <= set([center]+nearest):
        raise ValueError('quadrilateral is outside the declared local patch')
    radius = log_enclosure(1/min(center_products[x] for x in nearest))
    eccentricity = log_enclosure(1/min(center_products))
    cache = {center: center_products}
    bounds = {}
    for i, j in itertools.combinations(range(4), 2):
        source, target = vertices[i], vertices[j]
        if source not in cache:
            cache[source] = maximum_products(rows, source)
        bounds[f'{i}{j}'] = log_enclosure(1/cache[source][target])
    relative = Fraction(relative)
    if relative <= 0 or radius[0] <= 0 or eccentricity[0] <= 0:
        raise ValueError('nonpositive relative obstruction or radius')
    delta = relative*radius[1]
    gap = check_gap(bounds, delta)
    return {'center': center, 'k': k, 'nearest_neighbors': nearest, 'vertices': vertices,
            'relative_error_lower_bound': str(relative),
            'distance_bounds': {key: [str(lo), str(hi)] for key, (lo, hi) in bounds.items()},
            'radius_bounds': [str(x) for x in radius],
            'radius_over_global_diameter_upper_bound': str(radius[1]/eccentricity[0]),
            'absolute_fit_error_excluded': str(delta), 'strict_rational_gap': str(gap)}


def generate_case(kind, level, side, canonical=False):
    size = side if kind == 'grid' else level
    P = grid_2d(side, .5) if kind == 'grid' else sierpinski(level, lazy=.5)[1]
    n = len(P)
    m = 128 if canonical else min(256, max(16, n//3))
    levels = CANONICAL_LEVELS if canonical else [m]
    labels = hierarchical_diffusion_partition(P, levels, 6)['labels_list'][-1].tolist()
    rows = exact_macro_rows(kind, size, labels)
    d = float_distances(rows)
    k = min(24, m-1)
    knn = metric_knn(d, k)
    candidates = []
    for center in range(m):
        radius = float(d[center, knn[center]].max())
        candidates.append(best_quad(d, [center]+knn[center].tolist(), radius)[0])
    # A witness is selected from the median of the complete finite search.
    # Only the selected witness is exact-certified, not this distribution.
    center = sorted(range(m), key=lambda x: (candidates[x], x))[m//2]
    products = maximum_products(rows, center)
    nearest = sorted((x for x in range(m) if x != center), key=lambda x: (-products[x], x))[:k]
    proposed, vertices = best_quad(d, [center]+nearest, -math.log(float(min(products[x] for x in nearest))))
    relative = Fraction(max(1, math.floor(proposed*1000)-1), 1000)
    witness = witness_data(rows, center, k, vertices, relative)
    return {'kind': kind, 'size': size, 'n': n, 'm': m, 'tau': 5, 'lazy': '1/2',
            'n_eigs': 6, 'seed': 0, 'prototypes': 'uniform', 'labels': labels,
            'lens_levels': levels, 'configuration_scope': 'canonical' if canonical else 'size_family',
            'floating_search': {'scope': 'all macro centers and all four-point subsets of each k-nearest patch; candidate statistics only',
                                'k': k, 'median_relative_lower_bound': float(np.median(candidates)),
                                'minimum_relative_lower_bound': float(min(candidates))},
            'witness': witness}


def verify_case(case):
    if (case['tau'] != 5 or case['lazy'] != '1/2' or case['prototypes'] != 'uniform'
            or case['n_eigs'] != 6 or case['seed'] != 0):
        raise ValueError('unsupported dynamics or prototypes')
    rows = exact_macro_rows(case['kind'], case['size'], case['labels'], case['tau'])
    if case['configuration_scope'] not in ['canonical', 'size_family']:
        raise ValueError('unknown configuration scope')
    expected_m = 128 if case['configuration_scope'] == 'canonical' else min(256, max(16, case['n']//3))
    expected_levels = CANONICAL_LEVELS if case['configuration_scope'] == 'canonical' else [expected_m]
    if (type(case['n']) is not int or type(case['m']) is not int
            or case['n'] != len(case['labels']) or case['m'] != len(rows)
            or case['m'] != expected_m or case['lens_levels'] != expected_levels):
        raise ValueError('incorrect state counts')
    witness = case['witness']
    if type(witness['center']) is not int or not 0 <= witness['center'] < len(rows):
        raise ValueError('invalid center')
    if type(witness['k']) is not int or witness['k'] != min(24, len(rows)-1):
        raise ValueError('incorrect patch size')
    if any(type(x) is not int or not 0 <= x < len(rows) for x in witness['vertices']):
        raise ValueError('invalid witness vertices')
    actual = witness_data(rows, witness['center'], witness['k'], witness['vertices'], witness['relative_error_lower_bound'])
    if witness != actual:
        raise ValueError('saved witness disagrees with reconstructed exact distances')


def verify(record):
    if record['log_series_degree'] != DEGREE or record['eta'] != str(ETA) or record['eps_edge'] != str(EPS_EDGE):
        raise ValueError('incorrect cost or logarithm protocol')
    expected = [(kind, side if kind == 'grid' else level, 'size_family') for kind, level, side in CASES]
    expected += [(kind, side if kind == 'grid' else level, 'canonical') for kind, level, side in CANONICAL_CASES]
    actual = [(case['kind'], case['size'], case['configuration_scope']) for case in record['cases']]
    if actual != expected:
        raise ValueError('missing, duplicated, or changed comparison cases')
    for case in record['cases']:
        verify_case(case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/local_nonembedding_review')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(json.loads(args.verify.read_text()))
        print('Exact macro probabilities, paths, locality and Hilbert obstructions verified.')
        return
    sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    cases = []
    configurations = [(kind, level, side, False) for kind, level, side in CASES]
    configurations += [(kind, level, side, True) for kind, level, side in CANONICAL_CASES]
    for kind, level, side, canonical in configurations:
        case = generate_case(kind, level, side, canonical=canonical)
        verify_case(case)
        cases.append(case)
        w = case['witness']
        print(case['configuration_scope'], kind, case['n'], 'relative error lower bound', w['relative_error_lower_bound'],
              'radius / diameter <=', float(Fraction(w['radius_over_global_diameter_upper_bound'])), flush=True)
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise AssertionError('sources changed during certificate generation')
    record = {'scope': 'Exact finite local nonembedding witnesses for supplied learned-lens labels; no asymptotic or fractality certificate.',
              'log_series_degree': DEGREE, 'eta': str(ETA), 'eps_edge': str(EPS_EDGE), 'cases': cases,
              'source_sha256': hashes}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'certificate.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
