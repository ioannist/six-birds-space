"""Exact finite certificates for the canonical lazy isotropic displacement cost.

Integer recurrence weights are 4 for a stay and 1 for each cardinal move;
probability equals count / 8**tau. With N > 2*tau there is no torus aliasing.
Log bounds are verified by rational Taylor enclosures for exp; no floating
probability, fitted coefficient, or RMS statistic enters the certificate check.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

STAGES = [4, 8, 16, 32, 64, 128]


@lru_cache(maxsize=None)
def exp_enclosure(x: Fraction, degree: int = 80) -> tuple[Fraction, Fraction]:
    """S_n <= exp(x) <= S_n + t_(n+1)/(1-x/(n+2)), for 0<=x<n+2.

    All Taylor terms are nonnegative. Past term n+1, each successive
    ratio is at most x/(n+2); the geometric tail bounds the exponential
    remainder. Returned bounds use only exact rational arithmetic.
    """
    x = Fraction(x)
    if type(degree) is not int or x < 0 or degree < 0 or x >= degree + 2:
        raise ValueError('Taylor enclosure requires 0 <= x < degree+2')
    term = Fraction(1)
    total = term
    for k in range(1, degree+1):
        term *= x / k
        total += term
    next_term = term * x / (degree+1)
    upper = total + next_term / (1 - x / (degree+2))
    return total, upper


def walk_counts(max_stage: int):
    if type(max_stage) is not int or max_stage < 0:
        raise ValueError('stage must be a nonnegative integer')
    counts = {(0, 0): 1}
    for tau in range(1, max_stage+1):
        result = {}
        for (x, y), count in counts.items():
            for dx, dy, weight in [(0, 0, 4), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1)]:
                pair = (x+dx, y+dy)
                result[pair] = result.get(pair, 0) + weight * count
        counts = result
        if sum(counts.values()) != 8**tau:
            raise AssertionError('incorrect exact probability mass')
        yield tau, counts


def check_row(row: dict) -> None:
    tau = row['tau']
    D = row['D']
    delta = Fraction(row['uniform_cost_error'])
    if any(type(row[key]) is not int for key in ['tau', 'D', 'torus_N', 'origin_count', 'points']):
        raise ValueError('stage, domain, and counts must be integers')
    if tau <= 0 or D != math.isqrt(tau) or row['torus_N'] <= 2*tau or delta <= 0:
        raise ValueError('invalid stage, domain, or error bound')
    expected_points = {(x, y) for x in range(-D, D+1) for y in range(-D, D+1)}
    seen = set()
    origin = row['origin_count']
    for item in row['counts']:
        if len(item) != 3 or any(type(value) is not int for value in item):
            raise ValueError('witness coordinates and counts must be integers')
        x, y, count = item
        if (x, y) in seen or count <= 0 or count > origin:
            raise ValueError('duplicate, unreachable, or negative-centered-cost point')
        seen.add((x, y))
        ratio = Fraction(origin, count)
        target = Fraction(2 * (x*x + y*y), tau)
        # exp(target-delta) <= ratio <= exp(target+delta) implies
        # |log(ratio)-target| <= delta by monotonicity of log.
        if target-delta > 0:
            _, upper = exp_enclosure(target-delta)
            if upper > ratio:
                raise ValueError('lower logarithmic bound failed')
        lower, _ = exp_enclosure(target+delta)
        if ratio > lower:
            raise ValueError('upper logarithmic bound failed')
    if seen != expected_points:
        raise ValueError('certificate does not cover the entire declared window')


def generate() -> dict:
    rows = []
    for tau, counts in walk_counts(max(STAGES)):
        if tau not in STAGES:
            continue
        D = math.isqrt(tau)
        origin = counts[(0, 0)]
        data = [[x, y, counts[(x, y)]] for x in range(-D, D+1) for y in range(-D, D+1)]
        # Floating values propose a bound only. check_row certifies it using
        # integers and rational inequalities and rejects an incorrect proposal.
        proposed = max(abs(math.log(origin/count)-2*(x*x+y*y)/tau) for x, y, count in data)
        delta = Fraction(math.ceil(proposed*1000)+1, 1000)
        row = {'tau': tau, 'torus_N': 512, 'D': D, 'origin_count': origin,
               'coefficient': str(Fraction(2, tau)), 'uniform_cost_error': str(delta),
               'points': len(data), 'counts': data,
               'normalized_readout_error_squared_bound': str(delta/2)}
        check_row(row)
        rows.append(row)
        print(tau, 'points', len(data), 'uniform_cost_error', str(delta), flush=True)
    return {'scope': 'Finite uniform displacement-cost and square-root readout bounds; no asymptotic or path-metric claim.',
            'kernel': {'stay_weight': 4, 'cardinal_move_weight': 1, 'denominator': 8},
            'domain': 'all integer displacements |x|, |y| <= floor(sqrt(tau))',
            'exponential_taylor_degree': 80,
            'rows': rows,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def verify(record: dict) -> None:
    if record['kernel'] != {'stay_weight': 4, 'cardinal_move_weight': 1, 'denominator': 8}:
        raise ValueError('unsupported kernel')
    if record['exponential_taylor_degree'] != 80:
        raise ValueError('unsupported Taylor degree')
    rows = {row['tau']: row for row in record['rows']}
    if len(rows) != len(record['rows']) or sorted(rows) != STAGES:
        raise ValueError('missing or duplicate stages')
    # Reconstruct the exact recurrence to prevent a free-standing table of
    # arbitrary positive integers from masquerading as transition probabilities.
    for tau, counts in walk_counts(max(STAGES)):
        if tau not in rows:
            continue
        row = rows[tau]
        if row['origin_count'] != counts[(0, 0)] or row['points'] != len(row['counts']):
            raise ValueError('incorrect origin or sample count')
        if row['coefficient'] != str(Fraction(2, tau)):
            raise ValueError('theoretical coefficient has been changed')
        if row['normalized_readout_error_squared_bound'] != str(Fraction(row['uniform_cost_error'])/2):
            raise ValueError('readout normalization has been changed')
        for x, y, count in row['counts']:
            if counts.get((x, y)) != count:
                raise ValueError('a witness count disagrees with the kernel recurrence')
        check_row(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/quadratic_walk_certificate.json'))
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(json.loads(args.verify.read_text()))
        print('Exact recurrence, complete windows, and rational logarithmic bounds verified.')
    else:
        record = generate()
        verify(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
