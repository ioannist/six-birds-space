"""Adversarial checks for the exact finite cost certificate."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quadratic_certificate', ROOT / 'scripts/certify_quadratic_walk.py')
cert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cert)


def record():
    return json.loads((ROOT / 'docs/notes/refinement_review_20261003/quadratic_walk_certificate.json').read_text())


def test_integer_walk_moments_determine_the_coefficient():
    for tau, counts in cert.walk_counts(4):
        mass = 8**tau
        assert sum(counts.values()) == mass
        assert sum(x * n for (x, y), n in counts.items()) == 0
        assert sum(x * y * n for (x, y), n in counts.items()) == 0
        assert sum(x*x * n for (x, y), n in counts.items()) == tau * mass // 4
        assert sum(y*y * n for (x, y), n in counts.items()) == tau * mass // 4
        assert all(n == counts[(-x, y)] == counts[(y, x)] for (x, y), n in counts.items())
        if tau == 2:
            assert counts[(0, 0)] == 20
            assert counts[(1, 1)] == 2
            assert (3, 0) not in counts


def test_taylor_enclosures_use_exact_rationals():
    assert cert.exp_enclosure(0) == (Fraction(1), Fraction(1))
    assert cert.exp_enclosure(1, degree=0) == (Fraction(1), Fraction(3))
    assert cert.exp_enclosure(Fraction(1), degree=2) == (Fraction(5, 2), Fraction(49, 18))
    for x, degree in [(Fraction(-1), 80), (Fraction(2), 0), (Fraction(1), -1)]:
        with pytest.raises(ValueError):
            cert.exp_enclosure(x, degree)


def test_certificate_rejects_missing_domain_and_false_uniform_bound():
    row = record()['rows'][0]
    cert.check_row(row)
    incomplete = deepcopy(row)
    incomplete['counts'].pop()
    with pytest.raises(ValueError, match='entire declared window'):
        cert.check_row(incomplete)
    false_bound = deepcopy(row)
    false_bound['uniform_cost_error'] = '1/1000'
    with pytest.raises(ValueError, match='logarithmic bound'):
        cert.check_row(false_bound)
    fractional_count = deepcopy(row)
    fractional_count['counts'][0][2] += .5
    with pytest.raises(ValueError, match='must be integers'):
        cert.check_row(fractional_count)


def test_certificate_rejects_fabricated_probabilities_and_normalization():
    fabricated = record()
    fabricated['rows'][0]['counts'][0][2] += 1
    with pytest.raises(ValueError, match='kernel recurrence'):
        cert.verify(fabricated)
    wrong_coefficient = record()
    wrong_coefficient['rows'][0]['coefficient'] = '1/4'
    with pytest.raises(ValueError, match='coefficient'):
        cert.verify(wrong_coefficient)
    wrong_readout = record()
    wrong_readout['rows'][0]['normalized_readout_error_squared_bound'] = '0'
    with pytest.raises(ValueError, match='normalization'):
        cert.verify(wrong_readout)
