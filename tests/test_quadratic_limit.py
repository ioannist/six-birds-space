"""Check the proof's kernel bridge and falsify forged matched-control data."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quadratic_limit', ROOT/'scripts/audit_quadratic_limit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_float_convolution_matches_exact_original_counts_and_no_aliasing_torus():
    for n, counts in audit.integer_counts(4, audit.ISOTROPIC):
        actual = audit.floating_distribution(n)
        expected = np.zeros_like(actual)
        for (x, y), value in counts.items():
            expected[x+n, y+n] = value/(8**n)
        assert np.array_equal(actual, expected)
    torus = audit.torus_rw_distribution(9, .5, 4)
    ix = np.arange(-4, 5) % 9
    assert np.array_equal(torus[ix[:, None], ix], expected)


def test_correlated_covariance_is_derived_from_the_actual_step_law():
    for n, counts in audit.integer_counts(4):
        mass = 16**n
        assert sum(x*v for (x, y), v in counts.items()) == 0
        assert sum(y*v for (x, y), v in counts.items()) == 0
        assert Fraction(sum(x*x*v for (x, y), v in counts.items()), mass) == Fraction(3*n, 8)
        assert Fraction(sum(x*y*v for (x, y), v in counts.items()), mass) == Fraction(n, 4)
    # Covariance cross terms destroy the supplied axis identity, not the
    # positive quadratic form or all weighted Pythagorean identities.
    covariance = np.array([[3/8, 1/4], [1/4, 3/8]])
    assert np.allclose(np.linalg.inv(covariance), [[24/5, -16/5], [-16/5, 24/5]])
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_uniform_formula_has_a_real_log_domain_and_decays():
    assert audit.isotropic_bounds(16, np.sqrt(2))['uniform_centered_cost_bound'] is None
    bounds = [audit.isotropic_bounds(n, np.sqrt(2)) for n in [512, 1024, 4096, 16384]]
    errors = [row['uniform_relative_probability_bound'] for row in bounds]
    assert all(a>b>0 for a, b in zip(errors, errors[1:]))
    assert bounds[-1]['uniform_normalized_sqrt_readout_bound'] < .11
    for bad_n in [True, 1, 2.5]:
        with pytest.raises(ValueError):
            audit.isotropic_bounds(bad_n, 1)


def test_exact_control_rejects_tampered_counts_bounds_and_covariance():
    record = json.loads((ROOT/'docs/notes/quadratic_limit_review_20261003/evidence.json').read_text())['matched_control']
    row = record['rows'][0]
    counts = next(counts for n, counts in audit.integer_counts(16) if n == 16)
    assert audit.control_row(16, counts) == row
    assert Fraction(row['exact_residual_interval'][1]) < -1
    # Verify only one stage in the test; the separate CLI receipt covers all
    # four stages without repeating the costly exact recurrence in every case.
    for change in ['counts', 'interval', 'covariance']:
        changed = deepcopy(record)
        if change == 'counts':
            changed['rows'][0]['points_and_counts'][0][2] += 1
        elif change == 'interval':
            changed['rows'][0]['exact_residual_interval'][1] = '0'
        else:
            changed['covariance'][0][1] = '0'
        with pytest.raises(ValueError):
            audit.verify_control(changed)
    changed = deepcopy(record)
    changed['rows'][0]['stage'] = 16.0
    with pytest.raises(ValueError, match='integers'):
        audit.verify_control(changed)
    with pytest.raises(ValueError, match='integer displacements'):
        next(audit.integer_counts(2, [(0.5, 0, 1)]))
