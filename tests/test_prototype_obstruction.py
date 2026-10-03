"""Test universal prototype optimization against the actual lifted TV defect."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from geo_sbt.packaging import make_C, prototype_stabilities
from geo_sbt.substrates.grid import grid_2d

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prototype_obstruction', ROOT/'scripts/audit_prototype_obstruction.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_exact_return_bounds_match_original_kernel_and_point_prototypes_attain_them():
    labels = [z % 3 for z in range(9)]
    counts, denominator = audit.staged_counts('grid', 3, labels)
    P = grid_2d(3, .5)
    C = make_C(np.array(labels))
    assert np.allclose(counts/denominator, np.linalg.matrix_power(P, 5)@C, atol=1e-14)
    result = audit.analyze('grid', 3, labels)
    U = np.zeros((3, 9))
    U[np.arange(3), result['optimal_point_prototype_microstates']] = 1
    actual = prototype_stabilities(P, 5, C, U)
    assert np.allclose(actual, [float(Fraction(x)) for x in result['optimal_escape_by_fiber']], atol=1e-14)
    # A fully supported random alternative cannot evade the same lower bound.
    rng = np.random.default_rng(3)
    for x in range(3):
        fiber = np.flatnonzero(np.array(labels) == x)
        weights = rng.uniform(size=len(fiber))
        U[x, fiber] = weights/weights.sum()
    assert np.all(prototype_stabilities(P, 5, C, U) >= actual-1e-14)


def test_exact_canonical_certificate_rejects_fabricated_improvement():
    saved = json.loads((ROOT/'docs/notes/prototype_obstruction_review_20261003/evidence.json').read_text())
    audit.verify(saved)
    changed = deepcopy(saved)
    changed['cases'][0]['optimal_worst_prototype_defect'] = '1/100'
    with pytest.raises(ValueError, match='reconstruction'):
        audit.verify(changed)
    changed = deepcopy(saved)
    changed['cases'][1]['worst_fiber_return_counts'][0] += 1
    with pytest.raises(ValueError, match='reconstruction'):
        audit.verify(changed)
    changed = deepcopy(saved)
    changed['cases'][0]['tau'] = 5.0
    with pytest.raises(ValueError, match='integers'):
        audit.verify(changed)


def test_exact_domain_rejects_fractional_labels_and_overflow():
    with pytest.raises(ValueError, match='integer'):
        audit.staged_counts('grid', 3, [0]*8+[.5])
    with pytest.raises(ValueError, match='range'):
        audit.staged_counts('grid', 3, [0]*9, tau=100)
