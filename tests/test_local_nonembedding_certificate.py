"""Adversarial tests of the exact lower-bound certificate, not an MDS fit."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from geo_sbt.geometry.metric import macro_kernel
from geo_sbt.lenses.prototypes import prototypes_uniform
from geo_sbt.packaging import make_C
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.sierpinski import sierpinski

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_certificate', ROOT / 'scripts/certify_local_nonembedding.py')
cert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cert)


@pytest.mark.parametrize('kind,size', [('grid', 3), ('gasket', 1)])
def test_exact_counts_represent_original_staged_macro_kernel(kind, size):
    P = grid_2d(size, .5) if kind == 'grid' else sierpinski(size, lazy=.5)[1]
    labels = (np.arange(len(P)) % 3).tolist()
    rows = cert.exact_macro_rows(kind, size, labels)
    C, U = make_C(np.array(labels)), prototypes_uniform(np.array(labels))
    K = macro_kernel(P, 5, C, U)
    W = (K+K.T)/2
    for x, row in enumerate(rows):
        for y in range(len(rows)):
            if x != y:
                assert float(row.get(y, 0)) == pytest.approx(W[x, y], abs=1e-14)


def test_exact_path_search_prefers_composed_protocol_and_rejects_invalid_edges():
    rows = [{1: Fraction(1, 2), 2: Fraction(1, 8)},
            {0: Fraction(1, 2), 2: Fraction(1, 2)},
            {0: Fraction(1, 8), 1: Fraction(1, 2)}]
    assert cert.maximum_products(rows, 0) == [Fraction(1), Fraction(1, 2), Fraction(1, 4)]
    with pytest.raises(ValueError, match='disconnected'):
        cert.maximum_products([{}, {}], 0)
    with pytest.raises(ValueError, match='weights'):
        cert.maximum_products([{1: Fraction(2)}, {0: Fraction(2)}], 0)
    with pytest.raises(ValueError, match='exact rationals'):
        cert.maximum_products([{1: .5}, {0: .5}], 0)


def test_logarithm_bounds_are_rational_and_enclose_known_inequalities():
    assert cert.log_enclosure(1) == (Fraction(0), Fraction(0))
    lo, hi = cert.log_enclosure(2)
    assert isinstance(lo, Fraction) and isinstance(hi, Fraction)
    assert Fraction(2, 3) < lo < hi < Fraction(7, 10)
    direct = cert.log_unit_interval(2)
    assert all(isinstance(x, Fraction) for x in direct)
    with pytest.raises(ValueError):
        cert.log_enclosure(Fraction(1, 2))


def test_hilbert_gap_distinguishes_l1_square_from_euclidean_control():
    l1 = {key: (Fraction(2), Fraction(2)) if key in ['01', '23']
          else (Fraction(1), Fraction(1)) for key in ['01', '02', '03', '12', '13', '23']}
    assert cert.check_gap(l1, Fraction(1, 5)) > 0
    with pytest.raises(ValueError, match='obstruction'):
        cert.check_gap(l1, Fraction(1, 4))
    # A true Euclidean square has diagonals sqrt(2) in [7/5,3/2].
    euclidean = deepcopy(l1)
    euclidean['01'] = euclidean['23'] = (Fraction(7, 5), Fraction(3, 2))
    with pytest.raises(ValueError, match='obstruction'):
        cert.check_gap(euclidean, Fraction(0))


def test_saved_witness_rejects_fabricated_bounds_and_locality():
    record = json.loads((ROOT / 'docs/notes/local_nonembedding_review_20261003/certificate.json').read_text())
    case = record['cases'][0]
    cert.verify_case(case)
    fabricated = deepcopy(case)
    fabricated['witness']['distance_bounds']['01'][0] = '1000'
    with pytest.raises(ValueError, match='reconstructed'):
        cert.verify_case(fabricated)
    duplicated = deepcopy(case)
    duplicated['witness']['vertices'][1] = duplicated['witness']['vertices'][0]
    with pytest.raises(ValueError, match='local patch'):
        cert.verify_case(duplicated)
    false_margin = deepcopy(case)
    false_margin['witness']['relative_error_lower_bound'] = '1'
    with pytest.raises(ValueError):
        cert.verify_case(false_margin)


def test_exact_integer_domain_rejects_truncation_and_overflow():
    with pytest.raises(ValueError, match='integers'):
        cert.exact_macro_rows('grid', 3, [0]*8+[.5])
    with pytest.raises(ValueError, match='int64'):
        cert.exact_macro_rows('grid', 3, [0]*9, tau=100)
