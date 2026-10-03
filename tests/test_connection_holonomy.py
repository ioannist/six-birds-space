"""Common-space operators distinguish true loop residue from edge noncommutation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('connection_audit', ROOT/'scripts/audit_connection_holonomy.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_commuting_edge_rotations_can_give_noncommuting_covariant_shifts():
    a, b, area = audit.sphere_rectangle(np.arcsin(1/4), np.arcsin(1/2), 1/4)
    assert np.allclose(a @ b, b @ a)
    operator = audit.commutator_matrix(a, b)
    assert np.linalg.norm(operator, 2) == pytest.approx(2*np.sin(1/32), abs=1e-14)
    rng = np.random.default_rng(19)
    section = rng.normal(size=(2, 2, 2))
    coefficient, holonomy, second = audit.route_data(a, b)
    by_formula = np.einsum('...ij,...j->...i', coefficient, np.roll(np.roll(section, -1, 0), -1, 1))
    assert np.allclose(operator @ section.ravel(), by_formula.ravel(), atol=1e-14)
    assert np.allclose(coefficient, (holonomy-np.eye(2)) @ second, atol=1e-14)


def test_nonzero_constant_transports_are_flat_and_pure_gauges_remain_flat():
    a = np.broadcast_to(audit.rotation(.7), (3, 3, 2, 2)).copy()
    b = np.broadcast_to(audit.rotation(-.4), a.shape).copy()
    assert np.linalg.norm(audit.commutator_matrix(a, b), 2) < 1e-14
    q = np.stack([audit.rotation(t) for t in np.linspace(-2, 2, 9)]).reshape(a.shape)
    q[1, 1] = q[1, 1] @ np.diag([1., -1.])
    ga, gb = audit.gauge_connection(q, a, b)
    assert np.linalg.norm(audit.commutator_matrix(ga, gb), 2) < 1e-14


def test_curved_connection_is_gauge_invariant_and_has_area_scaling():
    rng = np.random.default_rng(0)
    rows = [audit.check_rectangle(.3-h/2, .3+h/2, h, rng) for h in [.2, .1, .05]]
    assert rows[-1]['operator_commutator_norm'] < rows[0]['operator_commutator_norm']
    errors = [abs(1-r['normalized_by_coordinate_area_and_center_density']) for r in rows]
    assert errors[2] < errors[1] < errors[0]


def test_invalid_connection_and_principal_angle_domains_are_rejected():
    a = np.broadcast_to(np.eye(2), (2, 2, 2, 2)).copy()
    with pytest.raises(ValueError, match='orthogonal'):
        audit.route_data(2*a, a)
    a[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        audit.route_data(a, a)
    with pytest.raises(ValueError, match='below pi'):
        audit.check_rectangle(-1.2, 1.2, 3., np.random.default_rng(0))
