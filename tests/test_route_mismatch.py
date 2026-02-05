import numpy as np

from geo_sbt.packaging import make_C
from geo_sbt.route_mismatch import (
    d_fro,
    d_tv_sup,
    rm_closure_two_step,
)


def test_rm_commuting_case():
    P = np.eye(4, dtype=np.float64)
    labels0 = np.array([0, 0, 1, 1], dtype=int)
    C0 = make_C(labels0, m=2)
    U0 = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])

    labels1 = np.array([0, 1, 2, 3], dtype=int)
    C1 = make_C(labels1, m=4)
    U1 = np.eye(4, dtype=np.float64)

    rm = rm_closure_two_step(P, 1, 1, C0, U0, C1, U1, distance="tv_sup")
    assert rm <= 1e-12


def test_rm_noncommuting_case():
    P = np.eye(4, dtype=np.float64)
    labels0 = np.array([0, 0, 1, 1], dtype=int)
    C0 = make_C(labels0, m=2)
    U0 = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])

    labels1 = np.array([0, 1, 0, 1], dtype=int)
    C1 = make_C(labels1, m=2)
    U1 = np.array([[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]])

    rm = rm_closure_two_step(P, 1, 1, C0, U0, C1, U1, distance="tv_sup")
    assert rm >= 0.1


def test_distances_symmetric_zero():
    A = np.array([[1.0, 0.0], [0.5, 0.5]])
    B = np.array([[0.5, 0.5], [0.0, 1.0]])
    assert d_tv_sup(A, A) == 0.0
    assert d_fro(A, A) == 0.0
    assert np.isclose(d_tv_sup(A, B), d_tv_sup(B, A))
    assert np.isclose(d_fro(A, B), d_fro(B, A))
