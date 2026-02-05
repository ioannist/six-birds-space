import numpy as np

from geo_sbt.packaging import (
    E_apply,
    E_matrix,
    Q_f,
    U_f,
    idempotence_defect_delta,
    make_C,
)


def test_idempotent_case_delta_zero():
    labels = np.array([0, 0, 1, 1], dtype=int)
    C = make_C(labels, m=2)
    U = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    P = np.eye(4, dtype=np.float64)
    delta = idempotence_defect_delta(P, tau=1, C=C, U=U)
    assert delta <= 1e-12


def test_Q_U_identity_on_macros():
    labels = np.array([0, 0, 1, 1], dtype=int)
    C = make_C(labels, m=2)
    U = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])

    for nu in [np.array([1.0, 0.0]), np.array([0.0, 1.0])]:
        mu = U_f(nu, U)
        nu_back = Q_f(mu, C)
        assert np.allclose(nu_back, nu, atol=1e-12)

    rng = np.random.default_rng(0)
    nu = rng.random(2)
    nu = nu / nu.sum()
    mu = U_f(nu, U)
    nu_back = Q_f(mu, C)
    assert np.allclose(nu_back, nu, atol=1e-12)


def test_E_maps_distributions():
    labels = np.array([0, 0, 1], dtype=int)
    C = make_C(labels, m=2)
    U = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    P = np.array(
        [
            [0.7, 0.3, 0.0],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(1)
    mu = rng.random(3)
    mu = mu / mu.sum()

    out = E_apply(mu, P, tau=2, C=C, U=U)
    assert np.isclose(out.sum(), 1.0, atol=1e-12)
    assert np.min(out) >= -1e-12

    E = E_matrix(P, tau=2, C=C, U=U)
    assert np.allclose(E.sum(axis=1), 1.0, atol=1e-12)
