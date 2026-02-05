import numpy as np

from geo_sbt.metrics_tv import tv, tv_rows


def test_tv_sanity():
    p = np.array([0.25, 0.25, 0.5])
    q = np.array([0.1, 0.2, 0.7])
    assert tv(p, p) == 0.0
    assert np.isclose(tv(p, q), tv(q, p))
    assert 0.0 <= tv(p, q) <= 1.0


def test_tv_rows_sanity():
    P = np.array([[0.5, 0.5], [0.0, 1.0]])
    Q = np.array([[0.5, 0.5], [1.0, 0.0]])
    rows = tv_rows(P, Q)
    assert rows.shape == (2,)
    assert rows[0] == 0.0
    assert np.isclose(rows[1], 1.0)
