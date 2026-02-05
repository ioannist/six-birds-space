import numpy as np

from geo_sbt.geometry.holonomy import classical_mds, procrustes_rotation, rotation_angle


def test_classical_mds_square():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    coords = classical_mds(D, dim=2)
    assert coords.shape == (4, 2)
    assert np.isfinite(coords).all()


def test_procrustes_rotation_proper():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(10, 2))
    theta = 0.4
    R_true = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    B = A @ R_true
    R = procrustes_rotation(A, B, enforce_proper=True)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    assert np.allclose(R, R_true, atol=1e-5)


def test_rotation_angle_identity():
    H = np.eye(2)
    assert rotation_angle(H) == 0.0
