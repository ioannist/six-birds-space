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


def test_exact_flat_charts_invariant_under_independent_reflections():
    from geo_sbt.geometry.holonomy import holonomy_angles_for_triangles
    rng = np.random.default_rng(42)
    points = rng.normal(size=(12, 2))
    neighborhoods = [np.arange(9), np.arange(3, 12), np.array([0, 1, 2, 6, 7, 8, 9, 10, 11])]
    coords = [points[n].copy() for n in neighborhoods]
    for i, theta in enumerate([.3, -.7, 1.2]):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords[i] = coords[i] @ R + rng.normal(size=2)
    coords[1][:, 0] *= -1
    result = holonomy_angles_for_triangles([(0, 1, 2)], neighborhoods, coords, return_diagnostics=True)
    assert result['triangles_evaluated'] == 1
    assert result['orientation_reversing'] == 0
    assert result['angles'][0] < 1e-12


def test_curved_chart_residue_invariant_under_gauge_changes():
    from geo_sbt.geometry.holonomy import holonomy_angles_for_triangles
    rng = np.random.default_rng(41)
    neighborhoods = [np.arange(9), np.arange(3, 12), np.array([0, 1, 2, 6, 7, 8, 9, 10, 11])]
    points = rng.normal(size=(12, 2))
    coords = [points[n] + .1 * rng.normal(size=(len(n), 2)) for n in neighborhoods]
    before = holonomy_angles_for_triangles([(0, 1, 2)], neighborhoods, coords)
    coords[0][:, 0] *= -1
    coords[2][:, 1] *= -1
    after = holonomy_angles_for_triangles([(0, 1, 2)], neighborhoods, coords)
    assert len(before) == 1
    assert np.allclose(after, before, atol=1e-12)
