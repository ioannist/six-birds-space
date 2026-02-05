import numpy as np

from geo_sbt.substrates.constraints import anisotropic_gate
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.knn import knn_points
from geo_sbt.substrates.sierpinski import sierpinski
from geo_sbt.substrates.sphere import sphere_points
from geo_sbt.substrates.validate import validate_kernel


def test_grid_kernel_valid():
    P = grid_2d(5, lazy=0.3)
    info = validate_kernel(P)
    assert info["max_abs_row_sum_err"] < 1e-9
    assert info["min_entry"] >= -1e-12
    assert info["connected"] is True


def test_sphere_points_norms():
    rng = np.random.default_rng(0)
    pts = sphere_points(50, rng)
    norms = np.linalg.norm(pts, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-9


def test_knn_kernel_valid():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(20, 2))
    P = knn_points(points, k=19, sigma=1.0)
    info = validate_kernel(P)
    assert info["max_abs_row_sum_err"] < 1e-9
    assert info["min_entry"] >= -1e-12
    assert info["connected"] is True


def test_sierpinski_graph_kernel():
    adj, P = sierpinski(level=2, lazy=0.2)
    for i, nbrs in adj.items():
        for j in nbrs:
            assert i in adj[j]
    info = validate_kernel(P)
    assert info["max_abs_row_sum_err"] < 1e-9
    assert info["min_entry"] >= -1e-12
    assert info["connected"] is True


def test_anisotropic_gate_changes_kernel():
    P = grid_2d(4, lazy=0.1)
    P2 = anisotropic_gate(P, direction="east", strength=1.0)
    info = validate_kernel(P2)
    assert info["max_abs_row_sum_err"] < 1e-9
    assert info["min_entry"] >= -1e-12
    assert info["connected"] is True
    assert np.max(np.abs(P2 - P)) > 0.0
