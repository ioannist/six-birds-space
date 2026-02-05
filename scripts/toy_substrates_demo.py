from __future__ import annotations

import numpy as np

from geo_sbt.substrates.constraints import anisotropic_gate
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.knn import knn_points
from geo_sbt.substrates.sierpinski import sierpinski
from geo_sbt.substrates.sphere import sphere_points
from geo_sbt.substrates.validate import validate_kernel


def _print_summary(name: str, info: dict, extras: str) -> None:
    print(
        f"{name} | n={info['n']} | nnz={info['nnz']} | density={info['density']:.4f} "
        f"| max_row_err={info['max_abs_row_sum_err']:.2e} | min_entry={info['min_entry']:.2e} "
        f"| connected={info['connected']} | {extras}"
    )


def main() -> None:
    rng = np.random.default_rng(0)

    grid = grid_2d(n_side=7, lazy=0.5)
    info_grid = validate_kernel(grid)
    _print_summary("grid", info_grid, "lazy=0.5")

    points = sphere_points(n=40, rng=rng)
    P_sphere = knn_points(points, k=8, sigma=0.5)
    info_sphere = validate_kernel(P_sphere)
    _print_summary("sphere_knn", info_sphere, "n=40 k=8 sigma=0.5")

    adj_sier, P_sier = sierpinski(level=3, lazy=0.5)
    info_sier = validate_kernel(P_sier)
    _print_summary("sierpinski", info_sier, "level=3 lazy=0.5")

    P_aniso = anisotropic_gate(grid, direction="east", strength=1.0)
    info_aniso = validate_kernel(P_aniso)
    _print_summary("anisotropic", info_aniso, "dir=east strength=1.0")

    points2 = rng.normal(size=(50, 2))
    P_knn2 = knn_points(points2, k=8, sigma=0.6)
    info_knn2 = validate_kernel(P_knn2)
    _print_summary("knn2d", info_knn2, "n=50 k=8 sigma=0.6")


if __name__ == "__main__":
    main()
