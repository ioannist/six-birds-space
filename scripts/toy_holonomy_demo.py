from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geo_sbt.geometry.holonomy import (  # noqa: E402
    holonomy_angles_for_triangles,
    local_embeddings_from_metric,
    metric_knn,
    sample_triangles_from_knn,
)
from geo_sbt.geometry.metric import (  # noqa: E402
    all_pairs_shortest_path,
    cost_matrix_from_kernel,
    macro_kernel,
)
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition  # noqa: E402
from geo_sbt.lenses.prototypes import prototypes_uniform  # noqa: E402
from geo_sbt.packaging import make_C  # noqa: E402
from geo_sbt.substrates.grid import grid_2d  # noqa: E402
from geo_sbt.substrates.knn import knn_points  # noqa: E402
from geo_sbt.substrates.sphere import sphere_points  # noqa: E402


def _macro_metric(P: np.ndarray, *, tau: int) -> np.ndarray:
    ladder = hierarchical_diffusion_partition(
        P,
        levels=[128],
        n_eigs=6,
        method="kmeans_merge",
        seed=0,
    )
    labels = ladder["labels_list"][0]
    C = make_C(labels)
    U = prototypes_uniform(labels)
    P_hat = macro_kernel(P, tau, C, U)
    cost = cost_matrix_from_kernel(P_hat, symmetrize="weight_avg", eps_edge=1e-15)
    d_cost = all_pairs_shortest_path(cost)
    return d_cost


def _holonomy_stats(
    d_cost: np.ndarray,
    *,
    k_neigh: int,
    k_loop: int,
    seed: int = 0,
    expand_hops: int = 1,
) -> dict:
    knn_embed = metric_knn(d_cost, k=k_neigh)
    neighborhoods, coords_list = local_embeddings_from_metric(
        d_cost, knn_embed, dim=2, expand_hops=expand_hops
    )

    knn_loop = metric_knn(d_cost, k=k_loop)
    triangles = sample_triangles_from_knn(knn_loop, max_loops=800, seed=seed)
    angles = holonomy_angles_for_triangles(triangles, neighborhoods, coords_list, min_overlap=4)

    mean_angle = float(np.mean(angles)) if angles.size else float("nan")
    median_angle = float(np.median(angles)) if angles.size else float("nan")
    return {
        "triangles_sampled": len(triangles),
        "triangles_evaluated": int(angles.size),
        "mean_angle": mean_angle,
        "median_angle": median_angle,
    }


def _run_attempt(k_neigh: int) -> dict:
    tau = 5
    k_loop = 8
    expand_hops = 1

    P_plane = grid_2d(n_side=25, lazy=0.5)
    d_plane = _macro_metric(P_plane, tau=tau)
    plane_stats = _holonomy_stats(
        d_plane, k_neigh=k_neigh, k_loop=k_loop, seed=0, expand_hops=expand_hops
    )

    rng = np.random.default_rng(0)
    points = sphere_points(n=500, rng=rng)
    P_sphere = knn_points(points, k=10, sigma=0.5, symmetrize=True)
    d_sphere = _macro_metric(P_sphere, tau=tau)
    sphere_stats = _holonomy_stats(
        d_sphere, k_neigh=k_neigh, k_loop=k_loop, seed=1, expand_hops=expand_hops
    )

    return {
        "tau": tau,
        "levels": [128],
        "k_neigh": k_neigh,
        "k_loop": k_loop,
        "expand_hops": expand_hops,
        "plane": plane_stats,
        "sphere": sphere_stats,
        "n_micro_plane": int(P_plane.shape[0]),
        "n_micro_sphere": int(P_sphere.shape[0]),
        "m_macro": 128,
    }


def _meets_criteria(stats: dict) -> bool:
    plane = stats["plane"]
    sphere = stats["sphere"]
    if plane["triangles_evaluated"] < 50 or sphere["triangles_evaluated"] < 50:
        return False
    median_plane = plane["median_angle"]
    median_sphere = sphere["median_angle"]
    if not np.isfinite(median_plane) or not np.isfinite(median_sphere):
        return False
    if median_plane > 0.08:
        return False
    if not (median_sphere >= median_plane + 0.02 or median_sphere >= 1.5 * median_plane):
        return False
    return True


def _print_stats(label: str, stats: dict) -> None:
    plane = stats["plane"]
    sphere = stats["sphere"]
    print(
        f"{label} plane | sampled={plane['triangles_sampled']} | evaluated={plane['triangles_evaluated']} "
        f"| mean={plane['mean_angle']:.4f} | median={plane['median_angle']:.4f}"
    )
    print(
        f"{label} sphere | sampled={sphere['triangles_sampled']} | evaluated={sphere['triangles_evaluated']} "
        f"| mean={sphere['mean_angle']:.4f} | median={sphere['median_angle']:.4f}"
    )


def _plot_histogram(plane_angles: np.ndarray, sphere_angles: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; plot skipped")
        return False

    fig_dir = ROOT / "docs" / "notes" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "holonomy_plane_vs_sphere.png"

    plt.figure(figsize=(6, 4))
    plt.hist(plane_angles, bins=30, alpha=0.6, label="plane", density=True)
    plt.hist(sphere_angles, bins=30, alpha=0.6, label="sphere", density=True)
    plt.xlabel("|holonomy angle|")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return True


def main() -> None:
    attempt1 = _run_attempt(k_neigh=18)
    _print_stats("attempt1", attempt1)

    if not _meets_criteria(attempt1):
        attempt2 = _run_attempt(k_neigh=24)
        _print_stats("attempt2", attempt2)
        stats = attempt2
    else:
        stats = attempt1

    summary_path = ROOT / "docs" / "notes" / "holonomy_demo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as f:
        payload = {"attempts": [attempt1], "selected": stats}
        if "attempt2" in locals():
            payload["attempts"].append(attempt2)
        json.dump(payload, f, indent=2, sort_keys=True)

    # For plotting, recompute angles from chosen stats inputs.
    k_neigh = stats["k_neigh"]
    k_loop = stats["k_loop"]
    tau = stats["tau"]

    P_plane = grid_2d(n_side=25, lazy=0.5)
    d_plane = _macro_metric(P_plane, tau=tau)
    plane_knn = metric_knn(d_plane, k=k_neigh)
    plane_neigh, plane_coords = local_embeddings_from_metric(
        d_plane, plane_knn, dim=2, expand_hops=stats["expand_hops"]
    )
    plane_tri = sample_triangles_from_knn(metric_knn(d_plane, k=k_loop), max_loops=800, seed=0)
    plane_angles = holonomy_angles_for_triangles(plane_tri, plane_neigh, plane_coords, min_overlap=4)

    rng = np.random.default_rng(0)
    points = sphere_points(n=500, rng=rng)
    P_sphere = knn_points(points, k=10, sigma=0.5, symmetrize=True)
    d_sphere = _macro_metric(P_sphere, tau=tau)
    sphere_knn = metric_knn(d_sphere, k=k_neigh)
    sphere_neigh, sphere_coords = local_embeddings_from_metric(
        d_sphere, sphere_knn, dim=2, expand_hops=stats["expand_hops"]
    )
    sphere_tri = sample_triangles_from_knn(metric_knn(d_sphere, k=k_loop), max_loops=800, seed=1)
    sphere_angles = holonomy_angles_for_triangles(sphere_tri, sphere_neigh, sphere_coords, min_overlap=4)

    _plot_histogram(plane_angles, sphere_angles)


if __name__ == "__main__":
    main()
