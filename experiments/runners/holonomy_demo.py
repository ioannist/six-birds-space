from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np

from geo_sbt.geometry.holonomy import (
    holonomy_angles_for_triangles,
    local_embeddings_from_metric,
    metric_knn,
    sample_triangles_from_knn,
)
from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel, macro_kernel
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition
from geo_sbt.lenses.prototypes import prototypes_uniform
from geo_sbt.packaging import make_C
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.knn import knn_points
from geo_sbt.substrates.sphere import sphere_points


def _require_keys(config: dict, keys: list[str]) -> None:
    for key in keys:
        if key not in config:
            raise ValueError(f"missing required config key: {key}")


def _macro_metric(P: np.ndarray, *, levels: list[int], n_eigs: int, seed: int, tau: int) -> np.ndarray:
    ladder = hierarchical_diffusion_partition(
        P,
        levels=levels,
        n_eigs=n_eigs,
        method="kmeans_merge",
        seed=seed,
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
    expand_hops: int,
    min_overlap: int,
    max_loops: int,
    seed: int,
) -> dict:
    knn_embed = metric_knn(d_cost, k=k_neigh)
    neighborhoods, coords_list = local_embeddings_from_metric(
        d_cost, knn_embed, dim=2, expand_hops=expand_hops
    )
    knn_loop = metric_knn(d_cost, k=k_loop)
    triangles = sample_triangles_from_knn(knn_loop, max_loops=max_loops, seed=seed)
    angles = holonomy_angles_for_triangles(
        triangles, neighborhoods, coords_list, min_overlap=min_overlap
    )
    return {
        "triangles_sampled": int(len(triangles)),
        "triangles_evaluated": int(angles.size),
        "mean_angle": float(np.mean(angles)) if angles.size else float("nan"),
        "median_angle": float(np.median(angles)) if angles.size else float("nan"),
        "angles": angles,
    }


def _save_plot(
    plane_angles: np.ndarray,
    sphere_angles: np.ndarray,
    *,
    plots_dir: Path,
    docs_fig_dir: Path | None,
    write_docs: bool,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib unavailable; plots skipped")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    if write_docs and docs_fig_dir is not None:
        docs_fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    if plane_angles.size:
        plt.hist(plane_angles, bins=30, alpha=0.6, label="plane")
    if sphere_angles.size:
        plt.hist(sphere_angles, bins=30, alpha=0.6, label="sphere")
    plt.xlabel("|holonomy angle|")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()

    plot_name = "holonomy_plane_vs_sphere.png"
    if write_docs and docs_fig_dir is not None:
        plt.savefig(docs_fig_dir / plot_name)
    plt.savefig(plots_dir / plot_name)
    plt.close()


def run_holonomy_demo(config: dict) -> Dict[str, float]:
    _require_keys(config, ["run_name", "runner", "seed", "output_root"])

    write_docs = bool(config.get("write_docs_artifacts", True))

    plane_cfg = config.get("plane", {})
    sphere_cfg = config.get("sphere", {})
    macro_cfg = config.get("macro", {})
    hol_cfg = config.get("holonomy", {})

    _require_keys(plane_cfg, ["n_side", "lazy"])
    _require_keys(sphere_cfg, ["n_points", "knn_k", "sigma"])
    _require_keys(macro_cfg, ["levels", "n_eigs", "seed", "tau"])
    _require_keys(
        hol_cfg,
        ["k_neigh", "k_loop", "expand_hops", "min_overlap", "max_loops", "seed_plane", "seed_sphere"],
    )

    levels = list(macro_cfg["levels"])
    if len(levels) != 1:
        raise ValueError("macro.levels must contain a single scale for holonomy_demo")

    n_side = int(plane_cfg["n_side"])
    lazy = float(plane_cfg["lazy"])

    n_points = int(sphere_cfg["n_points"])
    knn_k = int(sphere_cfg["knn_k"])
    sigma = float(sphere_cfg["sigma"])
    self_loop = float(sphere_cfg.get("self_loop", 1e-6))

    n_eigs = int(macro_cfg["n_eigs"])
    macro_seed = int(macro_cfg["seed"])
    tau = int(macro_cfg["tau"])

    k_neigh = int(hol_cfg["k_neigh"])
    k_loop = int(hol_cfg["k_loop"])
    expand_hops = int(hol_cfg["expand_hops"])
    min_overlap = int(hol_cfg["min_overlap"])
    max_loops = int(hol_cfg["max_loops"])
    seed_plane = int(hol_cfg["seed_plane"])
    seed_sphere = int(hol_cfg["seed_sphere"])

    P_plane = grid_2d(n_side=n_side, lazy=lazy)

    rng = np.random.default_rng(int(config.get("seed", 0)))
    points = sphere_points(n_points, rng=rng)
    P_sphere = knn_points(points, k=knn_k, sigma=sigma, self_loop=self_loop, symmetrize=True)

    d_plane = _macro_metric(P_plane, levels=levels, n_eigs=n_eigs, seed=macro_seed, tau=tau)
    d_sphere = _macro_metric(P_sphere, levels=levels, n_eigs=n_eigs, seed=macro_seed, tau=tau)

    plane_stats = _holonomy_stats(
        d_plane,
        k_neigh=k_neigh,
        k_loop=k_loop,
        expand_hops=expand_hops,
        min_overlap=min_overlap,
        max_loops=max_loops,
        seed=seed_plane,
    )
    sphere_stats = _holonomy_stats(
        d_sphere,
        k_neigh=k_neigh,
        k_loop=k_loop,
        expand_hops=expand_hops,
        min_overlap=min_overlap,
        max_loops=max_loops,
        seed=seed_sphere,
    )

    median_diff = float(sphere_stats["median_angle"] - plane_stats["median_angle"])
    median_ratio = float(
        sphere_stats["median_angle"] / max(plane_stats["median_angle"], 1e-12)
    )

    run_id = config.get("run_id", "unknown")
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "plane": {
            "n_side": n_side,
            "lazy": lazy,
            "stats": {
                "triangles_sampled": plane_stats["triangles_sampled"],
                "triangles_evaluated": plane_stats["triangles_evaluated"],
                "mean_angle": plane_stats["mean_angle"],
                "median_angle": plane_stats["median_angle"],
            },
        },
        "sphere": {
            "n_points": n_points,
            "knn_k": knn_k,
            "sigma": sigma,
            "self_loop": self_loop,
            "stats": {
                "triangles_sampled": sphere_stats["triangles_sampled"],
                "triangles_evaluated": sphere_stats["triangles_evaluated"],
                "mean_angle": sphere_stats["mean_angle"],
                "median_angle": sphere_stats["median_angle"],
            },
        },
        "separation": {
            "median_diff": median_diff,
            "median_ratio": median_ratio,
        },
    }

    artifacts_dir = Path(config["artifacts_dir"])
    plots_dir = Path(config["plots_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = artifacts_dir / "holonomy_demo_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    docs_fig_dir = Path("docs") / "notes" / "figures"
    docs_summary_path = Path("docs") / "notes" / "holonomy_demo_summary.json"
    if write_docs:
        docs_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with docs_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    _save_plot(
        plane_stats["angles"],
        sphere_stats["angles"],
        plots_dir=plots_dir,
        docs_fig_dir=docs_fig_dir if write_docs else None,
        write_docs=write_docs,
    )

    metrics = {
        "tau": float(tau),
        "plane_median": float(plane_stats["median_angle"]),
        "sphere_median": float(sphere_stats["median_angle"]),
        "median_diff": median_diff,
        "median_ratio": median_ratio,
        "plane_evaluated": float(plane_stats["triangles_evaluated"]),
        "sphere_evaluated": float(sphere_stats["triangles_evaluated"]),
        "plane_mean": float(plane_stats["mean_angle"]),
        "sphere_mean": float(sphere_stats["mean_angle"]),
    }
    return metrics
