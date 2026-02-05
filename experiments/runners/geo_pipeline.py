from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from geo_sbt.geometry.holonomy import (
    holonomy_angles_for_triangles,
    local_embeddings_from_metric,
    metric_knn,
    sample_triangles_from_knn,
)
from geo_sbt.geometry.metric import (
    all_pairs_shortest_path,
    cost_matrix_from_kernel,
    distortion_between_scales,
    macro_kernel,
)
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition
from geo_sbt.lenses.prototypes import (
    prototypes_stationary_conditional,
    prototypes_uniform,
)
from geo_sbt.lenses.stationary import stationary_distribution
from geo_sbt.packaging import idempotence_defect_delta, make_C, prototype_stabilities
from geo_sbt.substrates.constraints import anisotropic_gate
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.knn import knn_points
from geo_sbt.substrates.sierpinski import sierpinski
from geo_sbt.substrates.sphere import sphere_points


def _mean_finite_offdiag(d: np.ndarray) -> float:
    n = d.shape[0]
    mask = np.isfinite(d) & ~np.eye(n, dtype=bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(d[mask]))


def _build_micro_kernel(config: dict) -> tuple[np.ndarray, dict]:
    substrate = config["substrate"]
    kind = substrate["kind"]
    if kind == "grid":
        n_side = int(substrate["n_side"])
        lazy = float(substrate.get("lazy", 0.5))
        P = grid_2d(n_side=n_side, lazy=lazy)
        info = {"kind": kind, "n_side": n_side, "lazy": lazy}
        return P, info
    if kind == "sphere_knn":
        n_points = int(substrate["n_points"])
        k = int(substrate["k"])
        sigma = float(substrate["sigma"])
        self_loop = float(substrate.get("self_loop", 1e-6))
        rng = np.random.default_rng(int(config.get("seed", 0)))
        points = sphere_points(n_points, rng)
        P = knn_points(points, k=k, sigma=sigma, self_loop=self_loop, symmetrize=True)
        info = {"kind": kind, "n_points": n_points, "k": k, "sigma": sigma, "self_loop": self_loop}
        return P, info
    if kind == "sierpinski":
        level = int(substrate["level"])
        lazy = float(substrate.get("lazy", 0.5))
        _, P = sierpinski(level=level, lazy=lazy)
        info = {"kind": kind, "level": level, "lazy": lazy}
        return P, info
    raise ValueError(f"unknown substrate kind: {kind}")


def _maybe_apply_constraints(P: np.ndarray, config: dict) -> tuple[np.ndarray, dict | None]:
    constraints = config.get("constraints")
    if not constraints:
        return P, None
    if constraints.get("kind") != "anisotropic_gate":
        raise ValueError("only anisotropic_gate constraint is supported")
    direction = constraints["direction"]
    strength = float(constraints["strength"])
    P2 = anisotropic_gate(P, direction=direction, strength=strength)
    return P2, {"kind": "anisotropic_gate", "direction": direction, "strength": strength}


def _make_prototypes(labels: np.ndarray, m: int, proto_kind: str, pi: np.ndarray | None) -> np.ndarray:
    if proto_kind == "uniform":
        return prototypes_uniform(labels, m)
    if proto_kind == "stationary_conditional":
        if pi is None:
            raise ValueError("stationary distribution required for stationary_conditional prototypes")
        return prototypes_stationary_conditional(labels, pi, m)
    raise ValueError(f"unknown prototypes: {proto_kind}")


def _plots_enabled(config: dict) -> bool:
    plots_cfg = config.get("plots", {})
    if isinstance(plots_cfg, dict):
        return bool(plots_cfg.get("enabled", True))
    return True


def run_geo_pipeline(config: dict) -> Dict[str, float]:
    """Run the end-to-end geometry pipeline."""
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)

    P, substrate_info = _build_micro_kernel(config)
    P, constraint_info = _maybe_apply_constraints(P, config)
    n_micro = int(P.shape[0])

    lens = config["lens"]
    levels = lens["levels"]
    n_eigs = int(lens["n_eigs"])
    lens_seed = int(lens.get("seed", seed))
    ladder = hierarchical_diffusion_partition(P, levels=levels, n_eigs=n_eigs, seed=lens_seed)
    labels_list = ladder["labels_list"]
    refine_maps = ladder["refine_maps"]
    cluster_counts = ladder["cluster_counts"]

    tau = int(config["tau"])
    proto_kind = config.get("prototypes", "uniform")

    pi = None
    if proto_kind == "stationary_conditional":
        pi = stationary_distribution(P)

    metric_cfg = config.get("metric", {})
    symmetrize = metric_cfg.get("symmetrize", "weight_avg")
    eps_edge = float(metric_cfg.get("eps_edge", 1e-15))
    eta = float(metric_cfg.get("eta", 1e-12))

    per_level = []
    dists = []
    deltas = []

    for labels, m in zip(labels_list, cluster_counts):
        C = make_C(labels, m)
        U = _make_prototypes(labels, m, proto_kind, pi)

        delta = idempotence_defect_delta(P, tau, C, U)
        stabs = prototype_stabilities(P, tau, C, U)
        stab_mean = float(np.mean(stabs))
        stab_max = float(np.max(stabs))

        P_hat = macro_kernel(P, tau, C, U)
        cost = cost_matrix_from_kernel(P_hat, symmetrize=symmetrize, eps_edge=eps_edge, eta=eta)
        d = all_pairs_shortest_path(cost)
        dists.append(d)
        mean_dist = _mean_finite_offdiag(d)
        inf_count = int(np.sum(~np.isfinite(d)))

        per_level.append(
            {
                "m": int(m),
                "delta": float(delta),
                "stab_mean": stab_mean,
                "stab_max": stab_max,
                "mean_finite_dist": mean_dist,
                "inf_count": inf_count,
            }
        )
        deltas.append(float(delta))

    distortions = []
    distortion_cfg = config.get("distortion", {})
    rescale_mode = distortion_cfg.get("rescale", "lstsq")
    max_raw = 0.0
    max_rescaled = 0.0
    last_rescaled = 0.0

    for i, r in enumerate(refine_maps):
        d_coarse = dists[i]
        d_fine = dists[i + 1]
        raw = distortion_between_scales(d_fine, d_coarse, r, rescale=None)
        rescaled = distortion_between_scales(d_fine, d_coarse, r, rescale=rescale_mode)
        max_raw = max(max_raw, raw["max_abs_diff"])
        max_rescaled = max(max_rescaled, rescaled["max_abs_diff"])
        last_rescaled = rescaled["max_abs_diff"]
        distortions.append(
            {
                "coarse_m": int(cluster_counts[i]),
                "fine_m": int(cluster_counts[i + 1]),
                "raw": float(raw["max_abs_diff"]),
                "rescaled": float(rescaled["max_abs_diff"]),
                "alpha": float(rescaled["alpha"]),
            }
        )

    holonomy_stats = None
    holonomy_angles = None
    hol_cfg = config.get("holonomy", {})
    if hol_cfg.get("enabled", False):
        level_m = hol_cfg.get("level_m")
        level_index = hol_cfg.get("level_index")
        if level_m is not None:
            if level_m not in cluster_counts:
                raise ValueError("holonomy level_m not found in cluster counts")
            idx = cluster_counts.index(level_m)
        elif level_index is not None:
            idx = int(level_index)
        else:
            idx = len(cluster_counts) - 1

        d = dists[idx]
        k_neigh = int(hol_cfg.get("k_neigh", 18))
        k_loop = int(hol_cfg.get("k_loop", 8))
        max_loops = int(hol_cfg.get("max_loops", 500))
        min_overlap = int(hol_cfg.get("min_overlap", 4))
        expand_hops = int(hol_cfg.get("expand_hops", 0))

        knn_embed = metric_knn(d, k=k_neigh)
        neighborhoods, coords_list = local_embeddings_from_metric(
            d, knn_embed, dim=2, expand_hops=expand_hops
        )
        knn_loop = metric_knn(d, k=k_loop)
        triangles = sample_triangles_from_knn(knn_loop, max_loops=max_loops, seed=seed)
        angles = holonomy_angles_for_triangles(
            triangles, neighborhoods, coords_list, min_overlap=min_overlap
        )
        holonomy_angles = angles

        holonomy_stats = {
            "level_m": int(cluster_counts[idx]),
            "triangles_sampled": int(len(triangles)),
            "triangles_evaluated": int(angles.size),
            "mean_angle": float(np.mean(angles)) if angles.size else float("nan"),
            "median_angle": float(np.median(angles)) if angles.size else float("nan"),
        }

    summary = {
        "substrate": substrate_info,
        "constraints": constraint_info,
        "n_micro": n_micro,
        "lens": {
            "levels": cluster_counts,
            "n_eigs": n_eigs,
            "seed": lens_seed,
        },
        "tau": tau,
        "prototypes": proto_kind,
        "metric": {
            "symmetrize": symmetrize,
            "eps_edge": eps_edge,
            "eta": eta,
        },
        "per_level": per_level,
        "distortions": distortions,
        "holonomy": holonomy_stats,
    }

    artifacts_dir = Path(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts_dir / "geo_pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if _plots_enabled(config):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            print("matplotlib unavailable; plots skipped")
        else:
            plots_dir = Path(config["plots_dir"])
            plots_dir.mkdir(parents=True, exist_ok=True)

            ms = [row["m"] for row in per_level]
            deltas_plot = [row["delta"] for row in per_level]
            mean_dists = [row["mean_finite_dist"] for row in per_level]
            dist_rescaled = [row["rescaled"] for row in distortions]
            dist_x = [row["fine_m"] for row in distortions]

            plt.figure(figsize=(5, 4))
            plt.plot(ms, deltas_plot, marker="o")
            plt.xlabel("m")
            plt.ylabel("delta")
            plt.tight_layout()
            plt.savefig(plots_dir / "delta_vs_level.png")
            plt.close()

            if dist_x:
                plt.figure(figsize=(5, 4))
                plt.plot(dist_x, dist_rescaled, marker="o")
                plt.xlabel("fine_m")
                plt.ylabel("distortion_rescaled")
                plt.tight_layout()
                plt.savefig(plots_dir / "distortion_vs_level.png")
                plt.close()

            plt.figure(figsize=(5, 4))
            plt.plot(ms, mean_dists, marker="o")
            plt.xlabel("m")
            plt.ylabel("mean_finite_dist")
            plt.tight_layout()
            plt.savefig(plots_dir / "mean_distance_vs_level.png")
            plt.close()

            if holonomy_angles is not None and holonomy_angles.size > 0:
                plt.figure(figsize=(5, 4))
                plt.hist(holonomy_angles, bins=30)
                plt.xlabel("|holonomy angle|")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(plots_dir / "holonomy_hist.png")
                plt.close()

    metrics = {
        "n_micro": float(n_micro),
        "m_finest": float(cluster_counts[-1]),
        "tau": float(tau),
        "delta_finest": float(per_level[-1]["delta"]),
        "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "stab_mean_finest": float(per_level[-1]["stab_mean"]),
        "stab_max_finest": float(per_level[-1]["stab_max"]),
        "mean_dist_finest": float(per_level[-1]["mean_finite_dist"]),
        "inf_count_finest": float(per_level[-1]["inf_count"]),
        "distortion_max_raw": float(max_raw),
        "distortion_max_rescaled": float(max_rescaled),
        "distortion_last_rescaled": float(last_rescaled),
    }

    if holonomy_stats is not None:
        metrics.update(
            {
                "holonomy_median": float(holonomy_stats["median_angle"]),
                "holonomy_mean": float(holonomy_stats["mean_angle"]),
                "holonomy_evaluated": float(holonomy_stats["triangles_evaluated"]),
            }
        )

    return metrics
