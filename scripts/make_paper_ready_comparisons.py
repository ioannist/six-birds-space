from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "notes"
RUNS = DOCS / "runs"
FIGS = DOCS / "figures"
TABLES = DOCS / "tables"

RUN_PACKS = {
    "grid": "grid_plane_20260202T193540Z_4b14",
    "sphere": "sphere_knn_20260202T193547Z_7681",
    "sierpinski": "sierpinski_20260202T193551Z_95ab",
    "anisotropic": "anisotropic_20260202T193556Z_310f",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _geo_summary(run_id: str) -> dict:
    path = RUNS / run_id / "artifacts" / "geo_pipeline_summary.json"
    return _load_json(path)


def _geo_metrics(run_id: str) -> dict:
    path = RUNS / run_id / "metrics.json"
    return _load_json(path)


def _per_level_arrays(summary: dict) -> dict:
    per_level = summary.get("per_level", [])
    m = [int(row.get("m")) for row in per_level]
    delta = [float(row.get("delta")) for row in per_level]
    stab_mean = [float(row.get("stab_mean")) for row in per_level]
    stab_max = [float(row.get("stab_max")) for row in per_level]
    mean_dist = [float(row.get("mean_finite_dist")) for row in per_level]
    inf_count = [float(row.get("inf_count")) for row in per_level]
    return {
        "m": m,
        "delta": delta,
        "stab_mean": stab_mean,
        "stab_max": stab_max,
        "mean_finite_dist": mean_dist,
        "inf_count": inf_count,
    }


def _distortion_arrays(summary: dict) -> dict:
    distortions = summary.get("distortions", [])
    fine_m = [int(row.get("fine_m")) for row in distortions]
    rescaled = [float(row.get("rescaled")) for row in distortions]
    alpha = [float(row.get("alpha")) for row in distortions]
    return {"fine_m": fine_m, "rescaled": rescaled, "alpha": alpha}


def _plot_compare_delta(data: dict, out_path: Path) -> None:
    plt.figure()
    for label, arrays in data.items():
        plt.plot(arrays["m"], arrays["delta"], marker="o", label=label)
    plt.xscale("log")
    plt.xlabel("m")
    plt.ylabel("delta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_compare_distortion(data: dict, out_path: Path) -> None:
    plt.figure()
    for label, arrays in data.items():
        plt.plot(arrays["fine_m"], arrays["rescaled"], marker="o", label=label)
    plt.xscale("log")
    plt.xlabel("fine_m")
    plt.ylabel("distortion_rescaled")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_compare_mean_distance(data: dict, out_path: Path) -> None:
    plt.figure()
    for label, arrays in data.items():
        plt.plot(arrays["m"], arrays["mean_finite_dist"], marker="o", label=label)
    plt.xscale("log")
    plt.xlabel("m")
    plt.ylabel("mean_finite_dist")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _csv_row(row: dict, columns: list[str]) -> list[str]:
    out = []
    for col in columns:
        val = row.get(col, "")
        if val is None:
            val = ""
        out.append(str(val))
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    per_level = {}
    distortions = {}
    metrics = {}
    inputs = []

    for label, run_id in RUN_PACKS.items():
        summary_path = RUNS / run_id / "artifacts" / "geo_pipeline_summary.json"
        metrics_path = RUNS / run_id / "metrics.json"
        inputs.extend([str(summary_path), str(metrics_path)])
        summary = _geo_summary(run_id)
        per_level[label] = _per_level_arrays(summary)
        distortions[label] = _distortion_arrays(summary)
        metrics[label] = _geo_metrics(run_id)

    # Figures
    out_delta = FIGS / "compare_delta_vs_m.png"
    out_dist = FIGS / "compare_distortion_vs_m.png"
    out_mean = FIGS / "compare_mean_distance_vs_m.png"

    _plot_compare_delta(per_level, out_delta)
    _plot_compare_distortion(distortions, out_dist)
    _plot_compare_mean_distance(per_level, out_mean)

    # Tables
    hol_path = DOCS / "holonomy_demo_summary.json"
    pyth_path = DOCS / "pythagoras_rw_grid_summary.json"
    inputs.extend([str(hol_path), str(pyth_path)])

    hol = _load_json(hol_path)
    pyth = _load_json(pyth_path)

    rows = []

    for label, run_id in RUN_PACKS.items():
        m = metrics[label]
        rows.append(
            {
                "exhibit": label,
                "run_id": run_id,
                "n_micro": m.get("n_micro", ""),
                "m_finest": m.get("m_finest", ""),
                "tau": m.get("tau", ""),
                "delta_finest": m.get("delta_finest", ""),
                "stab_mean_finest": m.get("stab_mean_finest", ""),
                "stab_max_finest": m.get("stab_max_finest", ""),
                "mean_dist_finest": m.get("mean_dist_finest", ""),
                "distortion_max_rescaled": m.get("distortion_max_rescaled", ""),
                "inf_count_finest": m.get("inf_count_finest", ""),
            }
        )

    plane = hol.get("plane", {}).get("stats", {})
    sphere = hol.get("sphere", {}).get("stats", {})
    rows.append(
        {
            "exhibit": "holonomy_demo",
            "run_id": hol.get("run_id", ""),
            "plane_median": plane.get("median_angle", ""),
            "sphere_median": sphere.get("median_angle", ""),
            "median_diff": hol.get("separation", {}).get("median_diff", ""),
            "plane_evaluated": plane.get("triangles_evaluated", ""),
            "sphere_evaluated": sphere.get("triangles_evaluated", ""),
        }
    )

    pyth_rows = pyth.get("per_tau", [])
    tau_min = pyth_rows[0]["tau"] if pyth_rows else ""
    tau_max = pyth_rows[-1]["tau"] if pyth_rows else ""
    fit_rms_min = pyth_rows[0]["fit_rms"] if pyth_rows else ""
    fit_rms_max = pyth_rows[-1]["fit_rms"] if pyth_rows else ""
    pyth_med_min = pyth_rows[0]["pyth_median_abs"] if pyth_rows else ""
    pyth_med_max = pyth_rows[-1]["pyth_median_abs"] if pyth_rows else ""
    axis_quad_max = pyth_rows[-1]["axis_quad_rms"] if pyth_rows else ""
    axis_lin_max = pyth_rows[-1]["axis_lin_rms"] if pyth_rows else ""
    axis_ratio = ""
    if axis_lin_max not in (0, ""):
        axis_ratio = axis_quad_max / axis_lin_max

    rows.append(
        {
            "exhibit": "pythagoras_rw_grid",
            "run_id": "pythagoras_rw_grid_20260202T193600Z_ed66",
            "tau_min": tau_min,
            "tau_max": tau_max,
            "fit_rms_tau_min": fit_rms_min,
            "fit_rms_tau_max": fit_rms_max,
            "pyth_median_abs_tau_min": pyth_med_min,
            "pyth_median_abs_tau_max": pyth_med_max,
            "axis_quad_over_lin_at_tau_max": axis_ratio,
        }
    )

    md_path = TABLES / "exhibit_quotables.md"
    csv_path = TABLES / "exhibit_quotables.csv"

    columns = [
        "exhibit",
        "run_id",
        "n_micro",
        "m_finest",
        "tau",
        "delta_finest",
        "stab_mean_finest",
        "stab_max_finest",
        "mean_dist_finest",
        "distortion_max_rescaled",
        "inf_count_finest",
        "plane_median",
        "sphere_median",
        "median_diff",
        "plane_evaluated",
        "sphere_evaluated",
        "tau_min",
        "tau_max",
        "fit_rms_tau_min",
        "fit_rms_tau_max",
        "pyth_median_abs_tau_min",
        "pyth_median_abs_tau_max",
        "axis_quad_over_lin_at_tau_max",
    ]

    # CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(_csv_row(row, columns))

    # Markdown
    def _md_val(value: object) -> str:
        if value == "" or value is None:
            return "—"
        return str(value)

    md_lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        md_lines.append("| " + " | ".join(_md_val(row.get(col, "")) for col in columns) + " |")

    md_path.write_text("\n".join(md_lines))

    manifest_path = TABLES / "paper_ready_manifest.json"
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "outputs": [
            str(out_delta),
            str(out_dist),
            str(out_mean),
            str(md_path),
            str(csv_path),
            str(manifest_path),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
