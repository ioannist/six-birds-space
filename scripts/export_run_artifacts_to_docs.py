from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

RECOMMENDED_RUNS = [
    "grid_plane_20260202T193540Z_4b14",
    "sphere_knn_20260202T193547Z_7681",
    "sierpinski_20260202T193551Z_95ab",
    "anisotropic_20260202T193556Z_310f",
    "pythagoras_rw_grid_20260202T193600Z_ed66",
]

GEO_PLOTS = [
    "delta_vs_level.png",
    "distortion_vs_level.png",
    "mean_distance_vs_level.png",
    "holonomy_hist.png",
]

PYTHAGORAS_SUMMARY_NAME = "pythagoras_rw_grid_summary.json"


def _copy_if_exists(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def _copy_log_if_small(src: Path, dst: Path, copied: list[str], max_size: int) -> None:
    if not src.exists():
        return
    if src.stat().st_size > max_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def export_run(run_id: str, *, results_root: Path, docs_root: Path, overwrite: bool) -> dict:
    src_root = results_root / run_id
    if not src_root.exists():
        print(f"missing run folder: {src_root}")
        return {"run_id": run_id, "missing": True, "copied": []}

    dst_root = docs_root / run_id
    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        else:
            print(f"skip existing: {dst_root}")
            return {"run_id": run_id, "missing": False, "copied": []}

    copied: list[str] = []

    _copy_if_exists(src_root / "pointer.json", dst_root / "pointer.json", copied)
    _copy_if_exists(src_root / "metrics.json", dst_root / "metrics.json", copied)
    _copy_if_exists(src_root / "config.json", dst_root / "config.json", copied)
    _copy_log_if_small(src_root / "log.txt", dst_root / "log.txt", copied, max_size=200_000)

    # Artifacts
    artifacts_dst = dst_root / "artifacts"
    geo_summary = src_root / "artifacts" / "geo_pipeline_summary.json"
    if geo_summary.exists():
        _copy_if_exists(geo_summary, artifacts_dst / "geo_pipeline_summary.json", copied)

    if run_id.startswith("pythagoras_rw_grid"):
        pyth_summary = src_root / "artifacts" / PYTHAGORAS_SUMMARY_NAME
        if pyth_summary.exists():
            _copy_if_exists(pyth_summary, artifacts_dst / PYTHAGORAS_SUMMARY_NAME, copied)
        else:
            docs_summary = Path("docs") / "notes" / PYTHAGORAS_SUMMARY_NAME
            if docs_summary.exists():
                _copy_if_exists(docs_summary, artifacts_dst / PYTHAGORAS_SUMMARY_NAME, copied)
                print(f"note: copied {docs_summary} into {run_id} artifacts")

    # Plots
    plots_src = src_root / "plots"
    plots_dst = dst_root / "plots"
    if plots_src.exists():
        for name in GEO_PLOTS:
            _copy_if_exists(plots_src / name, plots_dst / name, copied)
        # Copy any remaining PNGs for non-geo runs (e.g. pythagoras)
        for path in plots_src.glob("*.png"):
            if (plots_dst / path.name).exists():
                continue
            _copy_if_exists(path, plots_dst / path.name, copied)

    print(f"{run_id}: copied {len(copied)} files to {dst_root}")
    return {
        "run_id": run_id,
        "missing": False,
        "copied": [str(Path(p).resolve().relative_to(docs_root)) for p in copied],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export run artifacts into docs/notes/runs.")
    parser.add_argument("--run-id", action="append", help="Run ID to export (repeatable).")
    parser.add_argument("--recommended", action="store_true", help="Export recommended runs.")
    parser.add_argument("--results-root", default="results", help="Results root directory.")
    parser.add_argument("--docs-root", default="docs/notes/runs", help="Docs runs root directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exports.")
    args = parser.parse_args()

    run_ids = []
    if args.recommended:
        run_ids.extend(RECOMMENDED_RUNS)
    if args.run_id:
        run_ids.extend(args.run_id)
    if not run_ids:
        print("error: no run IDs provided (use --run-id or --recommended)")
        return 1

    results_root = Path(args.results_root).resolve()
    docs_root = Path(args.docs_root).resolve()
    docs_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runs": {},
    }

    for run_id in run_ids:
        record = export_run(
            run_id,
            results_root=results_root,
            docs_root=docs_root,
            overwrite=args.overwrite,
        )
        manifest["runs"][run_id] = {
            "missing": record["missing"],
            "copied": record["copied"],
        }

    manifest_path = docs_root / "_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
