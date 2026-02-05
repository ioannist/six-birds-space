from __future__ import annotations

import argparse
import json
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            pass

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ValueError(
                "config must be JSON or YAML; YAML requires PyYAML"
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ValueError("config must be JSON or YAML; YAML requires PyYAML")


def _resolve_config(config: dict) -> dict:
    required = ["run_name", "seed", "runner"]
    for key in required:
        if key not in config:
            raise ValueError(f"missing required config key: {key}")
    resolved = dict(config)
    resolved.setdefault("notes", "")
    resolved.setdefault("output_root", "results")
    return resolved


def _sanitize_run_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip())
    return safe.strip("-") or "run"


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _make_run_id(run_name: str, timestamp_utc: str) -> str:
    suffix = secrets.token_hex(2)
    return f"{run_name}_{timestamp_utc}_{suffix}"


def _paths_for_run(root: Path, output_root: str, run_id: str) -> Tuple[Path, Path, Path, Path, Path, Path]:
    results_root = root / output_root
    run_dir = results_root / run_id
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    pointer_path = run_dir / "pointer.json"
    log_path = run_dir / "log.txt"
    return run_dir, artifacts_dir, plots_dir, config_path, metrics_path, pointer_path, log_path


def _load_runner(runner: str):
    if runner != "smoke":
        if runner == "pythagoras_rw_grid":
            from experiments.runners.pythagoras_rw_grid import (  # type: ignore
                run_pythagoras_rw_grid,
            )

            return run_pythagoras_rw_grid
        if runner == "geo_pipeline":
            from experiments.runners.geo_pipeline import run_geo_pipeline  # type: ignore

            return run_geo_pipeline
        if runner == "holonomy_demo":
            from experiments.runners.holonomy_demo import run_holonomy_demo  # type: ignore

            return run_holonomy_demo
        raise ValueError(f"unknown runner: {runner}")
    from experiments.runners.smoke import run_smoke  # type: ignore

    return run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an experiment config.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    config_path = Path(args.config).resolve()

    config_raw = _load_config(config_path)
    config = _resolve_config(config_raw)

    run_name_safe = _sanitize_run_name(str(config["run_name"]))
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = _make_run_id(run_name_safe, timestamp_utc)

    run_dir, artifacts_dir, plots_dir, config_out, metrics_out, pointer_out, log_out = _paths_for_run(
        repo_root, config["output_root"], run_id
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config["run_id"] = run_id
    config["run_dir"] = str(run_dir)
    config["artifacts_dir"] = str(artifacts_dir)
    config["plots_dir"] = str(plots_dir)

    start_time = time.time()
    log_lines = [
        f"run_id: {run_id}",
        f"config_path: {config_path}",
        f"start_utc: {datetime.now(timezone.utc).isoformat()}\n",
    ]

    runner_fn = _load_runner(config["runner"])
    metrics: Dict[str, float] = runner_fn(config)

    elapsed = time.time() - start_time
    metrics["elapsed_sec"] = float(elapsed)
    metrics["seed"] = int(config["seed"])
    metrics["smoke_ok"] = float(metrics.get("smoke_ok", 1.0))

    config_payload = dict(config)
    config_payload["run_id"] = run_id

    with config_out.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    summary = {
        "smoke_ok": metrics["smoke_ok"],
        "seed": metrics["seed"],
        "elapsed_sec": metrics["elapsed_sec"],
    }
    extra_keys = config.get("pointer_summary_keys")
    if isinstance(extra_keys, list):
        for key in extra_keys:
            if key in metrics:
                summary[key] = metrics[key]

    pointer = {
        "run_id": run_id,
        "run_name": config["run_name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_short(),
        "config_path": str(config_path.relative_to(repo_root))
        if config_path.is_relative_to(repo_root)
        else str(config_path),
        "metrics_path": str(metrics_out.relative_to(repo_root)),
        "artifacts_dir": str(artifacts_dir.relative_to(repo_root)),
        "plots_dir": str(plots_dir.relative_to(repo_root)),
        "summary": summary,
    }

    with pointer_out.open("w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2, sort_keys=True)

    log_lines.append(f"end_utc: {datetime.now(timezone.utc).isoformat()}\n")
    with log_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
