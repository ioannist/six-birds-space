# Artifact Contract (v0)

This document defines the minimal artifact contract for all experiment runs.

## 1) Config schema (minimal v0)

Each run config must include:

- `run_name` (string)
- `seed` (int)
- `notes` (string, optional)
- `output_root` (string, default: `"results"`)
- `runner` (string, e.g. `"smoke"`)

Configs may include additional fields, but the above keys are required for v0 runs.

## 2) Output layout

Every run must create a folder:

```
results/<run_id>/
```

The run folder must contain:

- `config.json` (resolved config saved as JSON)
- `log.txt` (basic runtime log)
- `metrics.json` (dictionary of scalar metrics)
- `pointer.json` (stable interface for later automation)
- `artifacts/` (directory; may be empty)
- `plots/` (directory; may be empty)

## 3) pointer.json schema (minimal v0)

The pointer file must include at minimum:

- `run_id`
- `run_name`
- `timestamp_utc`
- `git_commit` (string; `"unknown"` allowed if not available)
- `config_path` (path relative to repo root if possible)
- `metrics_path`
- `artifacts_dir`
- `plots_dir`
- `summary` (short dict with 2–5 items; can mirror metrics)

Additional keys are allowed, but the above must always be present.
