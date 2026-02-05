# Result index (internal) — 2026-02-02

This is the single entry point for recommended runs, artifacts, and reproduction commands. It is internal only and intended to speed up later paper writing by pointing to concrete runs, plots, and metrics.

Portability note: `results/` is ignored by git and may be missing in a fresh clone. If any referenced `results/<run_id>/...` folder is missing, rerun the corresponding config to regenerate it. The docs packs under `docs/notes/runs/` are committed; `results/` is a regeneratable source.

## Quick-start reproduction (baseline suite)

```bash
python experiments/run.py --config experiments/configs/grid_plane.yaml
python experiments/run.py --config experiments/configs/sphere_knn.yaml
python experiments/run.py --config experiments/configs/sierpinski.yaml
python experiments/run.py --config experiments/configs/anisotropic.yaml
python experiments/run.py --config experiments/configs/pythagoras_rw_grid.yaml
```

```bash
python scripts/toy_holonomy_demo.py
python scripts/toy_dimension_demo.py
cd lean && lake build
```

## Exhibit index

### E1 — Emergent metric on plane-like substrate (grid)
- Recommended run: `grid_plane_20260202T193540Z_4b14`
- Config: `experiments/configs/grid_plane.yaml`
- Docs pack (preferred):
  - `docs/notes/runs/grid_plane_20260202T193540Z_4b14/artifacts/geo_pipeline_summary.json`
  - `docs/notes/runs/grid_plane_20260202T193540Z_4b14/metrics.json`
  - `docs/notes/runs/grid_plane_20260202T193540Z_4b14/pointer.json`
  - `docs/notes/runs/grid_plane_20260202T193540Z_4b14/plots/delta_vs_level.png`
  - `docs/notes/runs/grid_plane_20260202T193540Z_4b14/plots/distortion_vs_level.png`
  - `docs/notes/runs/grid_plane_20260202T193540Z_4b14/plots/mean_distance_vs_level.png`
- Results fallback:
  - `results/grid_plane_20260202T193540Z_4b14/artifacts/geo_pipeline_summary.json`
  - `results/grid_plane_20260202T193540Z_4b14/metrics.json`
  - `results/grid_plane_20260202T193540Z_4b14/pointer.json`
  - `results/grid_plane_20260202T193540Z_4b14/plots/delta_vs_level.png`
  - `results/grid_plane_20260202T193540Z_4b14/plots/distortion_vs_level.png`
  - `results/grid_plane_20260202T193540Z_4b14/plots/mean_distance_vs_level.png`
- Metrics to quote (names only): `delta_finest`, `stab_mean_finest`, `stab_max_finest`, `distortion_max_rescaled`, `mean_dist_finest`, `inf_count_finest`

### E2 — Curvature/holonomy separation (plane vs sphere)
- Recommended run: `sphere_knn_20260202T193547Z_7681`
- Config: `experiments/configs/sphere_knn.yaml`
- Docs pack (preferred):
  - `docs/notes/runs/sphere_knn_20260202T193547Z_7681/artifacts/geo_pipeline_summary.json`
  - `docs/notes/runs/sphere_knn_20260202T193547Z_7681/metrics.json`
  - `docs/notes/runs/sphere_knn_20260202T193547Z_7681/pointer.json`
  - `docs/notes/runs/sphere_knn_20260202T193547Z_7681/plots/holonomy_hist.png`
- Results fallback:
  - `results/sphere_knn_20260202T193547Z_7681/artifacts/geo_pipeline_summary.json`
  - `results/sphere_knn_20260202T193547Z_7681/metrics.json`
  - `results/sphere_knn_20260202T193547Z_7681/pointer.json`
  - `results/sphere_knn_20260202T193547Z_7681/plots/holonomy_hist.png`
- Recommended committed figure: `docs/notes/figures/holonomy_plane_vs_sphere.png`
- Recommended stats JSON: `docs/notes/holonomy_demo_summary.json`
- Metrics to quote (names only): `holonomy_median`, `holonomy_mean`, `holonomy_evaluated`
- Reproduce plane-vs-sphere holonomy (committed fig+JSON): `python experiments/run.py --config experiments/configs/holonomy_demo.yaml`
- Robustness note: failure mode run `sphere_holonomy_unstable_20260202T202932Z_1b76` (see `docs/notes/robustness_20260202.md`)

### E3 — Fractal regime (Sierpinski)
- Recommended run: `sierpinski_20260202T193551Z_95ab`
- Config: `experiments/configs/sierpinski.yaml`
- Docs pack (preferred):
  - `docs/notes/runs/sierpinski_20260202T193551Z_95ab/artifacts/geo_pipeline_summary.json`
  - `docs/notes/runs/sierpinski_20260202T193551Z_95ab/metrics.json`
  - `docs/notes/runs/sierpinski_20260202T193551Z_95ab/pointer.json`
  - `docs/notes/runs/sierpinski_20260202T193551Z_95ab/plots/`
- Results fallback:
  - `results/sierpinski_20260202T193551Z_95ab/artifacts/geo_pipeline_summary.json`
  - `results/sierpinski_20260202T193551Z_95ab/plots/`
- Metrics to quote (names only): `delta_finest`, `stab_mean_finest`, `distortion_max_rescaled`
- Dimension check: run `python scripts/toy_dimension_demo.py` and use ball_dim_on_d_cost (not sqrt(d_cost)).

### E4 — P2 constraints / anisotropy
- Recommended run: `anisotropic_20260202T193556Z_310f`
- Config: `experiments/configs/anisotropic.yaml`
- Compare against run: `grid_plane_20260202T193540Z_4b14`
- Docs pack (preferred):
  - `docs/notes/runs/anisotropic_20260202T193556Z_310f/artifacts/geo_pipeline_summary.json`
  - `docs/notes/runs/anisotropic_20260202T193556Z_310f/metrics.json`
  - `docs/notes/runs/anisotropic_20260202T193556Z_310f/pointer.json`
  - `docs/notes/runs/anisotropic_20260202T193556Z_310f/plots/`
- Results fallback:
  - `results/anisotropic_20260202T193556Z_310f/artifacts/geo_pipeline_summary.json`
  - `results/anisotropic_20260202T193556Z_310f/plots/`
- Metrics to quote (names only): `delta_finest`, `stab_mean_finest`, `stab_max_finest`, `distortion_max_rescaled`, `mean_dist_finest`

### E5 — Pythagoras emergence (accounting cost becomes quadratic)
- Recommended run: `pythagoras_rw_grid_20260202T193600Z_ed66`
- Config: `experiments/configs/pythagoras_rw_grid.yaml`
- Docs pack (preferred):
  - `docs/notes/runs/pythagoras_rw_grid_20260202T193600Z_ed66/metrics.json`
  - `docs/notes/runs/pythagoras_rw_grid_20260202T193600Z_ed66/pointer.json`
  - `docs/notes/runs/pythagoras_rw_grid_20260202T193600Z_ed66/config.json`
  - `docs/notes/runs/pythagoras_rw_grid_20260202T193600Z_ed66/artifacts/pythagoras_rw_grid_summary.json`
- Results fallback:
  - `results/pythagoras_rw_grid_20260202T193600Z_ed66/metrics.json`
  - `results/pythagoras_rw_grid_20260202T193600Z_ed66/pointer.json`
  - `results/pythagoras_rw_grid_20260202T193600Z_ed66/config.json`
  - `results/pythagoras_rw_grid_20260202T193600Z_ed66/artifacts/pythagoras_rw_grid_summary.json`
- Committed docs artifacts to cite:
  - `docs/notes/pythagoras_rw_grid_summary.json`
  - `docs/notes/figures/pythagoras_residuals_vs_tau.png`
  - `docs/notes/figures/pythagoras_contour_rw_tau_4.png`
  - `docs/notes/figures/pythagoras_contour_rw_tau_128.png`
  - `docs/notes/figures/pythagoras_contour_control_L1.png`
- Metrics to quote (names only): `fit_rms` vs tau trend, `pyth_median_abs` vs tau trend, `axis_quad_rms` vs `axis_lin_rms`, control L1 comparison.

## Robustness / failure modes
- Note: `docs/notes/robustness_20260202.md`
- Sweep map (authoritative config → run_id): `experiments/configs/sweeps/_run_map.json`

## Paper-ready comparisons
- Generator: `python scripts/make_paper_ready_comparisons.py`
- Figures: `docs/notes/figures/compare_delta_vs_m.png`, `docs/notes/figures/compare_distortion_vs_m.png`, `docs/notes/figures/compare_mean_distance_vs_m.png`
- Quotables table: `docs/notes/tables/exhibit_quotables.md` and `docs/notes/tables/exhibit_quotables.csv`
- Build manifest: `docs/notes/tables/paper_ready_manifest.json`

## Lean anchors
- Build: `cd lean && lake build`
- Theorems (as named in code):
  - `graph_edist_triangle` (triangle inequality for graph path cost)
  - `separation_quotient_metric`, `separation_quotient_dist_eq_zero` (zero-distance quotient gives metric)
  - `pythagoras_real` (orthogonality implies squared-norm additivity)

## Where the numbers live
- geo_pipeline per-level values: `docs/notes/runs/<run_id>/artifacts/geo_pipeline_summary.json` (preferred) or `results/<run_id>/artifacts/geo_pipeline_summary.json` (fallback)
- Pythagoras per-tau values: `docs/notes/pythagoras_rw_grid_summary.json` and `docs/notes/runs/pythagoras_rw_grid_20260202T193600Z_ed66/artifacts/pythagoras_rw_grid_summary.json`
- Holonomy stats: `docs/notes/holonomy_demo_summary.json` and `docs/notes/runs/sphere_knn_20260202T193547Z_7681/metrics.json`
