# Six Birds: Space Instantiation

This repository contains the **space/geometry instantiation** for the paper:

> **To Plot a Stone with Six Birds: A Geometry is A Theory**
>
> Archived at: https://zenodo.org/records/18494975
>
> DOI: https://doi.org/10.5281/zenodo.18494975

This paper is the geometry-focused instantiation of the emergence calculus introduced in *Six Birds: Foundations of Emergence Calculus*. It demonstrates how a space-like layer (points, distances, curvature) can be constructed as a closure artifact from micro-dynamics, and audited via falsification-first diagnostics.

## What this repository provides

The space instantiation implements:

- **Core packaging engine**: lenses, prototypes, closure operator, idempotence and stability defects
- **Route mismatch and distortion**: coherence diagnostics across refinement ladders
- **Substrate generators**: grid, sphere kNN, Sierpi\'nski gasket, anisotropic gating
- **Lens ladders**: diffusion/spectral embeddings with deterministic k-means and refinement maps
- **Emergent metric pipeline**: macro kernel, cost from likelihood, shortest-path distances
- **Holonomy diagnostic**: curvature-like loop residue via local MDS and Procrustes transport
- **Pythagoras experiment**: accounting cost becomes quadratic under staged isotropic diffusion, with a negative control
- **Artifact contract + run packs**: committed run packs under `docs/notes/runs/` and paper-ready comparison figures/tables
- **Lean anchors**: minimal formal support for shortest-path pseudometric and separation quotient

## Scope and limitations

The paper is explicit about what it does and does not establish:

- Diagnostics are audit gates, not proofs of manifold convergence
- Geometry is layer-relative: different lenses can yield different macro spaces
- Holonomy is a diagnostic curvature proxy, not a curvature tensor estimate
- Pythagoras emergence is a mechanism exhibit, not a general theorem about all emergent metrics

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd lean && lake build
```

## Test

```bash
pytest -q
```

## Run experiments (canonical configs)

```bash
python experiments/run.py --config experiments/configs/grid_plane.yaml
python experiments/run.py --config experiments/configs/sphere_knn.yaml
python experiments/run.py --config experiments/configs/sierpinski.yaml
python experiments/run.py --config experiments/configs/anisotropic.yaml
python experiments/run.py --config experiments/configs/holonomy_demo.yaml
python experiments/run.py --config experiments/configs/pythagoras_rw_grid.yaml
```

## Build paper

```bash
cd paper && make pdf
```

## Generate paper-ready comparisons

```bash
python scripts/make_paper_ready_comparisons.py
```
