# Six Birds: Space Instantiation

This repository contains the **space/geometry instantiation** for the paper:

> **To Plot a Stone with Six Birds: Constructing and Auditing Emergent Geometry from Markov Dynamics**
>
> Archived at: https://zenodo.org/records/18494975
>
> DOI: https://doi.org/10.5281/zenodo.18494975

This paper is the geometry-focused instantiation of the emergence calculus introduced in *Six Birds: Foundations of Emergence Calculus*. It demonstrates how a space-like layer (points, distances, curvature) can be constructed as a closure artifact from micro-dynamics, and audited via falsification-first diagnostics.

The [current mathematical review](docs/notes/review_claims_20261003.md) records
corrected claims, constructive replacements and verification receipts. The
archived manuscript and original run packs remain historical: in particular,
their learned-lens examples do not establish small-defect coherence.

## What this repository provides

The space instantiation implements:

- **Core packaging engine**: lenses, prototypes, closure operator, idempotence and stability defects
- **Route mismatch and distortion**: coherence diagnostics across refinement ladders
- **Substrate generators**: grid, sphere kNN, Sierpi\'nski gasket, anisotropic gating
- **Lens ladders**: diffusion/spectral embeddings with deterministic k-means and refinement maps
- **Emergent metric pipeline**: macro kernel, cost from likelihood, shortest-path distances
- **Holonomy diagnostic**: curvature-like loop residue via local MDS and Procrustes transport
- **Staged diffusion experiment**: finite quadratic-cost evidence and exact uniform certificates; L1 controls distinguish separability from a quadratic law
- **Artifact contract + run packs**: committed run packs under `docs/notes/runs/` and paper-ready comparison figures/tables
- **Constructive controls**: block-grid and recursive-gasket lenses with vanishing defects under explicit joint refinement and distance units
- **Lean anchors**: weighted extended path metrics, separation quotients, finite Markov closure bounds, Hilbert obstructions and conditional quadratic readouts

## Scope and limitations

The reviewed constructions have the following scope:

- Diagnostics are audit gates, not proofs of manifold convergence
- Geometry is layer-relative: different lenses can yield different macro spaces
- Holonomy is a diagnostic curvature proxy, not a curvature tensor estimate
- Pythagoras emergence is a mechanism exhibit, not a general theorem about all emergent metrics
- Constructive cell lenses are supplied interfaces; their proofs do not validate the canonical spectral partitions

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[audit]"
cd lean && lake build
```

The audit extra includes SciPy for the exact certificates, sparse constructive
audits and full test suite. The core library depends only on NumPy.

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
