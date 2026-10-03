"""Reproduce canonical mathematical evidence without overwriting historical packs.

Run from the repository root. Configs in this repository use JSON syntax,
including those named .yaml. This is numerical evidence, not a proof of a
continuum limit, small-defect closure, or identification with curvature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from experiments.runners.geo_pipeline import run_geo_pipeline
from experiments.runners.holonomy_demo import run_holonomy_demo
from experiments.runners.pythagoras_rw_grid import run_pythagoras_rw_grid
from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel, macro_kernel
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition
from geo_sbt.lenses.prototypes import prototypes_uniform
from geo_sbt.packaging import make_C, prototype_stabilities
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.constraints import anisotropic_gate
from scripts.toy_dimension_demo import _compute_dimensions
from geo_sbt.substrates.sierpinski import sierpinski


def clean(value):
    if isinstance(value, dict):
        return {key: clean(v) for key, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'math_review')
    parser.add_argument('--sweeps', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ['grid_plane', 'sphere_knn', 'sierpinski', 'anisotropic', 'holonomy_demo', 'pythagoras_rw_grid']
    configs = [ROOT / 'experiments' / 'configs' / (name + '.yaml') for name in names]
    if args.sweeps:
        configs += sorted((ROOT / 'experiments' / 'configs' / 'sweeps').glob('*.yaml'))
    runners = {'geo_pipeline': run_geo_pipeline, 'holonomy_demo': run_holonomy_demo,
               'pythagoras_rw_grid': run_pythagoras_rw_grid}
    sources = (sorted((ROOT / 'src').rglob('*.py'))
               + sorted((ROOT / 'experiments').rglob('*.py'))
               + [ROOT / 'scripts/audit_mathematics.py', ROOT / 'scripts/toy_dimension_demo.py']
               + sorted((ROOT / 'lean').glob('*.lean'))
               + sorted((ROOT / 'lean/GeoSBT').glob('*.lean')))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    evidence = {}
    for path in configs:
        cfg = json.loads(path.read_text())
        cfg.update(write_docs_artifacts=False, run_id='math_review_' + path.stem,
                   artifacts_dir=str(out / path.stem / 'artifacts'),
                   plots_dir=str(out / path.stem / 'plots'))
        cfg['plots'] = {'enabled': True}
        result = runners[cfg['runner']](cfg)
        evidence[path.stem] = {'config_source': str(path.relative_to(ROOT)), 'metrics': clean(result)}
        print(path.stem, json.dumps(clean(result), sort_keys=True), flush=True)

    # Exhaustive finite metric checks and the exact disjoint-fiber stability identity.
    P = grid_2d(8, lazy=.5)
    ladder = hierarchical_diffusion_partition(P, [2, 4, 8, 16], 4)
    checks = []
    for labels in ladder['labels_list']:
        C, U = make_C(labels), prototypes_uniform(labels)
        K = macro_kernel(P, 3, C, U)
        d = all_pairs_shortest_path(cost_matrix_from_kernel(K, symmetrize='weight_avg'))
        triangle_violation = max(float(np.max(d - (d[:, k, None] + d[None, k, :]))) for k in range(len(d)))
        stability_error = float(np.max(np.abs(prototype_stabilities(P, 3, C, U) - (1. - np.diag(K)))))
        assert triangle_violation <= 1e-10
        assert stability_error <= 1e-10
        assert np.allclose(d, d.T, atol=1e-10)
        checks.append({'m': len(d), 'triangle_violation': triangle_violation,
                       'stability_identity_error': stability_error})
    # Hold the grid lens fixed to isolate the dynamical effect of gating.
    base_grid = grid_2d(25, lazy=.5)
    fixed_labels = hierarchical_diffusion_partition(base_grid, [128], 6)['labels_list'][0]
    fixed_C, fixed_U = make_C(fixed_labels), prototypes_uniform(fixed_labels)
    gate_cfg = json.loads((ROOT / 'experiments/configs/anisotropic.yaml').read_text())['constraints']
    gated = anisotropic_gate(base_grid, gate_cfg['direction'], gate_cfg['strength'])
    kernels = [macro_kernel(p, 5, fixed_C, fixed_U) for p in [base_grid, gated]]
    metrics = [all_pairs_shortest_path(cost_matrix_from_kernel(k, symmetrize='weight_avg', eps_edge=1e-15)) for k in kernels]
    fixed_lens_control = {'constraints': gate_cfg,
                          'macro_kernel_max_change': float(np.max(np.abs(kernels[1] - kernels[0]))),
                          'distance_max_change': float(np.max(np.abs(metrics[1] - metrics[0])))}
    for name, dist in zip(['baseline', 'gated'], metrics):
        horizontal = [(r * 25 + c, r * 25 + c + 4) for r in range(25) for c in range(21)]
        vertical = [(r * 25 + c, (r + 4) * 25 + c) for r in range(21) for c in range(25)]
        means = [float(np.mean([dist[fixed_labels[a], fixed_labels[b]] for a, b in pairs])) for pairs in [horizontal, vertical]]
        fixed_lens_control[name] = {'horizontal_mean': means[0], 'vertical_mean': means[1], 'horizontal_over_vertical': means[0] / means[1]}
    assert fixed_lens_control['distance_max_change'] > 1e-6
    _, gasket = sierpinski(5, lazy=.5)
    dimensions = {'grid': _compute_dimensions(grid_2d(20, lazy=.5), also_sqrt=True),
                  'sierpinski': _compute_dimensions(gasket, also_sqrt=True)}
    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "source changed during audit"
    record = {'python': platform.python_version(), 'numpy': np.__version__,
              'source_sha256': hashes, 'runs': evidence, 'metric_checks': checks, 'fixed_lens_anisotropy': fixed_lens_control, 'dimension_proxies': dimensions}
    (out / 'evidence.json').write_text(json.dumps(clean(record), indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
