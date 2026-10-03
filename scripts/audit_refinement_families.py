"""Check finite refinement families through the existing spectral metric pipeline.

These are controlled finite comparisons, not a proof of a manifold/fractal limit.
All stages and comparison domains are recorded; no pass threshold is fitted to
results. Historical canonical packs are never overwritten.
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
from geo_sbt.geometry.dimension import ball_growth_dimension
from geo_sbt.geometry.holonomy import classical_mds, metric_knn
from geo_sbt.geometry.metric import all_pairs_shortest_path, cost_matrix_from_kernel, distortion_between_scales
from geo_sbt.lenses.ladder import hierarchical_diffusion_partition
from geo_sbt.lenses.prototypes import prototypes_uniform
from geo_sbt.packaging import idempotence_defect_from_factors, make_C
from geo_sbt.substrates.grid import grid_2d
from geo_sbt.substrates.sierpinski import sierpinski


def layer(P_tau, labels):
    C = make_C(labels)
    U = prototypes_uniform(labels)
    B = P_tau @ C
    K = U @ B
    d = all_pairs_shortest_path(cost_matrix_from_kernel(K, symmetrize='weight_avg', eps_edge=1e-15))
    escape = 1. - np.diag(K)
    return d, {'m': len(K), 'min_fiber_size': int(np.bincount(labels).min()),
               'delta': idempotence_defect_from_factors(B, U),
               'escape_mean': float(escape.mean()), 'escape_max': float(escape.max()),
               'diameter': float(d.max()), 'inf_count': int((~np.isfinite(d)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/refinement_family_review')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    grid = []
    for side in [25, 33, 49]:
        P = grid_2d(side, .5)
        ladder = hierarchical_diffusion_partition(P, [4, 8, 16, 32, 64, 128], 6)
        for tau in [1, 5]:
            P_tau = np.linalg.matrix_power(P, tau)
            ds, records = [], []
            for labels in ladder['labels_list']:
                d, record = layer(P_tau, labels)
                ds.append(d)
                records.append(record)
            distortions = []
            for i, r in enumerate(ladder['refine_maps']):
                result = distortion_between_scales(ds[i+1], ds[i], r, rescale='lstsq')
                result['relative_to_fine_diameter'] = result['max_abs_diff'] / ds[i+1].max()
                distortions.append(result)
            row = {'n_side': side, 'lazy': .5, 'tau': tau, 'n_eigs': 6,
                   'levels': records, 'distortions': distortions}
            grid.append(row)
            print('grid', side, tau, 'max_escape', records[-1]['escape_max'], flush=True)

    geometry = []
    for level, side in [(3, 7), (4, 11), (5, 19), (6, 33)]:
        for kind in ['grid', 'gasket']:
            P = grid_2d(side, .5) if kind == 'grid' else sierpinski(level, lazy=.5)[1]
            n = len(P)
            m = min(256, max(16, n // 3))
            labels = hierarchical_diffusion_partition(P, [m], 6)['labels_list'][0]
            d, record = layer(np.linalg.matrix_power(P, 5), labels)
            record.update(kind=kind, gasket_level=level if kind == 'gasket' else None,
                          n_side=side if kind == 'grid' else None, n=n, tau=5, lazy=.5, n_eigs=6)
            try:
                record['ball_growth_slope'] = ball_growth_dimension(d)['slope']
            except ValueError as exc:
                record['ball_growth_slope'] = None
                record['ball_growth_failure'] = str(exc)
            local = []
            for k in [8, 12, 24]:
                if k >= m:
                    continue
                neighbors = metric_knn(d, k)
                errs, radii = [], []
                for i in range(m):
                    ix = np.r_[i, neighbors[i]]
                    D = d[np.ix_(ix, ix)]
                    Y = classical_mds(D, 2)
                    approx = np.linalg.norm(Y[:, None] - Y[None, :], axis=2)
                    errs.append(float(np.linalg.norm(approx-D) / np.linalg.norm(D)))
                    radii.append(float(d[i, neighbors[i]].max()))
                local.append({'k': k, 'centers': m,
                              'median_relative_mds_error': float(np.median(errs)),
                              'median_radius_over_diameter': float(np.median(radii) / d.max())})
            record['local_mds'] = local
            geometry.append(record)
            print(kind, n, m, record['ball_growth_slope'], flush=True)

    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    evidence = {'python': platform.python_version(), 'numpy': np.__version__, 'source_sha256': hashes,
                'grid_ladders': grid, 'geometry_size_family': geometry,
                'scope': 'Finite pipeline evidence; MDS reconstruction is an upper bound on best Euclidean fit error, not a nonembedding certificate.'}
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
