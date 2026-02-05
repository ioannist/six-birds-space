import tempfile
from pathlib import Path

import numpy as np

from experiments.runners.holonomy_demo import run_holonomy_demo


def test_holonomy_demo_runner_fast():
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = Path(tmpdir) / "artifacts"
        plots = Path(tmpdir) / "plots"
        artifacts.mkdir(parents=True, exist_ok=True)
        plots.mkdir(parents=True, exist_ok=True)

        config = {
            "run_name": "holonomy_demo_test",
            "runner": "holonomy_demo",
            "seed": 0,
            "output_root": "results",
            "write_docs_artifacts": False,
            "plane": {"n_side": 10, "lazy": 0.5},
            "sphere": {"n_points": 120, "knn_k": 8, "sigma": 0.7, "self_loop": 1e-6},
            "macro": {"levels": [32], "n_eigs": 4, "seed": 0, "tau": 3},
            "holonomy": {
                "k_neigh": 12,
                "k_loop": 6,
                "expand_hops": 0,
                "min_overlap": 4,
                "max_loops": 120,
                "seed_plane": 0,
                "seed_sphere": 1,
            },
            "artifacts_dir": str(artifacts),
            "plots_dir": str(plots),
        }

        metrics = run_holonomy_demo(config)
        for key in [
            "plane_median",
            "sphere_median",
            "median_diff",
            "plane_evaluated",
            "sphere_evaluated",
        ]:
            assert key in metrics

        assert (artifacts / "holonomy_demo_summary.json").exists()

        plot_path = plots / "holonomy_plane_vs_sphere.png"
        if _matplotlib_available():
            assert plot_path.exists()


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return False
    return True
