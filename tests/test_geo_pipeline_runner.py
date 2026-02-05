import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.runners.geo_pipeline import run_geo_pipeline


def test_geo_pipeline_runner_basic():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp) / "artifacts"
        plots_dir = Path(tmp) / "plots"
        config = {
            "run_name": "geo_pipeline_test",
            "runner": "geo_pipeline",
            "seed": 0,
            "output_root": "results",
            "artifacts_dir": str(artifacts_dir),
            "plots_dir": str(plots_dir),
            "substrate": {"kind": "grid", "n_side": 8, "lazy": 0.5},
            "lens": {"levels": [2, 4, 8], "n_eigs": 3, "seed": 0},
            "tau": 3,
            "prototypes": "uniform",
            "metric": {"symmetrize": "weight_avg", "eps_edge": 1e-15, "eta": 1e-12},
            "distortion": {"rescale": "lstsq"},
            "plots": {"enabled": False},
            "holonomy": {"enabled": False},
        }

        metrics = run_geo_pipeline(config)
        summary_path = artifacts_dir / "geo_pipeline_summary.json"
        assert summary_path.exists()
        assert "n_micro" in metrics
        assert "m_finest" in metrics
        assert "delta_finest" in metrics
        assert np.isfinite(metrics["delta_finest"])
        assert metrics["inf_count_finest"] >= 0

        summary = json.loads(summary_path.read_text())
        assert "per_level" in summary
        assert len(summary["per_level"]) == 3
