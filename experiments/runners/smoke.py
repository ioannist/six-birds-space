from __future__ import annotations

from typing import Dict

import numpy as np


def run_smoke(config: dict) -> Dict[str, float]:
    """Deterministic smoke runner.

    Generates a small random vector with a fixed seed and reports mean/var.
    """
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    vec = rng.normal(loc=0.0, scale=1.0, size=16)
    mean = float(np.mean(vec))
    var = float(np.var(vec))
    return {
        "smoke_ok": 1.0,
        "seed": seed,
        "vec_mean": mean,
        "vec_var": var,
    }
