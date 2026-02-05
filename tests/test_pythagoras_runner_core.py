import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.runners.pythagoras_rw_grid import torus_rw_distribution


def test_fft_distribution_core():
    N = 64
    P_tau = torus_rw_distribution(N, lazy=0.5, tau=10)
    assert abs(P_tau.sum() - 1.0) < 1e-9
    assert P_tau.min() >= -1e-12
    assert np.isfinite(P_tau[0, 0])
    assert P_tau[0, 0] > 0.0
