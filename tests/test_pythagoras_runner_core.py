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


def test_support_and_parity_are_not_fft_roundoff():
    p = torus_rw_distribution(32, lazy=.5, tau=4)
    d = np.abs(np.fft.fftfreq(32) * 32)
    assert np.all(p[d[:, None] + d[None, :] > 4] == 0)
    p = torus_rw_distribution(32, lazy=0., tau=4)
    odd = (d[:, None] + d[None, :]) % 2 == 1
    assert np.all(p[odd] == 0)


def test_l1_control_passes_separability_but_fails_squared_distance():
    from experiments.runners.pythagoras_rw_grid import _compute_control_L1
    result = _compute_control_L1(5)
    assert result['pyth_median_abs_L1'] == 0.
    assert result['squared_distance_residual_median_L1'] > 0.
    assert result['axis_lin_rms_L1'] < 1e-12
    assert result['axis_quad_rms_L1'] > .1


def test_runner_rejects_fractional_stages_and_grid_size(tmp_path):
    import pytest
    from experiments.runners.pythagoras_rw_grid import run_pythagoras_rw_grid
    for config in [{'N': 64.5}, {'N': 64, 'tau_list': [4.5]}]:
        config.update(artifacts_dir=str(tmp_path/'artifacts'), write_docs_artifacts=False)
        with pytest.raises(ValueError, match='integer'):
            run_pythagoras_rw_grid(config)
    assert not (tmp_path/'artifacts').exists()
