from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _torus_kernel_fft(N: int, lazy: float) -> np.ndarray:
    if N <= 0:
        raise ValueError("N must be positive")
    if lazy < 0.0 or lazy >= 1.0:
        raise ValueError("lazy must be in [0,1)")
    K = np.zeros((N, N), dtype=np.float64)
    p = (1.0 - lazy) / 4.0
    K[0, 0] = lazy
    K[1 % N, 0] += p
    K[(N - 1) % N, 0] += p
    K[0, 1 % N] += p
    K[0, (N - 1) % N] += p
    return K


def torus_rw_distribution(N: int, lazy: float, tau: int) -> np.ndarray:
    """Compute P_tau on an N x N torus by FFT."""
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    K = _torus_kernel_fft(N, lazy)
    F = np.fft.fft2(K)
    P_tau = np.fft.ifft2(F**tau).real
    P_tau[P_tau < 0.0] = 0.0
    total = float(P_tau.sum())
    if total > 0:
        P_tau = P_tau / total
    return P_tau


def _fit_linear(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.vstack([X, np.ones_like(X)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    resid = y - (a * X + b)
    rms = float(np.sqrt(np.mean(resid**2)))
    return float(a), float(b), rms


def _compute_metrics_for_tau(
    P_tau: np.ndarray,
    tau: int,
    *,
    N: int,
    D_factor: float,
    D_max: int,
    p_floor: float,
    p_min_fit: float,
    n_triangle_samples: int,
    rng: np.random.Generator,
) -> dict:
    D = int(min(D_max, int(np.ceil(D_factor * np.sqrt(tau)))))
    if D < 1:
        D = 1

    dxs = np.arange(-D, D + 1)
    dys = np.arange(-D, D + 1)
    idx_x = np.mod(dxs, N)
    idx_y = np.mod(dys, N)
    P_win = P_tau[np.ix_(idx_x, idx_y)]

    C = -np.log(np.maximum(P_win, p_floor))
    r2 = dxs[:, None] ** 2 + dys[None, :] ** 2

    mask = P_win > p_min_fit
    X = r2[mask].astype(np.float64)
    y = C[mask].astype(np.float64)
    if X.size == 0:
        X = r2.ravel().astype(np.float64)
        y = C.ravel().astype(np.float64)
    a_fit, b_fit, fit_rms = _fit_linear(X, y)

    C_x = C[:, D]
    C_y = C[D, :]
    dx = dxs.astype(np.float64)

    aq, bq, axis_quad_rms = _fit_linear(dx * dx, C_x)
    al, bl, axis_lin_rms = _fit_linear(np.abs(dx), C_x)

    n_samples = n_triangle_samples
    dx_pos = rng.integers(1, D + 1, size=n_samples)
    dy_pos = rng.integers(1, D + 1, size=n_samples)

    idx_dx = dx_pos + D
    idx_dy = dy_pos + D
    C_xy = C[idx_dx, idx_dy]
    C_dx = C[idx_dx, D]
    C_dy = C[D, idx_dy]
    C_00 = C[D, D]
    residual = C_xy - (C_dx + C_dy - C_00)
    pyth_rms = float(np.sqrt(np.mean(residual**2)))
    pyth_median_abs = float(np.median(np.abs(residual)))

    r2_vals = r2.ravel()
    C_vals = C.ravel()
    circular_stds = []
    for r2_val in np.unique(r2_vals):
        idx = r2_vals == r2_val
        if np.sum(idx) >= 8:
            circular_stds.append(float(np.std(C_vals[idx])))
    circularity_mean_std = float(np.mean(circular_stds)) if circular_stds else float("nan")

    return {
        "tau": tau,
        "D": int(D),
        "a_fit": a_fit,
        "b_fit": b_fit,
        "fit_rms": fit_rms,
        "pyth_rms": pyth_rms,
        "pyth_median_abs": pyth_median_abs,
        "axis_quad_rms": axis_quad_rms,
        "axis_lin_rms": axis_lin_rms,
        "circularity_mean_std": circularity_mean_std,
        "C_grid": C,
    }


def _compute_control_L1(D: int) -> dict:
    dxs = np.arange(-D, D + 1)
    dys = np.arange(-D, D + 1)
    C = np.abs(dxs[:, None]) + np.abs(dys[None, :])
    r2 = dxs[:, None] ** 2 + dys[None, :] ** 2
    X = r2.ravel().astype(np.float64)
    y = C.ravel().astype(np.float64)
    a_fit, b_fit, fit_rms = _fit_linear(X, y)

    C_x = C[:, D]
    dx = dxs.astype(np.float64)
    aq, bq, axis_quad_rms = _fit_linear(dx * dx, C_x)
    al, bl, axis_lin_rms = _fit_linear(np.abs(dx), C_x)

    r2_vals = r2.ravel()
    C_vals = C.ravel()
    circular_stds = []
    for r2_val in np.unique(r2_vals):
        idx = r2_vals == r2_val
        if np.sum(idx) >= 8:
            circular_stds.append(float(np.std(C_vals[idx])))
    circularity_mean_std = float(np.mean(circular_stds)) if circular_stds else float("nan")

    return {
        "fit_rms_L1": fit_rms,
        "axis_quad_rms_L1": axis_quad_rms,
        "axis_lin_rms_L1": axis_lin_rms,
        "circularity_mean_std_L1": circularity_mean_std,
        "C_grid_L1": C,
    }


def _save_plots(
    summary: dict,
    *,
    plots_dir: Path,
    docs_fig_dir: Path | None,
    write_docs: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib unavailable; plots skipped")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    if write_docs and docs_fig_dir is not None:
        docs_fig_dir.mkdir(parents=True, exist_ok=True)

    per_tau = summary["per_tau"]
    taus = [row["tau"] for row in per_tau]
    fit_rms = [row["fit_rms"] for row in per_tau]
    pyth_med = [row["pyth_median_abs"] for row in per_tau]
    axis_quad = [row["axis_quad_rms"] for row in per_tau]
    axis_lin = [row["axis_lin_rms"] for row in per_tau]

    plt.figure(figsize=(6, 4))
    plt.plot(taus, fit_rms, marker="o", label="fit_rms")
    plt.plot(taus, pyth_med, marker="o", label="pyth_median_abs")
    plt.plot(taus, axis_quad, marker="o", label="axis_quad_rms")
    plt.plot(taus, axis_lin, marker="o", label="axis_lin_rms")
    plt.xscale("log", base=2)
    plt.xlabel("tau")
    plt.ylabel("residual")
    plt.legend()
    plt.tight_layout()

    if write_docs and docs_fig_dir is not None:
        plot_path = docs_fig_dir / "pythagoras_residuals_vs_tau.png"
        plt.savefig(plot_path)
    plt.savefig(plots_dir / "pythagoras_residuals_vs_tau.png")
    plt.close()

    min_tau = taus[0]
    max_tau = taus[-1]
    C_min = per_tau[0]["C_grid"]
    C_max = per_tau[-1]["C_grid"]
    C_L1 = summary["control_L1"]["C_grid_L1"]

    vmax = np.percentile(np.concatenate([C_min.ravel(), C_max.ravel(), C_L1.ravel()]), 95)
    vmin = 0.0

    def _contour(C, name):
        plt.figure(figsize=(4, 4))
        plt.contourf(C, levels=20, vmin=vmin, vmax=vmax)
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        if write_docs and docs_fig_dir is not None:
            plt.savefig(docs_fig_dir / name)
        plt.savefig(plots_dir / name)
        plt.close()

    _contour(C_min, f"pythagoras_contour_rw_tau_{min_tau}.png")
    _contour(C_max, f"pythagoras_contour_rw_tau_{max_tau}.png")
    _contour(C_L1, "pythagoras_contour_control_L1.png")


def run_pythagoras_rw_grid(config: dict) -> Dict[str, float]:
    """Run the Pythagoras emergence experiment on a torus grid."""
    N = int(config.get("N", 512))
    lazy = float(config.get("lazy", 0.5))
    tau_list = [int(t) for t in config.get("tau_list", [4, 8, 16, 32, 64, 128])]
    D_factor = float(config.get("D_factor", 3.0))
    D_max = int(config.get("D_max", 30))
    p_floor = float(config.get("p_floor", 1e-300))
    p_min_fit = float(config.get("p_min_fit", 1e-20))
    n_triangle_samples = int(config.get("n_triangle_samples", 2000))
    seed = int(config.get("seed", 0))
    write_docs = bool(config.get("write_docs_artifacts", True))
    allow_aliasing = bool(config.get("allow_aliasing", False))

    tau_list = sorted(tau_list)
    if N <= 2 * max(tau_list) + 1:
        if not allow_aliasing:
            raise ValueError("N must be > 2*tau_max + 1 to avoid aliasing")
        print("warning: N is too small for tau_max; aliasing likely")

    rng = np.random.default_rng(seed)

    per_tau: List[dict] = []
    for tau in tau_list:
        P_tau = torus_rw_distribution(N, lazy, tau)
        metrics = _compute_metrics_for_tau(
            P_tau,
            tau,
            N=N,
            D_factor=D_factor,
            D_max=D_max,
            p_floor=p_floor,
            p_min_fit=p_min_fit,
            n_triangle_samples=n_triangle_samples,
            rng=rng,
        )
        per_tau.append(metrics)

    control_by_tau = []
    for row in per_tau:
        D = row["D"]
        control_metrics = _compute_control_L1(D)
        control_by_tau.append({"tau": row["tau"], "D": D, **control_metrics})

    summary = {
        "config": {
            "N": N,
            "lazy": lazy,
            "tau_list": tau_list,
            "D_factor": D_factor,
            "D_max": D_max,
            "p_floor": p_floor,
            "p_min_fit": p_min_fit,
            "n_triangle_samples": n_triangle_samples,
        },
        "per_tau": [
            {k: v for k, v in row.items() if k != "C_grid"} for row in per_tau
        ],
        "control_L1": {
            "per_tau": [
                {k: v for k, v in row.items() if k != "C_grid_L1"} for row in control_by_tau
            ],
        },
    }

    if write_docs:
        docs_summary_path = Path("docs") / "notes" / "pythagoras_rw_grid_summary.json"
        docs_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with docs_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    artifacts_dir = Path(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with (artifacts_dir / "pythagoras_rw_grid_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    plots_dir = Path(config["plots_dir"])
    docs_fig_dir = Path("docs") / "notes" / "figures"
    # include grids for plotting in summary copy
    plot_summary = {
        **summary,
        "per_tau": per_tau,
        "control_L1": {"C_grid_L1": control_by_tau[0]["C_grid_L1"], **summary["control_L1"]},
    }
    _save_plots(
        plot_summary,
        plots_dir=plots_dir,
        docs_fig_dir=docs_fig_dir if write_docs else None,
        write_docs=write_docs,
    )

    tau_min = per_tau[0]
    tau_max = per_tau[-1]

    metrics = {
        "tau_min": float(tau_min["tau"]),
        "tau_max": float(tau_max["tau"]),
        "fit_rms_tau_min": float(tau_min["fit_rms"]),
        "fit_rms_tau_max": float(tau_max["fit_rms"]),
        "pyth_med_tau_min": float(tau_min["pyth_median_abs"]),
        "pyth_med_tau_max": float(tau_max["pyth_median_abs"]),
        "axis_quad_rms_tau_min": float(tau_min["axis_quad_rms"]),
        "axis_quad_rms_tau_max": float(tau_max["axis_quad_rms"]),
        "axis_lin_rms_tau_max": float(tau_max["axis_lin_rms"]),
        "fit_rms_improve_ratio": float(tau_max["fit_rms"] / tau_min["fit_rms"]),
        "axis_quad_over_lin_tau_max": float(
            tau_max["axis_quad_rms"] / tau_max["axis_lin_rms"]
        ),
        "axis_quad_over_lin_L1": float(
            control_by_tau[-1]["axis_quad_rms_L1"] / control_by_tau[-1]["axis_lin_rms_L1"]
        ),
    }
    return metrics
