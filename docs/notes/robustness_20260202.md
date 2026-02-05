# Robustness sweeps — 2026-02-02

## Executive summary
- Grid: increasing tau from 1 to 20 raises prototype instability (stab_mean 0.272 → 0.849; run_ids grid_tau_1_20260202T202923Z_5174 vs grid_tau_20_20260202T202924Z_8e0f).
- Grid: finer ladder [8..256] increases distortion (rescaled 5.598 → 8.708; run_ids grid_levels_coarse_20260202T202927Z_b787 vs grid_levels_finer_20260202T202928Z_d009).
- Sphere holonomy: brittle settings cut evaluated loops (eval 363 → 190) and inflate median angle (0.585 → 0.839); run_id sphere_holonomy_unstable_20260202T202932Z_1b76.
- Pythagoras aliasing (N too small): fit improvement ratio stalls at 0.589 and axis_quad_rms worsens with tau (min 0.004583 → max 0.01418); run_id pythagoras_aliasing_fail_20260202T202954Z_2779.
- Pythagoras large-tau sweep: strong quadratic regime (axis_quad/axis_lin@max 0.0047); run_id pythagoras_tau_large_20260202T202934Z_f96a.

## Grid (geo_pipeline) sweeps
| config | run_id | tau | delta_finest | stab_mean / stab_max | inf_count_finest | distortion_max_rescaled |
|---|---|---:|---:|---:|---:|---:|
| grid_tau_1.yaml | grid_tau_1_20260202T202923Z_5174 | 1 | 0.2656 | 0.2721/0.5 | 0.0000e+00 | 11.09 |
| grid_tau_20.yaml | grid_tau_20_20260202T202924Z_8e0f | 20 | 0.2775 | 0.8486/0.9681 | 0.0000e+00 | 5.187 |
| grid_prototypes_stationary.yaml | grid_prototypes_stationary_20260202T202925Z_9f4e | 5 | 0.3164 | 0.6305/0.8687 | 0.0000e+00 | 6.571 |
| grid_eps_edge_disconnect.yaml | grid_eps_edge_disconnect_20260202T202926Z_f6ec | 5 | 0.3182 | 0.6291/0.8687 | 0.0000e+00 | 6.491 |
| grid_levels_coarse.yaml | grid_levels_coarse_20260202T202927Z_b787 | 5 | 0.3162 | 0.3353/0.5659 | 0.0000e+00 | 5.598 |
| grid_levels_finer.yaml | grid_levels_finer_20260202T202928Z_d009 | 5 | 0.3025 | 0.7556/0.8687 | 0.0000e+00 | 8.708 |

Failure mode reproduced: large tau (20) yields high prototype instability (stab_mean≈0.85) indicating collapse toward unstable packaging; see run_id listed above.

## Sphere (geo_pipeline + holonomy) sweeps
| config | run_id | tau | delta_finest | stab_mean/stab_max | inf_count_finest | distortion_max_rescaled | holonomy_eval | holonomy_median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sphere_tau_1.yaml | sphere_tau_1_20260202T202928Z_daa9 | 1 | 0.5279 | 0.7323/1 | 0.0000e+00 | 6.305 | 361 | 0.8613 |
| sphere_tau_30.yaml | sphere_tau_30_20260202T202929Z_e3bf | 30 | 0.2396 | 0.9658/0.996 | 0.0000e+00 | 4.891 | 500 | 0.7732 |
| sphere_prototypes_stationary.yaml | sphere_prototypes_stationary_20260202T202930Z_76a4 | 5 | 0.2693 | 0.8551/0.9851 | 0.0000e+00 | 5.319 | 363 | 0.5847 |
| sphere_holonomy_unstable.yaml | sphere_holonomy_unstable_20260202T202932Z_1b76 | 5 | 0.2672 | 0.8552/0.9851 | 0.0000e+00 | 5.273 | 190 | 0.8393 |
| sphere_eps_edge_disconnect.yaml | sphere_eps_edge_disconnect_20260202T202932Z_36c8 | 5 | 0.2672 | 0.8552/0.9851 | 0.0000e+00 | 5.269 | 369 | 0.7042 |

Failure mode reproduced: holonomy instability config cuts evaluated loops and inflates median angle (see sphere_holonomy_unstable run_id).

## Pythagoras sweeps
| config | run_id | tau_list | fit_rms (min→max) | pyth_med (min→max) | axis_quad/axis_lin @ max |
|---|---|---|---:|---:|---:|
| pythagoras_tau_small.yaml | pythagoras_tau_small_20260202T202933Z_f332 | [1, 2, 4, 8, 16, 32] | 13.46→1.221 | 647.6→0.4687 | 0.05039 |
| pythagoras_tau_large.yaml | pythagoras_tau_large_20260202T202934Z_f96a | [16, 32, 64, 128, 256] | 4.548→0.01874 | 1.194→0.007755 | 0.004723 |
| pythagoras_aliasing_fail.yaml | pythagoras_aliasing_fail_20260202T202954Z_2779 | [64, 128, 256] | 0.03401→0.02004 | 0.01588→3.3445e-04 | 0.1949 |

Failure mode reproduced: aliasing config (small N) shows weak fit improvement and worsening axis_quad_rms with tau (see pythagoras_aliasing_fail run_id).

## Top 5 sensitivity findings
- Grid: increasing tau to 20 inflates stability defect (stab_mean 0.272 → 0.849); run_ids grid_tau_1_20260202T202923Z_5174 and grid_tau_20_20260202T202924Z_8e0f.
- Grid: finer ladders amplify distortion (rescaled 5.598 → 8.708); run_ids grid_levels_coarse_20260202T202927Z_b787 and grid_levels_finer_20260202T202928Z_d009.
- Sphere holonomy: smaller neighborhoods and higher min_overlap reduce evaluated loops (eval 190) and raise median angle (0.839); run_id sphere_holonomy_unstable_20260202T202932Z_1b76.
- Pythagoras: aliasing (N too small) stalls fit improvement (ratio 0.589) and degrades axis_quad_rms with tau; run_id pythagoras_aliasing_fail_20260202T202954Z_2779.
- Pythagoras: large-tau sweep strengthens quadratic axis scaling (axis_quad/axis_lin@max 0.0047); run_id pythagoras_tau_large_20260202T202934Z_f96a.
