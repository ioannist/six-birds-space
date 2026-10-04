"""Generate the figures of the v3 manuscript from committed evidence packs.

Every plotted value is either read from a committed JSON pack under
``docs/notes/`` or recomputed here by a deterministic calculation whose
inputs are stated in the code (the canonical holonomy loops, the lattice
walk probabilities and the gasket drawing).  Figures are written as PDF to
``paper/figures/``.

    .venv/bin/python3 scripts/make_paper_figures_v3.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
NOTES = ROOT / "docs" / "notes"
OUT = ROOT / "paper" / "figures"

# Validated categorical palette (light surface), fixed slot order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _frac(s: str) -> float:
    num, _, den = str(s).partition("/")
    return float(num) / float(den or 1)


def _panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, pad=4)


# ---------------------------------------------------------------------------
# Learned spectral ladders: per-level audit values (corrected canonical runs)
# ---------------------------------------------------------------------------
LEARNED = [
    ("grid_plane", "grid", BLUE, "o"),
    ("sphere_knn", "sphere kNN", ORANGE, "s"),
    ("sierpinski", "gasket", AQUA, "^"),
    ("anisotropic", "gated grid", YELLOW, "D"),
]


def fig_learned_audit() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.15))
    for key, label, color, marker in LEARNED:
        s = _load(NOTES / "math_review_20261003" / key / "artifacts" / "geo_pipeline_summary.json")
        m = [lv["m"] for lv in s["per_level"]]
        axes[0].plot(m, [lv["delta"] for lv in s["per_level"]], color=color, marker=marker, label=label)
        axes[1].plot(m, [lv["stab_max"] for lv in s["per_level"]], color=color, marker=marker, label=label)
        rm_m = [r["fine_m"] for r in s["route_mismatch"]]
        axes[2].plot(rm_m, [r["tv_sup"] for r in s["route_mismatch"]], color=color, marker=marker, label=label)
    titles = [
        "(a) closure defect $\\delta$",
        "(b) worst prototype defect",
        "(c) route mismatch",
    ]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(["4", "8", "16", "32", "64", "128"])
        ax.set_ylim(0, 1.0)
        _panel_label(ax, title)
    axes[0].set_xlabel("macro states $m$")
    axes[1].set_xlabel("macro states $m$")
    axes[2].set_xlabel("finest macro states in triple")
    axes[0].set_ylabel("total-variation defect")
    axes[2].set_xticks([16, 32, 64, 128])
    axes[2].set_xticklabels(["16", "32", "64", "128"])
    axes[2].set_xlim(11, 180)
    axes[2].legend(loc="upper left", handlelength=1.6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_learned_audit.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Learned versus constructed: worst prototype defect against cell size
# ---------------------------------------------------------------------------
KAPPA = 11905 / 16384


def fig_defects_vs_cell() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.7))

    # Learned ladders (hollow markers).
    for key, label, color, marker in LEARNED[:3]:
        s = _load(NOTES / "math_review_20261003" / key / "artifacts" / "geo_pipeline_summary.json")
        n = s["n_micro"]
        cells = [n / lv["m"] for lv in s["per_level"]]
        ax.plot(
            cells,
            [lv["stab_max"] for lv in s["per_level"]],
            color=color,
            marker=marker,
            mfc="white",
            mew=1.2,
            lw=0.9,
            alpha=0.9,
        )

    # Constructed families: measured receipts (filled markers).
    grid_pts = {}
    for ladder in _load(NOTES / "staged_block_lens_review_20261003" / "evidence.json")["finite_ladders"]:
        for lv in ladder["levels"]:
            grid_pts[lv["block_width"] ** 2] = max(grid_pts.get(lv["block_width"] ** 2, 0), lv["escape_max"])
    gx = sorted(grid_pts)
    ax.plot(gx, [grid_pts[c] for c in gx], ls="none", color=BLUE, marker="o")

    gasket_pts = {}
    for ladder in _load(NOTES / "recursive_gasket_review_20261003" / "evidence.json")["finite_ladders"]:
        for lv in ladder["levels"]:
            c = lv["cell_size_before_ownership"]
            gasket_pts[c] = max(gasket_pts.get(c, 0), lv["escape_max"])
    kx = sorted(gasket_pts)
    ax.plot(kx, [gasket_pts[c] for c in kx], ls="none", color=AQUA, marker="^")

    res_pts = {}
    for ladder in _load(NOTES / "curved_reservoir_review_20261003" / "evidence.json")["finite_ladders"]:
        V = ladder["reservoir_volume"]
        for lv in ladder["levels"]:
            if lv["fine_reservoirs_per_fiber"] != 1:
                continue  # the proved bound 5/(8V) is plotted against V; use the finest level only
            res_pts[V] = max(res_pts.get(V, 0), lv["worst_prototype_defect"])
    rx = sorted(res_pts)
    ax.plot(rx, [res_pts[c] for c in rx], ls="none", color=ORANGE, marker="s")

    # Proved upper bounds (thin lines).
    b = np.logspace(math.log10(8), 4, 60)
    ax.plot(b**2, 5 / b, color=BLUE, lw=1.0, ls=(0, (1, 1.5)))
    s_vals = np.arange(3, 20)
    Vs = (3.0 ** (s_vals + 1) + 3) / 2
    ax.plot(Vs, 3 * KAPPA / (Vs - 3), color=AQUA, lw=1.0, ls=(0, (1, 1.5)))
    V = 2.0 ** np.arange(3, 26)
    ax.plot(V, 5 / (8 * V), color=ORANGE, lw=1.0, ls=(0, (1, 1.5)))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 3e7)
    ax.set_ylim(1e-8, 1.4)
    ax.set_xlabel("microstates per macro state (cell size)")
    ax.set_ylabel("worst prototype defect")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=BLUE, marker="o", ls="none", label="grid"),
        Line2D([], [], color=AQUA, marker="^", ls="none", label="gasket"),
        Line2D([], [], color=ORANGE, marker="s", ls="none", label="sphere"),
        Line2D([], [], color=INK2, marker="o", mfc="white", lw=0.9, label="learned spectral ladder"),
        Line2D([], [], color=INK2, marker="o", ls="none", label="constructed lens (measured)"),
        Line2D([], [], color=INK2, lw=1.0, ls=(0, (1, 1.5)), label="constructed lens (proved bound)"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2, handlelength=2.2, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_defects_vs_cell.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Recursive gasket: micro graph coloured by cells, and the cell graph G_3
# ---------------------------------------------------------------------------
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])


def _gasket(level: int):
    """Vertices (positions), edges and address of the level-``level`` gasket."""
    pts: dict[tuple, int] = {}
    pos: list[np.ndarray] = []
    owners: list[tuple] = []
    edges: set[tuple[int, int]] = set()

    def vid(p: np.ndarray, addr: tuple) -> int:
        key = (round(p[0], 9), round(p[1], 9))
        if key not in pts:
            pts[key] = len(pos)
            pos.append(p)
            owners.append(addr)
        else:
            i = pts[key]
            if addr < owners[i]:
                owners[i] = addr
        return pts[key]

    def rec(depth: int, origin: np.ndarray, scale: float, addr: tuple) -> None:
        if depth == 0:
            ids = [vid(origin + scale * c, addr) for c in CORNERS]
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
            return
        for k, c in enumerate(CORNERS):
            rec(depth - 1, origin + scale * c / 2, scale / 2, addr + (k,))

    rec(level, np.zeros(2), 1.0, ())
    return np.array(pos), sorted(edges), owners


def _cell_graph(m: int):
    """Cell contact graph G_m: vertices are length-m addresses."""
    pos, edges, owners = _gasket(m)
    # Use the centroid of each level-m elementary triangle as a cell position.
    centres = {}

    def rec(depth, origin, scale, addr):
        if depth == 0:
            centres[addr] = origin + scale * CORNERS.mean(axis=0)
            return
        for k, c in enumerate(CORNERS):
            rec(depth - 1, origin + scale * c / 2, scale / 2, addr + (k,))

    rec(m, np.zeros(2), 1.0, ())
    addrs = sorted(centres)
    # Two cells are in contact when their elementary triangles share a vertex.
    tri_vertices = {}

    def rec2(depth, origin, scale, addr):
        if depth == 0:
            tri_vertices[addr] = {(round(p[0], 9), round(p[1], 9)) for p in origin + scale * CORNERS}
            return
        for k, c in enumerate(CORNERS):
            rec2(depth - 1, origin + scale * c / 2, scale / 2, addr + (k,))

    rec2(m, np.zeros(2), 1.0, ())
    cedges = []
    for i, a in enumerate(addrs):
        for b in addrs[i + 1 :]:
            if tri_vertices[a] & tri_vertices[b]:
                cedges.append((a, b))
    return centres, cedges


def _bfs(adj: dict, src) -> dict:
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def fig_gasket_cells() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.85))

    # (a) level-6 micro gasket, cells of level 3 (27 cells, 3 shades by first digit).
    pos, edges, owners = _gasket(6)
    shades = {
        0: ["#cde2fb", "#86b6ef", "#3987e5"],
        1: ["#c6eedd", "#6fd0aa", "#1baf7a"],
        2: ["#fbe3bf", "#f5c56b", "#eda100"],
    }
    colors = [shades[a[0]][(a[1] + a[2]) % 3] for a in owners]
    ax = axes[0]
    seg = np.array([[pos[i], pos[j]] for i, j in edges])
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(seg, colors=GRID, linewidths=0.35, zorder=1))
    ax.scatter(pos[:, 0], pos[:, 1], s=2.2, c=colors, linewidths=0, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    _panel_label(ax, "(a) level-6 gasket, 1095 microstates in 27 cells")

    # (b) cell contact graph G_3 with the four-point witness.
    centres, cedges = _cell_graph(3)
    ax = axes[1]
    for a, b in cedges:
        p, q = centres[a], centres[b]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.8, zorder=1)
    xy = np.array([centres[a] for a in sorted(centres)])
    ax.scatter(xy[:, 0], xy[:, 1], s=14, color="white", edgecolors=INK2, linewidths=0.8, zorder=2)
    witness = {"A": (0, 1, 1), "B": (2, 0, 1), "C": (0, 1, 2), "D": (1, 2, 2)}
    adj = {a: set() for a in centres}
    for a, b in cedges:
        adj[a].add(b)
        adj[b].add(a)
    names = list(witness)
    dist = {n: _bfs(adj, witness[n]) for n in names}
    expect = {("A", "B"): 5, ("A", "C"): 1, ("A", "D"): 4, ("B", "C"): 4, ("B", "D"): 3, ("C", "D"): 5}
    for (u, v), d in expect.items():
        assert dist[u][witness[v]] == d, (u, v, dist[u][witness[v]])
    offsets = {"A": (-0.055, 0.0), "B": (0.05, 0.0), "C": (-0.06, 0.0), "D": (0.05, 0.0)}
    for n, addr in witness.items():
        p = centres[addr]
        ax.scatter([p[0]], [p[1]], s=34, color=ORANGE, edgecolors="white", linewidths=1.2, zorder=3)
        dx, dy = offsets[n]
        ax.text(p[0] + dx, p[1] + dy, n, ha="center", va="center", color=INK, fontsize=8, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    _panel_label(ax, "(b) cell graph $G_3$ and the witness $A,B,C,D$")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT / "fig_gasket_cells.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Holonomy: canonical finite loops and the exact spherical refinement study
# ---------------------------------------------------------------------------
def _canonical_holonomy_angles():
    from experiments.runners.holonomy_demo import _holonomy_stats, _macro_metric
    from geo_sbt.substrates.grid import grid_2d
    from geo_sbt.substrates.knn import knn_points
    from geo_sbt.substrates.sphere import sphere_points

    cfg = _load(ROOT / "experiments" / "configs" / "holonomy_demo.yaml")
    mac, hol = cfg["macro"], cfg["holonomy"]
    common = dict(levels=list(mac["levels"]), n_eigs=int(mac["n_eigs"]), seed=int(mac["seed"]), tau=int(mac["tau"]))
    kw = dict(
        k_neigh=int(hol["k_neigh"]),
        k_loop=int(hol["k_loop"]),
        expand_hops=int(hol["expand_hops"]),
        min_overlap=int(hol["min_overlap"]),
        max_loops=int(hol["max_loops"]),
    )
    pl, sp = cfg["plane"], cfg["sphere"]
    P_plane = grid_2d(n_side=int(pl["n_side"]), lazy=float(pl["lazy"]))
    plane = _holonomy_stats(_macro_metric(P_plane, **common), seed=int(hol["seed_plane"]), **kw)
    pts = sphere_points(int(sp["n_points"]), rng=np.random.default_rng(int(cfg.get("seed", 0))))
    P_sphere = knn_points(
        pts, k=int(sp["knn_k"]), sigma=float(sp["sigma"]), self_loop=float(sp.get("self_loop", 1e-6)), symmetrize=True
    )
    sphere = _holonomy_stats(_macro_metric(P_sphere, **common), seed=int(hol["seed_sphere"]), **kw)
    return plane["angles"], sphere["angles"]


def fig_holonomy() -> None:
    plane, sphere = _canonical_holonomy_angles()
    ref = _load(NOTES / "math_review_20261003" / "holonomy_demo" / "artifacts" / "holonomy_demo_summary.json")
    assert plane.size == ref["plane"]["stats"]["triangles_evaluated"]
    assert sphere.size == ref["sphere"]["stats"]["triangles_evaluated"]
    assert abs(np.median(plane) - ref["plane"]["stats"]["median_angle"]) < 1e-12
    assert abs(np.median(sphere) - ref["sphere"]["stats"]["median_angle"]) < 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.35))
    ax = axes[0]
    floor = 1e-6  # angles below this are numerically zero; shown in the leftmost bin
    n_small = {"grid": int(np.sum(plane < floor)), "sphere": int(np.sum(sphere < floor))}
    print("holonomy angles below 1e-6:", n_small)
    top = np.nextafter(max(plane.max(), sphere.max()) * 1.001, np.inf)
    bins = np.logspace(math.log10(floor) - 0.15, math.log10(top), 41)
    ax.hist(np.clip(plane, floor, None), bins=bins, color=BLUE, alpha=0.75, label=f"grid ({plane.size} loops)", edgecolor="white", linewidth=0.4)
    ax.hist(np.clip(sphere, floor, None), bins=bins, color=ORANGE, alpha=0.75, label=f"sphere kNN ({sphere.size} loops)", edgecolor="white", linewidth=0.4)
    ax.text(floor * 1.25, n_small["grid"] + 3, "$\\leq10^{-6}$", ha="left", va="bottom", color=INK2, fontsize=6.5)
    for arr, c in ((plane, BLUE), (sphere, ORANGE)):
        ax.axvline(np.median(arr), color=c, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("loop rotation angle (rad)")
    ax.set_ylabel("loops")
    ax.legend(loc="upper left")
    _panel_label(ax, "(a) canonical learned metrics (medians marked)")

    ax = axes[1]
    est = _load(NOTES / "holonomy_estimator_review_20261003" / "evidence.json")
    hs = [c["h"] for c in est["refinement_cases"]]
    mds = [c["mds_angle_divided_by_area"] for c in est["refinement_cases"]]
    sph = [c["spherical_transport_angle_divided_by_area"] for c in est["refinement_cases"]]
    lim = _frac(est["exact_leading_certificate"]["absolute_area_normalized_limit"])
    ax.axhline(lim, color=BLUE, lw=0.8, ls=(0, (1, 1.5)))
    ax.axhline(1.0, color=AQUA, lw=0.8, ls=(0, (1, 1.5)))
    ax.plot(hs, mds, color=BLUE, marker="o", label="local MDS + Procrustes")
    ax.plot(hs, sph, color=AQUA, marker="^", label="landmark transport")
    ax.plot(hs, [0.0] * len(hs), color=INK2, marker="s", ms=3.5, label="local MDS, identical charts")
    ax.text(hs[2], lim + 0.06, f"exact limit {lim:.4f}", ha="center", va="bottom", color=INK2, fontsize=7)
    ax.text(hs[2], 1.0 - 0.06, "true value 1", ha="center", va="top", color=INK2, fontsize=7)
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    ax.set_ylim(-0.08, 1.5)
    ax.set_xlabel("patch scale $h$ (exact unit-sphere distances)")
    ax.set_ylabel("loop angle / loop area")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
    _panel_label(ax, "(b) area-normalised holonomy under refinement")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_holonomy.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quadratic accounting: lattice walk probabilities computed by convolution
# ---------------------------------------------------------------------------
ISO_STEPS = [((0, 0), 0.5), ((1, 0), 0.125), ((-1, 0), 0.125), ((0, 1), 0.125), ((0, -1), 0.125)]
COR_STEPS = [
    ((0, 0), 0.5),
    ((1, 0), 1 / 16),
    ((-1, 0), 1 / 16),
    ((0, 1), 1 / 16),
    ((0, -1), 1 / 16),
    ((1, 1), 1 / 8),
    ((-1, -1), 1 / 8),
]


def _walk(steps, n_max: int, checkpoints: set[int]) -> dict[int, np.ndarray]:
    """Exact-support n-step lattice probabilities by nonnegative convolution."""
    L = n_max
    p = np.zeros((2 * L + 1, 2 * L + 1))
    p[L, L] = 1.0
    out = {}
    for n in range(1, n_max + 1):
        q = np.zeros_like(p)
        for (dx, dy), w in steps:
            q += w * np.roll(np.roll(p, dx, axis=0), dy, axis=1)
        p = q
        if n in checkpoints:
            out[n] = p.copy()
    return out


def _centered_cost(p: np.ndarray, x: int, y: int) -> float:
    L = (p.shape[0] - 1) // 2
    return math.log(p[L, L] / p[L + x, L + y])


def fig_quadratic() -> None:
    stages = [16, 32, 64, 128, 256, 512]
    iso = _walk(ISO_STEPS, max(stages), set(stages) | {4, 8})
    cor = _walk(COR_STEPS, max(stages), set(stages))

    # Check the floating convolution against the exact integer counts of the certificate.
    cert = _load(NOTES / "refinement_review_20261003" / "quadratic_walk_certificate.json")
    for row in cert["rows"]:
        tau = int(row["tau"])
        p = iso[tau]
        L = (p.shape[0] - 1) // 2
        for x, y, count in row["counts"]:
            exact = count / 8.0**tau
            assert abs(p[L + x, L + y] - exact) <= 1e-9 * exact, (tau, x, y)

    fig = plt.figure(figsize=(6.5, 4.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.95], hspace=0.55, wspace=1.1)
    axes_top = [fig.add_subplot(gs[0, 2 * i : 2 * i + 2]) for i in range(3)]
    ax_cert = fig.add_subplot(gs[1, 0:3])
    ax_res = fig.add_subplot(gs[1, 3:6])

    def cost_map(p, n, half):
        L = (p.shape[0] - 1) // 2
        F = np.full((2 * half + 1, 2 * half + 1), np.nan)
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                if p[L + i, L + j] > 0:
                    F[i + half, j + half] = math.log(p[L, L] / p[L + i, L + j])
        return F

    panels = [
        (iso[4], 4, "(c1) isotropic walk, $\\tau=4$"),
        (iso[128], 128, "(c2) isotropic walk, $\\tau=128$"),
        (cor[128], 128, "(c3) correlated walk, $\\tau=128$"),
    ]
    vmax = 8.0
    for ax, (p, n, title) in zip(axes_top, panels):
        half = int(math.ceil(2.0 * math.sqrt(n)))
        half = min(half, n)
        F = cost_map(p, n, half)
        ext = (half + 0.5) / math.sqrt(n)
        ax.imshow(F.T, origin="lower", extent=(-ext, ext, -ext, ext), cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest")
        th = np.linspace(0, 2 * math.pi, 200)
        for level in (1, 2, 4, 6):
            r = math.sqrt(level / 2)  # 2|u|^2 = level in diffusion units
            ax.plot(r * np.cos(th), r * np.sin(th), color="white", lw=0.8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_xlabel("$x/\\sqrt{\\tau}$")
        if ax is axes_top[0]:
            ax.set_ylabel("$y/\\sqrt{\\tau}$")
        _panel_label(ax, title)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes_top, fraction=0.025, pad=0.02)
    cb.set_label("centred cost $F_\\tau$", color=INK)
    cb.outline.set_visible(False)

    # (d) exact certificate: uniform centred-cost error on the declared windows.
    taus = [int(r["tau"]) for r in cert["rows"]]
    deltas = [_frac(r["uniform_cost_error"]) for r in cert["rows"]]
    ax_cert.plot(taus, deltas, color=BLUE, marker="o", label="certified error $\\delta_\\tau$")
    ax_cert.plot(taus, [math.sqrt(d / 2) for d in deltas], color=AQUA, marker="^", label="readout error $\\sqrt{\\delta_\\tau/2}$")
    ax_cert.set_xscale("log", base=2)
    ax_cert.set_xticks(taus)
    ax_cert.set_xticklabels([str(t) for t in taus])
    ax_cert.set_ylim(0, 0.8)
    ax_cert.set_xlabel("stage $\\tau$")
    ax_cert.set_ylabel("uniform error on window")
    ax_cert.legend(loc="upper right")
    _panel_label(ax_cert, "(d) exact certificate, $|x|,|y|\\leq\\lfloor\\sqrt{\\tau}\\rfloor$")

    # (e) axis residual at x = y = floor(sqrt(n)).
    def residual(p, n):
        k = int(math.isqrt(n))
        return _centered_cost(p, k, k) - _centered_cost(p, k, 0) - _centered_cost(p, 0, k)

    ax_res.axhline(0.0, color=BLUE, lw=0.8, ls=(0, (1, 1.5)))
    ax_res.axhline(-16 / 5, color=ORANGE, lw=0.8, ls=(0, (1, 1.5)))
    ax_res.plot(stages, [residual(iso[n], n) for n in stages], color=BLUE, marker="o", label="isotropic walk")
    ax_res.plot(stages, [residual(cor[n], n) for n in stages], color=ORANGE, marker="s", label="correlated walk")
    ax_res.text(stages[-1], -16 / 5 + 0.12, "limit $-16/5$", ha="right", va="bottom", color=INK2, fontsize=7)
    ax_res.set_xscale("log", base=2)
    ax_res.set_xticks(stages)
    ax_res.set_xticklabels([str(s) for s in stages])
    ax_res.set_ylim(-3.8, 0.6)
    ax_res.set_xlabel("stage $n$")
    ax_res.set_ylabel("axis residual $R_n(k,k)$")
    ax_res.legend(loc="center right")
    _panel_label(ax_res, "(e) axis residual at $k=\\lfloor\\sqrt{n}\\rfloor$")

    fig.savefig(OUT / "fig_quadratic.pdf")
    plt.close(fig)
    return {
        "iso": {n: residual(iso[n], n) for n in stages},
        "cor": {n: residual(cor[n], n) for n in stages},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_learned_audit()
    fig_defects_vs_cell()
    fig_gasket_cells()
    fig_holonomy()
    res = fig_quadratic()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
