"""Experiment B figures and Section 4.2 numbers from the grid-search results.

* Figures 6 and 7: heatmaps of the model-averaged test F1 (macro) over the
  WW x WC grid, i.e. the mean over 4 architectures x 5 seeds = 20 runs per
  cell. The clinical window (WW=80, WC=40) is marked with a blue square and
  the optimum with a magenta star. The optimum is selected on the mean
  VALIDATION F1, so the test set is only used for reporting.
* Figures 8 and 9: marginal effect of WW and of WC on test F1. Each point is
  the mean over every run sharing that WW (16 WC x 4 architectures x 5 seeds
  = 320 runs) or that WC (22 x 4 x 5 = 440 runs); error bars are the standard
  deviation over the same runs, not standard errors.
* A text report with the descriptive numbers of Section 4.2 (clinical cell,
  validation-selected optimum, its difference to the clinical window per
  seed, test-set maximum, marginals and per-architecture optima). The
  hypothesis tests of the paper (paired t-tests, multiple-comparison
  corrections) are not computed by this script.

    python scripts/analysis/grid_figures.py \
        --ischemic-dir results/grid/ischemic --hemorrhagic-dir results/grid/hemorrhagic \
        --out-dir figures

Each task folder must contain one folder per architecture, named after the
timm model or as in the original archive (see ctwin.analysis); every one of
the 4 x 5 x 352 result files must be present.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ctwin.analysis import ARCH_ORDER, load_grid
from ctwin.constants import ARCHITECTURES, BRAIN_WINDOW, SEEDS, WC_VALUES, WW_VALUES

WWS = np.array(WW_VALUES)
WCS = np.array(WC_VALUES)
CLIN = BRAIN_WINDOW  # (WW, WC)

TASKS = (  # (task, display name, heatmap file, marginal file)
    ("ischemic", "Ischemic", "fig_f1_heatmap_ischemic.png", "fig_marginal_ischemic_v2.png"),
    ("hemorrhagic", "Hemorrhagic", "fig_f1_heatmap_hemorrhagic.png", "fig_marginal_hemorrhagic_v2.png"),
)


def cell(ww, wc):
    return int(np.where(WWS == ww)[0][0]), int(np.where(WCS == wc)[0][0])


def heatmap(grid, task, path, star, star_f1):
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, ax = plt.subplots(figsize=(9.6, 7.6), dpi=200)
    im = ax.imshow(
        grid, cmap="RdYlGn", origin="lower", aspect="auto",
        extent=[WCS[0] - 5, WCS[-1] + 5, WWS[0] - 5, WWS[-1] + 5],
        interpolation="nearest",
    )
    ax.plot(CLIN[1], CLIN[0], marker="s", markersize=11, color="blue",
            markeredgecolor="white", markeredgewidth=1.2, linestyle="none",
            label=f"Clinical (WW={CLIN[0]}, WC={CLIN[1]})")
    ax.plot(star[1], star[0], marker="*", markersize=22, color="magenta",
            markeredgecolor="white", markeredgewidth=1.2, linestyle="none",
            label=f"Optimal (WW={star[0]}, WC={star[1]}, F1={star_f1:.3f})")
    ax.set_xlabel("Window Center (WC)")
    ax.set_ylabel("Window Width (WW)")
    ax.set_title(f"Normal vs. {task}", fontsize=17, pad=12)
    ax.set_xticks(np.arange(0, 151, 20))
    ax.set_yticks(np.arange(10, 220, 20))
    leg = ax.legend(loc="upper right", fontsize=11, framealpha=0.92,
                    facecolor="#eef4f2", edgecolor="#8fa8a0")
    leg.set_zorder(10)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("Mean Test F1 (macro)", fontsize=13)
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)


def marginal(tf1, axis):
    """Mean and SD (ddof=1) over every run sharing a WW (axis=2) or WC (axis=3) value."""
    flat = np.moveaxis(tf1, axis, 0).reshape(tf1.shape[axis], -1)
    return flat.mean(axis=1), flat.std(axis=1, ddof=1)


def marginal_figure(tf1, task, path, log):
    ww_m, ww_s = marginal(tf1, 2)
    wc_m, wc_s = marginal(tf1, 3)

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=200)

    ax = axes[0]
    ax.errorbar(WWS, ww_m, yerr=ww_s, marker="o", capsize=3, linewidth=1.5,
                markersize=5, color="#1f77b4")
    ax.set_xlabel("Window Width (WW)", fontsize=12)
    ax.set_ylabel("Mean Test F1 (macro)", fontsize=12)
    ax.set_title("Effect of WW — All WC and Seed Averaged", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(WCS, wc_m, yerr=wc_s, marker="s", capsize=3, linewidth=1.5,
                markersize=5, color="darkorange")
    ax.set_xlabel("Window Center (WC)", fontsize=12)
    ax.set_ylabel("Mean Test F1 (macro)", fontsize=12)
    ax.set_title("Effect of WC — All WW and Seed Averaged", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Normal vs. {task}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    log(f"  WW marginal: min {ww_m.min():.4f} @ WW={int(WWS[ww_m.argmin()])}   "
        f"max {ww_m.max():.4f} @ WW={int(WWS[ww_m.argmax()])}   span {ww_m.max() - ww_m.min():.4f}")
    log(f"  WC marginal: min {wc_m.min():.4f} @ WC={int(WCS[wc_m.argmin()])}   "
        f"max {wc_m.max():.4f} @ WC={int(WCS[wc_m.argmax()])}   span {wc_m.max() - wc_m.min():.4f}")
    log(f"  first/last WW point: {ww_m[0]:.4f}+-{ww_s[0]:.4f} / {ww_m[-1]:.4f}+-{ww_s[-1]:.4f}")
    log(f"  first/last WC point: {wc_m[0]:.4f}+-{wc_s[0]:.4f} / {wc_m[-1]:.4f}+-{wc_s[-1]:.4f}")
    d = np.diff(ww_m)
    log(f"  WW curve: {int((d > 0).sum())}/{len(d)} steps increasing"
        + ("  (strictly monotonic)" if (d > 0).all() else ""))


def report(val, tf1, tau, log):
    """Section 4.2 numbers for one task. Returns (grid, optimum (WW, WC), its test F1)."""
    grid = np.mean(tf1, axis=(0, 1))   # model + seed averaged test F1
    vgrid = np.mean(val, axis=(0, 1))  # model + seed averaged validation F1
    agrid = np.mean(tau, axis=(0, 1))  # model + seed averaged test AUROC

    ci, cj = cell(*CLIN)
    log(f"  grid test F1 : min {grid.min():.4f}  max {grid.max():.4f}  "
        f"mean {grid.mean():.4f}  range {grid.max() - grid.min():.4f}")
    log(f"  clinical cell (WW={CLIN[0]}, WC={CLIN[1]}): test F1 {grid[ci, cj]:.4f} "
        f"+-{np.std(np.mean(tf1[:, :, ci, cj], axis=0)):.4f} (SD over seeds, ddof=0)  "
        f"AUROC {agrid[ci, cj]:.4f}")

    bi, bj = np.unravel_index(np.argmax(vgrid), vgrid.shape)
    bww, bwc = int(WWS[bi]), int(WCS[bj])
    seedmean_opt = np.mean(tf1[:, :, bi, bj], axis=0)
    seedmean_clin = np.mean(tf1[:, :, ci, cj], axis=0)
    d = seedmean_opt - seedmean_clin
    log(f"  VAL-selected optimum : WW={bww} WC={bwc}  val F1 {vgrid[bi, bj]:.4f} "
        f"-> test F1 {grid[bi, bj]:.4f} +-{seedmean_opt.std(ddof=1):.4f} (SD over seeds, ddof=1)  "
        f"AUROC {agrid[bi, bj]:.4f}")
    log(f"  delta vs clinical    : {d.mean():+.4f}  (per seed "
        + ", ".join(f"{s}: {v:+.4f}" for s, v in zip(SEEDS, d)) + ")")

    ti, tj = np.unravel_index(np.argmax(grid), grid.shape)
    log(f"  TEST-set maximum     : WW={int(WWS[ti])} WC={int(WCS[tj])} test F1 {grid[ti, tj]:.4f}")

    locs = []
    for s in range(len(SEEDS)):
        g = np.mean(tf1[:, s], axis=0)
        a, b = np.unravel_index(np.argmax(g), g.shape)
        locs.append(f"{SEEDS[s]}: ({int(WWS[a])}, {int(WCS[b])})")
    log(f"  per-seed test argmax (WW, WC): {', '.join(locs)}")

    mww, mwc = grid.mean(axis=1), grid.mean(axis=0)
    log(f"  marginal WW : min {mww.min():.4f} @ WW={int(WWS[mww.argmin()])}  "
        f"max {mww.max():.4f} @ WW={int(WWS[mww.argmax()])}")
    log(f"  marginal WC : min {mwc.min():.4f} @ WC={int(WCS[mwc.argmin()])}  "
        f"max {mwc.max():.4f} @ WC={int(WCS[mwc.argmax()])}")

    log("  per-architecture VAL-selected optima (test F1 / AUROC, mean+-SD over seeds):")
    for m, arch in enumerate(ARCH_ORDER):
        vm = np.mean(val[m], axis=0)
        a, b = np.unravel_index(np.argmax(vm), vm.shape)
        of, oa = tf1[m, :, a, b], tau[m, :, a, b]
        cf, ca = tf1[m, :, ci, cj], tau[m, :, ci, cj]
        log(f"    {ARCHITECTURES[arch]:<16} WW*={int(WWS[a]):<4} WC*={int(WCS[b]):<4} "
            f"opt {of.mean():.4f}+-{of.std(ddof=1):.4f} / {oa.mean():.4f}+-{oa.std(ddof=1):.4f}   "
            f"clin {cf.mean():.4f}+-{cf.std(ddof=1):.4f} / {ca.mean():.4f}+-{ca.std(ddof=1):.4f}   "
            f"dF1 {of.mean() - cf.mean():+.4f}")
    return grid, (bww, bwc), grid[bi, bj]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ischemic-dir", required=True, help="grid-search folder of the ischemic task")
    p.add_argument("--hemorrhagic-dir", required=True, help="grid-search folder of the hemorrhagic task")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    lines: list[str] = []

    def log(s=""):
        print(s)
        lines.append(s)

    dirs = {"ischemic": args.ischemic_dir, "hemorrhagic": args.hemorrhagic_dir}
    for task, name, heat_file, marg_file in TASKS:
        log(f"\n### Normal vs. {name}  ({dirs[task]})")
        data, arch_dirs = load_grid(dirs[task], task,
                                    metrics=(("VAL", "f1_macro"), ("TEST", "f1_macro"), ("TEST", "auroc")))
        val, tf1, tau = data[("VAL", "f1_macro")], data[("TEST", "f1_macro")], data[("TEST", "auroc")]
        log(f"  runs parsed : {tf1.size}  (seeds {list(SEEDS)})")
        for arch, folder in arch_dirs.items():
            log(f"    {ARCHITECTURES[arch]:<16} {folder}")

        grid, star, star_f1 = report(val, tf1, tau, log)
        heatmap(grid, name, os.path.join(args.out_dir, heat_file), star, star_f1)
        log(f"  saved {heat_file}")
        marginal_figure(tf1, name, os.path.join(args.out_dir, marg_file), log)
        log(f"  saved {marg_file}")
        np.save(os.path.join(args.out_dir, f"grid_mean_test_f1_{task}.npy"), grid)

    with open(os.path.join(args.out_dir, "grid_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
