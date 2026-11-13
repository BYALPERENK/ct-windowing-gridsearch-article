"""Statistical analysis of Experiments A and B (paper Sec. 3.6 and 4).

This script was written for the repository to recompute the statistics
reported in the paper from the per-run result files; the original numbers
were computed with ad hoc scripts that are not part of the experiment code.

Experiment A (Table "motivation-stats"): for each task, architecture and
metric (test F1, test AUROC) the paired difference BRAIN - RAW over the five
seeds, its 95% CI, Cohen's d_z, the paired t-test, the Wilcoxon signed-rank
and Shapiro-Wilk p-values, and Benjamini-Hochberg / Bonferroni adjustment
over the family of 16 comparisons.

Experiment B:
* selected windows (Table "per-model-optimal"): for each architecture, and
  for the model-averaged surface, the window with the highest mean
  validation F1, its test F1 / AUROC and those of the clinical window
  (mean +- SD over seeds, ddof=1);
* their comparison with the clinical window (Table "per-model-optimal-stats",
  family of 10);
* the grid-wide family (Table "gridwide-perarch"): every one of the 351
  non-clinical windows against the clinical window, BH-adjusted within each
  task and surface;
* transfer of windows across architectures and across tasks (Table
  "crosstask"), read off those families;
* reproducibility and agreement of the surfaces (Table "surface-agreement"):
  grid range, clinical seed SD, and split-half Spearman correlations averaged
  over the ten 3-vs-2 seed splits;
* further numbers quoted in the text.

The model-averaged surface averages the four architectures within each seed.
Among exactly equal maxima the first window in grid order (smaller WW, then
smaller WC) is selected, as numpy.argmax does. BH results hold within each
family (one per task and surface for the grid-wide analysis), not jointly.

    python scripts/analysis/compute_statistics.py \
        --grid-ischemic-dir results/grid/ischemic --grid-hemorrhagic-dir results/grid/hemorrhagic \
        --expa-ischemic-dir results/experiment_a/ischemic \
        --expa-hemorrhagic-dir results/experiment_a/hemorrhagic --out-dir analysis/statistics
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import combinations

import numpy as np
import scipy
import statsmodels

from ctwin.analysis import ARCH_ORDER, load_experiment_a, load_grid
from ctwin.constants import ARCHITECTURES, BRAIN_WINDOW, SEEDS, WC_VALUES, WW_VALUES
from ctwin.stats import adjust, paired_comparison, spearman

TASKS = ("ischemic", "hemorrhagic")
MA = "model_averaged"
SURFACES = ARCH_ORDER + (MA,)
NAMES = dict(ARCHITECTURES, **{MA: "Model-Averaged"})
WWS, WCS = np.array(WW_VALUES), np.array(WC_VALUES)
CI, CJ = int(np.where(WWS == BRAIN_WINDOW[0])[0][0]), int(np.where(WCS == BRAIN_WINDOW[1])[0][0])
ALPHA = 0.05
# surfaces are means of 4-decimal values; differences below this are floating-point ties
TIE_TOL = 1e-9


def win(i, j):
    return f"{int(WWS[i])}/{int(WCS[j])}"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


class Report:
    def __init__(self, out_dir):
        self.out_dir, self.lines = out_dir, []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)

    def csv(self, name, header, rows):
        with open(os.path.join(self.out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def save(self):
        with open(os.path.join(self.out_dir, "statistics_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


# ------------------------------------------------------------------ Experiment A

def experiment_a(dirs, log):
    log("=" * 100)
    log("EXPERIMENT A - BRAIN vs. RAW, paired over seeds (family of 16)")
    log("=" * 100)
    rows = []
    for task in TASKS:
        records, _ = load_experiment_a(dirs[task])
        by = {(r["arch"], r["condition"], r["seed"]): r for r in records}
        for arch in ARCH_ORDER:
            for metric, label in (("f1_macro", "F1"), ("auroc", "AUROC")):
                raw = [by[(arch, "RAW", s)]["splits"]["TEST"][metric] for s in SEEDS]
                brain = [by[(arch, "BRAIN", s)]["splits"]["TEST"][metric] for s in SEEDS]
                rows.append(dict(task=task, arch=arch, metric=label, **paired_comparison(brain, raw)))
    bh, bonf = adjust([r["p"] for r in rows])
    for r, a, b in zip(rows, bh, bonf):
        r["p_bh"], r["p_bonf"] = float(a), float(b)

    log(f"{'Task':<12} {'Architecture':<16} {'Metric':<6} {'Delta':>7} {'95% CI':>18} {'d_z':>6} "
        f"{'p':>7} {'p_BH':>7} {'p_Bonf':>7} {'p_Wilc':>7} {'p_SW':>6}")
    for r in rows:
        log(f"{r['task']:<12} {NAMES[r['arch']]:<16} {r['metric']:<6} {r['delta']:+7.3f} "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] {r['dz']:+6.2f} {fmt_p(r['p']):>7} "
            f"{fmt_p(r['p_bh']):>7} {fmt_p(r['p_bonf']):>7} {r['p_wilcoxon']:7.4f} {r['p_shapiro']:6.3f}")
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task]
        log(f"  {task}: {sum(r['p_bh'] < ALPHA for r in sub)}/8 survive BH, "
            f"{sum(r['p_bonf'] < ALPHA for r in sub)}/8 survive Bonferroni")
    log(f"  smallest Shapiro-Wilk p in this family: {min(r['p_shapiro'] for r in rows):.3f}")
    log.csv("experiment_a_stats.csv", ["task", "architecture", "metric", "delta", "ci_low", "ci_high", "d_z",
                                       "p", "p_bh", "p_bonf", "p_wilcoxon", "p_shapiro"],
            [[r["task"], r["arch"], r["metric"], r["delta"], r["ci_low"], r["ci_high"], r["dz"], r["p"],
              r["p_bh"], r["p_bonf"], r["p_wilcoxon"], r["p_shapiro"]] for r in rows])
    return rows


# ------------------------------------------------------------------ Experiment B

def load_surfaces(dirs):
    """{task: {"val"|"test"|"auc": {surface: array[seed, ww, wc]}}}"""
    out = {}
    for task in TASKS:
        data, _ = load_grid(dirs[task], task,
                            metrics=(("VAL", "f1_macro"), ("TEST", "f1_macro"), ("TEST", "auroc")))
        out[task] = {}
        for key, (split, metric) in (("val", ("VAL", "f1_macro")), ("test", ("TEST", "f1_macro")),
                                     ("auc", ("TEST", "auroc"))):
            arr = data[(split, metric)]  # [arch, seed, ww, wc]
            surf = {arch: arr[a] for a, arch in enumerate(ARCH_ORDER)}
            surf[MA] = arr.mean(axis=0)
            out[task][key] = surf
    return out


def selected_windows(S, log):
    log("\n" + "=" * 100)
    log("EXPERIMENT B - validation-selected windows vs. the clinical window")
    log("=" * 100)
    sel, rows = {}, []
    for task in TASKS:
        for surf in SURFACES:
            val, test, auc = S[task]["val"][surf], S[task]["test"][surf], S[task]["auc"][surf]
            i, j = np.unravel_index(np.argmax(val.mean(axis=0)), val.shape[1:])
            sel[(task, surf)] = (int(i), int(j))
            of, oa, cf, ca = test[:, i, j], auc[:, i, j], test[:, CI, CJ], auc[:, CI, CJ]
            rows.append(dict(task=task, surf=surf, i=int(i), j=int(j),
                             opt_f1=(of.mean(), of.std(ddof=1)), opt_auc=(oa.mean(), oa.std(ddof=1)),
                             clin_f1=(cf.mean(), cf.std(ddof=1)), clin_auc=(ca.mean(), ca.std(ddof=1)),
                             **paired_comparison(of, cf)))
    bh, bonf = adjust([r["p"] for r in rows])
    for r, a, b in zip(rows, bh, bonf):
        r["p_bh"], r["p_bonf"] = float(a), float(b)

    log(f"{'Task':<12} {'Surface':<16} {'WW*/WC*':>8} | {'opt F1':>15} {'opt AUROC':>15} | "
        f"{'clin F1':>15} {'clin AUROC':>15} | {'dF1':>7}")
    for r in rows:
        f = lambda t: f"{t[0]:.4f}+-{t[1]:.4f}"
        log(f"{r['task']:<12} {NAMES[r['surf']]:<16} {win(r['i'], r['j']):>8} | {f(r['opt_f1']):>15} "
            f"{f(r['opt_auc']):>15} | {f(r['clin_f1']):>15} {f(r['clin_auc']):>15} | {r['delta']:+7.3f}")
    log("\nPaired comparison of the selected windows (family of 10)")
    log(f"{'Task':<12} {'Surface':<16} {'window':>8} {'dF1':>7} {'95% CI':>18} {'d_z':>6} {'p':>7} "
        f"{'p_BH':>7} {'p_Bonf':>7} {'p_Wilc':>7} {'p_SW':>6}")
    for r in rows:
        log(f"{r['task']:<12} {NAMES[r['surf']]:<16} {win(r['i'], r['j']):>8} {r['delta']:+7.3f} "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] {r['dz']:+6.2f} {fmt_p(r['p']):>7} "
            f"{fmt_p(r['p_bh']):>7} {fmt_p(r['p_bonf']):>7} {r['p_wilcoxon']:7.4f} {r['p_shapiro']:6.3f}")
    log(f"  {sum(r['p_bh'] < ALPHA for r in rows)}/10 survive BH, "
        f"{sum(r['p_bonf'] < ALPHA for r in rows)}/10 survive Bonferroni; "
        f"smallest Shapiro-Wilk p: {min(r['p_shapiro'] for r in rows):.3f}")
    log.csv("selected_windows.csv",
            ["task", "surface", "ww", "wc", "opt_f1_mean", "opt_f1_sd", "opt_auroc_mean", "opt_auroc_sd",
             "clin_f1_mean", "clin_f1_sd", "clin_auroc_mean", "clin_auroc_sd", "delta_f1", "ci_low", "ci_high",
             "d_z", "p", "p_bh", "p_bonf", "p_wilcoxon", "p_shapiro"],
            [[r["task"], r["surf"], int(WWS[r["i"]]), int(WCS[r["j"]]), *r["opt_f1"], *r["opt_auc"],
              *r["clin_f1"], *r["clin_auc"], r["delta"], r["ci_low"], r["ci_high"], r["dz"], r["p"],
              r["p_bh"], r["p_bonf"], r["p_wilcoxon"], r["p_shapiro"]] for r in rows])
    return sel


def gridwide(S):
    """Every non-clinical window vs. the clinical one, BH within (task, surface)."""
    fam = {}
    for task in TASKS:
        for surf in SURFACES:
            test = S[task]["test"][surf]
            clin = test[:, CI, CJ]
            cells = [(i, j) for i in range(len(WWS)) for j in range(len(WCS)) if (i, j) != (CI, CJ)]
            comps = [paired_comparison(test[:, i, j], clin) for i, j in cells]
            bh, _ = adjust([c["p"] for c in comps])
            fam[(task, surf)] = {cell: dict(c, p_bh=float(a)) for cell, c, a in zip(cells, comps, bh)}
    return fam


def gridwide_report(fam, sel, log):
    log("\n" + "=" * 100)
    log("EXPERIMENT B - grid-wide family: 351 windows vs. the clinical window, BH per task and surface")
    log("=" * 100)
    log(f"{'Task':<12} {'Surface':<16} {'nominal':>7} {'nom.better':>10} {'better':>6} {'worse':>6} "
        f"{'largest gain':>12} {'at':>7} {'d_z':>6} {'min p_BH (gain)':>15} {'selected p_BH':>13}")
    rows = []
    for (task, surf), f in fam.items():
        better = {c: v for c, v in f.items() if v["p_bh"] < ALPHA and v["delta"] > 0}
        worse = [c for c, v in f.items() if v["p_bh"] < ALPHA and v["delta"] < 0]
        nominal = [c for c, v in f.items() if v["p"] < ALPHA]
        nom_better = [c for c in nominal if f[c]["delta"] > 0]
        gains = [v["p_bh"] for v in f.values() if v["delta"] > 0]
        best = max(sorted(better), key=lambda c: better[c]["delta"]) if better else None
        s = sel[(task, surf)]
        sel_p = f[s]["p_bh"] if s in f else float("nan")
        log(f"{task:<12} {NAMES[surf]:<16} {len(nominal):>7} {len(nom_better):>10} {len(better):>6} "
            f"{len(worse):>6} " + (f"{better[best]['delta']:+12.3f} {win(*best):>7} {better[best]['dz']:+6.2f}"
                                    f" (SD of diff {better[best]['sd']:.4f})"
                                    if best else f"{'---':>12} {'---':>7} {'---':>6}")
            + f" {min(gains) if gains else float('nan'):15.3f} {sel_p:13.3f}")
        rows.append([task, surf, len(nominal), len(nom_better), len(better), len(worse),
                     better[best]["delta"] if best else "", win(*best) if best else "", sel_p])
    log.csv("gridwide.csv", ["task", "surface", "nominal", "nominal_better", "better_bh", "worse_bh",
                             "largest_gain", "largest_gain_window", "selected_window_p_bh"], rows)
    log.csv("gridwide_cells.csv", ["task", "surface", "ww", "wc", "delta_f1", "ci_low", "ci_high", "d_z",
                                   "p", "p_bh", "p_wilcoxon", "p_shapiro"],
            [[t, s, int(WWS[i]), int(WCS[j]), v["delta"], v["ci_low"], v["ci_high"], v["dz"], v["p"],
              v["p_bh"], v["p_wilcoxon"], v["p_shapiro"]]
             for (t, s), f in fam.items() for (i, j), v in sorted(f.items())])
    sw = [v["p_shapiro"] for f in fam.values() for v in f.values()]
    log(f"  Shapiro-Wilk on the grid-wide paired differences: {sum(x <= ALPHA for x in sw)} of {len(sw)} "
        f"have p <= {ALPHA} (not used as a decision rule)")

    # surviving windows per architecture, and whether any survives twice
    log("\nWindows surviving as better on an architecture (not model-averaged):")
    surv = {(t, a): {c for c, v in fam[(t, a)].items() if v["p_bh"] < ALPHA and v["delta"] > 0}
            for t in TASKS for a in ARCH_ORDER}
    for t in TASKS:
        union = set().union(*(surv[(t, a)] for a in ARCH_ORDER))
        shared = [c for c in union if sum(c in surv[(t, a)] for a in ARCH_ORDER) > 1]
        log(f"  {t}: {len(union)} distinct windows, {len(shared)} surviving on two or more architectures")
        for a in ARCH_ORDER:
            cells = sorted(surv[(t, a)])
            if cells:
                low_wc = sum(int(WCS[j]) <= 30 for _, j in cells)
                log(f"    {NAMES[a]:<16} {len(cells):>3}: " + ", ".join(win(*c) for c in cells)
                    + f"   ({low_wc} at WC<=30)")
    u = [set().union(*(surv[(t, a)] for a in ARCH_ORDER)) for t in TASKS]
    log(f"  windows surviving on some architecture in both tasks: {len(u[0] & u[1])}")

    log("\nEach architecture's largest surviving window applied to the other architectures "
        "(delta F1 vs. their clinical window, p_BH in their family):")
    trows = []
    for t in TASKS:
        for a in ARCH_ORDER:
            if not surv[(t, a)]:
                continue
            best = max(sorted(surv[(t, a)]), key=lambda c: fam[(t, a)][c]["delta"])
            parts = [f"{NAMES[b]} {fam[(t, b)][best]['delta']:+.3f} (p_BH {fmt_p(fam[(t, b)][best]['p_bh'])})"
                     for b in ARCH_ORDER if b != a]
            log(f"  {t:<12} {NAMES[a]:<16} {win(*best):>7}: " + "; ".join(parts))
            trows += [[t, a, int(WWS[best[0]]), int(WCS[best[1]]), b, fam[(t, b)][best]["delta"],
                       fam[(t, b)][best]["dz"], fam[(t, b)][best]["p_bh"]] for b in ARCH_ORDER if b != a]
    log.csv("architecture_transfer.csv", ["task", "source_architecture", "ww", "wc", "target_architecture",
                                          "delta_f1", "d_z", "p_bh"], trows)


def crosstask(fam, sel, log):
    log("\n" + "=" * 100)
    log("EXPERIMENT B - cross-task transfer of the selected windows (p_BH from the grid-wide family)")
    log("=" * 100)
    log(f"{'Applied to':<12} {'Selected on other task for':<26} {'WW/WC':>7} {'dF1':>7} {'d_z':>7} {'p_BH':>7}")
    rows = []
    for target, source in (("ischemic", "hemorrhagic"), ("hemorrhagic", "ischemic")):
        for surf in SURFACES:
            cell = sel[(source, surf)]
            v = fam[(target, surf)].get(cell)
            if v is None:  # the transferred window is the clinical window itself
                log(f"{target:<12} {NAMES[surf]:<26} {win(*cell):>7}   (clinical window)")
                continue
            log(f"{target:<12} {NAMES[surf]:<26} {win(*cell):>7} {v['delta']:+7.3f} {v['dz']:+7.2f} "
                f"{fmt_p(v['p_bh']):>7}")
            rows.append([target, surf, int(WWS[cell[0]]), int(WCS[cell[1]]), v["delta"], v["dz"], v["p_bh"]])
    log.csv("crosstask.csv", ["applied_to", "surface", "ww", "wc", "delta_f1", "d_z", "p_bh"], rows)


def agreement(S, log):
    log("\n" + "=" * 100)
    log("EXPERIMENT B - reproducibility and cross-architecture agreement of the test-F1 surfaces")
    log("  rho_self: an architecture against itself on disjoint seed subsets (3 vs 2);")
    log("  rho_cross: against each other architecture on the same subsets, both directions,")
    log("  averaged over the other three; both averaged over all ten 3-vs-2 splits.")
    log("=" * 100)
    idx = {s: k for k, s in enumerate(SEEDS)}
    splits = [([idx[s] for s in c], [idx[s] for s in SEEDS if s not in c]) for c in combinations(SEEDS, 3)]
    rows = []
    for task in TASKS:
        test = S[task]["test"]
        halves = {a: [(test[a][p].mean(axis=0), test[a][q].mean(axis=0)) for p, q in splits] for a in ARCH_ORDER}
        rho_self = {a: np.mean([spearman(h1, h2) for h1, h2 in halves[a]]) for a in ARCH_ORDER}
        pair = {}
        for a, b in combinations(ARCH_ORDER, 2):
            vals = []
            for k in range(len(splits)):
                vals.append(spearman(halves[a][k][0], halves[b][k][1]))
                vals.append(spearman(halves[b][k][0], halves[a][k][1]))
            pair[(a, b)] = pair[(b, a)] = float(np.mean(vals))
        log(f"{'Task':<12} {'Architecture':<16} {'grid range':>10} {'seed SD':>8} {'rho_self':>9} "
            f"{'rho_cross':>9} {'ratio':>6}")
        for a in ARCH_ORDER:
            surf = test[a].mean(axis=0)
            cross = np.mean([pair[(a, b)] for b in ARCH_ORDER if b != a])
            sd = test[a][:, CI, CJ].std(ddof=1)
            ratio = f"{100 * cross / rho_self[a]:5.0f}%" if rho_self[a] > 0 else "   n/a"
            log(f"{task:<12} {NAMES[a]:<16} {surf.max() - surf.min():10.3f} {sd:8.4f} {rho_self[a]:+9.3f} "
                f"{cross:+9.3f} {ratio}")
            rows.append([task, a, surf.max() - surf.min(), sd, rho_self[a], cross])
        ms = np.mean(list(rho_self.values()))
        mc = np.mean([r[5] for r in rows if r[0] == task])
        log(f"{task:<12} {'Mean':<16} {'':>10} {'':>8} {ms:+9.3f} {mc:+9.3f} {100 * mc / ms:5.0f}%")
    log.csv("surface_agreement.csv", ["task", "architecture", "grid_range", "clinical_seed_sd",
                                      "rho_self", "rho_cross"], rows)


def text_numbers(S, sel, log):
    log("\n" + "=" * 100)
    log("EXPERIMENT B - further numbers quoted in the text")
    log("=" * 100)
    for task in TASKS:
        test, val = S[task]["test"][MA], S[task]["val"][MA]
        g = test.mean(axis=0)
        clin = g[CI, CJ]
        above = g[g > clin + TIE_TOL]
        log(f"{task}: model-averaged surface")
        log(f"  cells with mean test F1 >= 0.88: {int((g >= 0.88).sum())} of {g.size}")
        log(f"  clinical window rank: {len(above) + 1} of {g.size}; cells above it exceed it by at most "
            f"{(above.max() - clin) if len(above) else 0:.4f}")
        ti, tj = np.unravel_index(np.argmax(g), g.shape)
        si, sj = sel[(task, MA)]
        c = paired_comparison(test[:, ti, tj], test[:, si, sj])
        log(f"  test-set maximum {win(ti, tj)} ({g[ti, tj]:.4f}) vs. validation-selected {win(si, sj)} "
            f"({g[si, sj]:.4f}): paired t-test p = {c['p']:.3f}")
    # per architecture: test argmax, cost of validation-based selection, flatness of the validation surface
    for task in TASKS:
        for a in ARCH_ORDER:
            val, test = S[task]["val"][a].mean(axis=0), S[task]["test"][a].mean(axis=0)
            si, sj = sel[(task, a)]
            ti, tj = np.unravel_index(np.argmax(test), test.shape)
            order = np.sort(val.ravel())[::-1]
            clin_rank = int((val > val[CI, CJ] + TIE_TOL).sum()) + 1
            log(f"{task} {NAMES[a]}: test argmax {win(ti, tj)} ({test[ti, tj]:.4f}); selected window reaches "
                f"{test[si, sj]:.4f} (shortfall {test[ti, tj] - test[si, sj]:.4f}); validation surface: top ten "
                f"{order[0]:.4f}..{order[9]:.4f} (spread {order[0] - order[9]:.4f}), clinical rank {clin_rank} "
                f"of {val.size}, {order[0] - val[CI, CJ]:.4f} below the argmax")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--grid-ischemic-dir", required=True)
    p.add_argument("--grid-hemorrhagic-dir", required=True)
    p.add_argument("--expa-ischemic-dir", required=True)
    p.add_argument("--expa-hemorrhagic-dir", required=True)
    p.add_argument("--out-dir", default="analysis/statistics")
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    log = Report(args.out_dir)
    log(f"scipy {scipy.__version__}, statsmodels {statsmodels.__version__}, numpy {np.__version__}; "
        f"seeds {list(SEEDS)}; alpha {ALPHA}")

    experiment_a({"ischemic": args.expa_ischemic_dir, "hemorrhagic": args.expa_hemorrhagic_dir}, log)
    S = load_surfaces({"ischemic": args.grid_ischemic_dir, "hemorrhagic": args.grid_hemorrhagic_dir})
    sel = selected_windows(S, log)
    fam = gridwide(S)
    gridwide_report(fam, sel, log)
    crosstask(fam, sel, log)
    agreement(S, log)
    text_numbers(S, sel, log)
    log.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
