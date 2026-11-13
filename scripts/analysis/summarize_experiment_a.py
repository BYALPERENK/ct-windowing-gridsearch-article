"""Descriptive summary of Experiment A (RAW vs. BRAIN) for both tasks.

For each task it prints and saves:
* every run (architecture x condition x seed): epochs, duration and the
  TRAIN / VAL / TEST metrics,
* mean +- SD over the five seeds for every architecture, condition and split
  (SD with ddof=1, as in the paper),
* the BRAIN - RAW difference of the mean test metrics,
* rankings of the architecture/condition pairs by mean test metric,
* the test confusion matrices,
and, over both tasks, the rows of the Experiment A table of the paper (test
F1 and AUROC, mean +- SD, and their differences). Every architecture and
condition must have exactly one result file per seed. The paired tests of
the paper are not computed by this script.

    python scripts/analysis/summarize_experiment_a.py \
        --ischemic-dir results/experiment_a/ischemic \
        --hemorrhagic-dir results/experiment_a/hemorrhagic --out-dir analysis/experiment_a
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

from ctwin.analysis import ARCH_ORDER, load_experiment_a
from ctwin.constants import ARCHITECTURES, SEEDS

METRICS = ("acc", "f1_macro", "f1_weighted", "sensitivity", "specificity", "auroc")
LABELS = {"acc": "Accuracy (%)", "f1_macro": "F1 macro", "f1_weighted": "F1 weighted",
          "sensitivity": "Sensitivity", "specificity": "Specificity", "auroc": "AUROC"}
SPLITS = ("TRAIN", "VAL", "TEST")
CONDITIONS = ("RAW", "BRAIN")


def _key(r):
    return (ARCH_ORDER.index(r["arch"]), CONDITIONS.index(r["condition"]), r["seed"])


def aggregate(records):
    """{(arch, condition): {split: {metric: dict(mean, std, min, max, n, values)}}}"""
    groups = defaultdict(list)
    for r in records:
        groups[(r["arch"], r["condition"])].append(r)
    agg = {}
    for key, recs in groups.items():
        agg[key] = {}
        for split in SPLITS:
            agg[key][split] = {}
            for metric in METRICS:
                vals = [r["splits"][split][metric] for r in recs
                        if split in r["splits"] and r["splits"][split].get(metric) is not None]
                if vals:
                    agg[key][split][metric] = dict(
                        mean=float(np.mean(vals)),
                        std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        min=float(np.min(vals)), max=float(np.max(vals)), n=len(vals), values=vals)
    return agg


def ordered_keys(agg):
    return sorted(agg, key=lambda k: (ARCH_ORDER.index(k[0]), CONDITIONS.index(k[1])))


def summarize_task(task, records, folders, out_dir, log):
    log("=" * 110)
    log(f"  EXPERIMENT A - {task.upper()}  ({len(records)} runs, seeds {list(SEEDS)})")
    log("=" * 110)
    for arch, folder in folders.items():
        log(f"  {ARCHITECTURES[arch]:<16} {folder}")

    log("\nTraining summary")
    log(f"{'Architecture':<16} {'Cond':<6} {'Seed':>5} | {'Epochs':>6} {'Max':>4} {'Duration (s)':>13}")
    for r in sorted(records, key=_key):
        log(f"{ARCHITECTURES[r['arch']]:<16} {r['condition']:<6} {r['seed']:>5} | "
            f"{r['epochs']!s:>6} {r['max_epochs']!s:>4} {r['duration_s'] or 0:>13.1f}")

    log("\nIndividual runs")
    log(f"{'Architecture':<16} {'Cond':<6} {'Seed':>5} | {'Split':<5} {'Acc%':>7} {'F1m':>7} "
        f"{'F1w':>7} {'Sens':>7} {'Spec':>7} {'AUROC':>7}")
    for r in sorted(records, key=_key):
        for i, split in enumerate(SPLITS):
            m = r["splits"].get(split, {})
            head = f"{ARCHITECTURES[r['arch']]:<16} {r['condition']:<6} {r['seed']:>5}" if i == 0 else " " * 29
            log(f"{head} | {split:<5} {m.get('acc', np.nan):>7.2f} {m.get('f1_macro', np.nan):>7.4f} "
                f"{m.get('f1_weighted', np.nan):>7.4f} {m.get('sensitivity', np.nan):>7.4f} "
                f"{m.get('specificity', np.nan):>7.4f} {m.get('auroc', np.nan):>7.4f}")

    agg = aggregate(records)
    for split in SPLITS:
        log(f"\nMean +- SD over seeds - {split}")
        log(f"{'Architecture':<16} {'Cond':<6} {'n':>2} | " + " ".join(f"{LABELS[m]:>15}" for m in METRICS))
        for key in ordered_keys(agg):
            sp = agg[key][split]
            cells = []
            for m in METRICS:
                d = sp.get(m)
                if d is None:
                    cells.append(f"{'-':>15}")
                elif m == "acc":
                    cells.append(f"{d['mean']:>8.2f}+-{d['std']:.2f}".rjust(15))
                else:
                    cells.append(f"{d['mean']:.4f}+-{d['std']:.4f}".rjust(15))
            n = sp.get("f1_macro", {}).get("n", 0)
            log(f"{ARCHITECTURES[key[0]]:<16} {key[1]:<6} {n:>2} | " + " ".join(cells))

    log("\nBRAIN - RAW, mean test metrics (positive: BRAIN better)")
    log(f"{'Architecture':<16} | " + " ".join(f"{'d ' + LABELS[m]:>17}" for m in METRICS))
    for arch in ARCH_ORDER:
        if (arch, "RAW") not in agg or (arch, "BRAIN") not in agg:
            continue
        raw, brain = agg[(arch, "RAW")]["TEST"], agg[(arch, "BRAIN")]["TEST"]
        cells = []
        for m in METRICS:
            if m in raw and m in brain:
                d = brain[m]["mean"] - raw[m]["mean"]
                cells.append(f"{d:+17.2f}" if m == "acc" else f"{d:+17.4f}")
            else:
                cells.append(f"{'-':>17}")
        log(f"{ARCHITECTURES[arch]:<16} | " + " ".join(cells))

    for metric in ("f1_macro", "auroc", "acc", "sensitivity"):
        log(f"\nRanking by mean test {LABELS[metric]}")
        ranking = sorted(((k, agg[k]["TEST"][metric]) for k in agg if metric in agg[k]["TEST"]),
                         key=lambda kv: kv[1]["mean"], reverse=True)
        for i, ((arch, cond), d) in enumerate(ranking, 1):
            log(f"{i:>3}. {ARCHITECTURES[arch]:<16} {cond:<6} {d['mean']:.4f} +- {d['std']:.4f} (n={d['n']})")

    log("\nTest confusion matrices (rows: true, columns: predicted)")
    for r in sorted(records, key=_key):
        cm = r["splits"].get("TEST", {}).get("cm")
        if cm:
            log(f"  {ARCHITECTURES[r['arch']]} / {r['condition']} / seed {r['seed']}: {cm}")

    with open(os.path.join(out_dir, f"{task}_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "condition", "seed", "split", *METRICS, "epochs", "duration_s"])
        for r in sorted(records, key=_key):
            for split in SPLITS:
                m = r["splits"].get(split, {})
                w.writerow([r["arch"], r["condition"], r["seed"], split,
                            *[m.get(k, "") for k in METRICS], r["epochs"], r["duration_s"]])
    with open(os.path.join(out_dir, f"{task}_mean_sd.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "condition", "split", "n",
                    *[f"{m}_{s}" for m in METRICS for s in ("mean", "sd")]])
        for key in ordered_keys(agg):
            for split in SPLITS:
                sp = agg[key][split]
                row = [key[0], key[1], split, sp.get("f1_macro", {}).get("n", 0)]
                for m in METRICS:
                    d = sp.get(m)
                    row += [f"{d['mean']:.6f}", f"{d['std']:.6f}"] if d else ["", ""]
                w.writerow(row)
    return agg


def paper_table(aggs, log):
    log("\n" + "=" * 110)
    log("  EXPERIMENT A TABLE: test F1 / AUROC, mean +- SD over five seeds, and BRAIN - RAW")
    log("=" * 110)
    log(f"{'Task':<12} {'Architecture':<16} | {'RAW F1':>13} {'RAW AUROC':>13} | "
        f"{'BRAIN F1':>13} {'BRAIN AUROC':>13} | {'dF1':>7} {'dAUROC':>7}")
    for task, agg in aggs.items():
        for arch in ARCH_ORDER:
            try:
                raw, brain = agg[(arch, "RAW")]["TEST"], agg[(arch, "BRAIN")]["TEST"]
                cells = [f"{x[m]['mean']:.3f}+-{x[m]['std']:.3f}" for x in (raw, brain) for m in ("f1_macro", "auroc")]
                df = brain["f1_macro"]["mean"] - raw["f1_macro"]["mean"]
                da = brain["auroc"]["mean"] - raw["auroc"]["mean"]
            except KeyError:
                log(f"{task:<12} {ARCHITECTURES[arch]:<16} | missing data")
                continue
            log(f"{task:<12} {ARCHITECTURES[arch]:<16} | {cells[0]:>13} {cells[1]:>13} | "
                f"{cells[2]:>13} {cells[3]:>13} | {df:+7.3f} {da:+7.3f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ischemic-dir", required=True, help="Experiment A folder of the ischemic task")
    p.add_argument("--hemorrhagic-dir", required=True, help="Experiment A folder of the hemorrhagic task")
    p.add_argument("--out-dir", default="analysis/experiment_a")
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    lines: list[str] = []

    def log(s=""):
        print(s)
        lines.append(s)

    aggs = {}
    for task, folder in (("ischemic", args.ischemic_dir), ("hemorrhagic", args.hemorrhagic_dir)):
        records, folders = load_experiment_a(folder, conditions=CONDITIONS)
        aggs[task] = summarize_task(task, records, folders, args.out_dir, log)
    paper_table(aggs, log)

    with open(os.path.join(args.out_dir, "experiment_a_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
