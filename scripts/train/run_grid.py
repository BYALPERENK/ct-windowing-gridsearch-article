"""Experiment B: WW x WC grid search for one task and one architecture
(paper Sec. 3.3.2; 352 windows x 5 seeds per task and architecture).

For every seed, window width and window center (in that order), the curated
HU slices are windowed, a model is trained and its best-validation-loss
checkpoint is evaluated on the training, validation and test splits. Each run
writes one result file; runs whose file is already complete are skipped, so
an interrupted search can simply be restarted.

    python scripts/train/run_grid.py --task ischemic --arch resnet50 \
        --data-root data/curated --results-dir results/grid

Hyperparameters come from ctwin.configs.GRID_CONFIGS (paper Table 2).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import torch

from ctwin.configs import GRID_CONFIGS, SEEDS, WC_VALUES, WW_VALUES
from ctwin.data import TASKS, load_split, task_class_names, train_transform
from ctwin.layout import SPLITS
from ctwin.metrics import compute_class_weights
from ctwin.models import ARCHITECTURES
from ctwin.results_io import grid_result_path, is_complete, write_grid_result
from ctwin.training import NUM_CLASSES, amp_setup, train_and_evaluate
from ctwin.windowing import apply_window, window_bounds


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--arch", required=True, choices=sorted(ARCHITECTURES))
    p.add_argument("--data-root", required=True, help="curated data root with train/val/test")
    p.add_argument("--results-dir", default="results/grid")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--ww", type=int, nargs="+", default=list(WW_VALUES), help="window widths (default: full grid)")
    p.add_argument("--wc", type=int, nargs="+", default=list(WC_VALUES), help="window centers (default: full grid)")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--grad-scaler", choices=("config", "fresh", "shared"), default="config",
                   help="AMP GradScaler per run (fresh) or per process (shared); default: as in configs.py")
    p.add_argument("--no-pretrained", action="store_true",
                   help="random initialization instead of ImageNet weights, for quick tests only; "
                        "results go to <results-dir>/no_pretrained")
    args = p.parse_args(argv)
    for name in ("seeds", "ww", "wc"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            p.error(f"--{name} contains duplicates")
    if any(not 0 <= s < 2**32 for s in args.seeds):
        p.error("seeds must be in [0, 2**32)")
    if any(w <= 0 for w in args.ww):
        p.error("window widths must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = GRID_CONFIGS[args.task][args.arch]
    if args.grad_scaler != "config":
        cfg = dataclasses.replace(cfg, fresh_grad_scaler=args.grad_scaler == "fresh")
    results_dir = os.path.join(args.results_dir, "no_pretrained") if args.no_pretrained else args.results_dir
    class_names = task_class_names(args.task)
    torch.backends.cudnn.benchmark = True

    print(f"task={args.task}  arch={args.arch}  device={device}")
    print(f"config: {cfg}")
    print(f"seeds={args.seeds}  WW={args.ww}  WC={args.wc}")

    missing = {
        seed: [(ww, wc) for ww in args.ww for wc in args.wc
               if not is_complete(grid_result_path(results_dir, args.task, args.arch, seed, ww, wc))]
        for seed in args.seeds
    }
    n_total = len(args.seeds) * len(args.ww) * len(args.wc)
    n_missing = sum(len(v) for v in missing.values())
    print(f"runs: {n_total} expected, {n_total - n_missing} complete, {n_missing} to run")
    if n_missing == 0:
        return 0

    label_map = TASKS[args.task]
    raw = {}
    for split in SPLITS:
        print(f"loading {split}")
        raw[split] = load_split(os.path.join(args.data_root, split), label_map)
        print(f"    -> {raw[split][0].shape}")
    tr_lab = raw["train"][1]
    class_weights = compute_class_weights(tr_lab, NUM_CLASSES)
    print(f"class weights: {class_weights.numpy().round(4)}")

    train_tf = train_transform()
    amp_ctx, scaler = amp_setup(device)

    done = 0
    for seed in args.seeds:
        todo = set(missing[seed])
        for ww in args.ww:
            for wc in args.wc:
                if (ww, wc) not in todo:
                    continue
                out_path = grid_result_path(results_dir, args.task, args.arch, seed, ww, wc)
                if is_complete(out_path):  # finished meanwhile by another process
                    continue
                done += 1
                lo, hi = window_bounds(ww, wc)
                print(f"\n### seed {seed} | run {done}/{n_missing} | WW={ww} WC={wc} -> [{lo:.0f}, {hi:.0f}] HU")

                t_start = time.time()
                splits = {s: (apply_window(raw[s][0], ww, wc), raw[s][1]) for s in SPLITS}
                run = train_and_evaluate(
                    args.arch, cfg, seed, splits["train"], splits["val"], splits["test"],
                    class_weights=class_weights, device=device, amp_ctx=amp_ctx, scaler=scaler,
                    train_tf=train_tf, class_names=class_names, pretrained=not args.no_pretrained, start_time=t_start,
                )
                write_grid_result(out_path, args.task, args.arch, cfg, seed, ww, wc, run, class_names)
                print(f"  {run['epochs']} epochs, {run['duration']:.1f} s -> {out_path}")
                del splits
    return 0


if __name__ == "__main__":
    sys.exit(main())
