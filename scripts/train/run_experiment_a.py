"""Experiment A: RAW input vs. the clinical brain window (WW=80, WC=40)
for one task (paper Sec. 3.3.1; 4 architectures x 2 conditions x 5 seeds).

RAW scales each split to [0, 1] with that split's own global HU minimum and
maximum; BRAIN applies the brain window. Each run writes one result file;
runs whose file is already complete are skipped.

    python scripts/train/run_experiment_a.py --task ischemic \
        --data-root data/curated --results-dir results/experiment_a

Hyperparameters come from ctwin.configs.EXPERIMENT_A_CONFIGS.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import torch

from ctwin.configs import EXPERIMENT_A_CONDITIONS, EXPERIMENT_A_CONFIGS, SEEDS
from ctwin.data import TASKS, load_split, task_class_names, train_transform
from ctwin.layout import SPLITS
from ctwin.metrics import compute_class_weights
from ctwin.results_io import experiment_a_result_path, is_complete, write_experiment_a_result
from ctwin.training import NUM_CLASSES, amp_setup, train_and_evaluate
from ctwin.windowing import apply_window, raw_minmax_normalize


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--data-root", required=True, help="curated data root with train/val/test")
    p.add_argument("--results-dir", default="results/experiment_a")
    p.add_argument("--archs", nargs="+", default=list(EXPERIMENT_A_CONFIGS),
                   choices=list(EXPERIMENT_A_CONFIGS))
    p.add_argument("--conditions", nargs="+", default=list(EXPERIMENT_A_CONDITIONS),
                   choices=list(EXPERIMENT_A_CONDITIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--grad-scaler", choices=("config", "fresh", "shared"), default="config",
                   help="AMP GradScaler per run (fresh) or per process (shared); default: as in configs.py")
    p.add_argument("--no-pretrained", action="store_true",
                   help="random initialization instead of ImageNet weights, for quick tests only; "
                        "results go to <results-dir>/no_pretrained")
    args = p.parse_args(argv)
    for name in ("archs", "conditions", "seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            p.error(f"--{name} contains duplicates")
    if any(not 0 <= s < 2**32 for s in args.seeds):
        p.error("seeds must be in [0, 2**32)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_names = task_class_names(args.task)
    results_dir = os.path.join(args.results_dir, "no_pretrained") if args.no_pretrained else args.results_dir
    torch.backends.cudnn.benchmark = True

    # keep the original training order (model -> condition -> seed) whatever
    # order the options were given in
    archs = [a for a in EXPERIMENT_A_CONFIGS if a in args.archs]
    conditions = [c for c in EXPERIMENT_A_CONDITIONS if c in args.conditions]

    missing = [(a, c, s) for a in archs for c in conditions for s in args.seeds
               if not is_complete(experiment_a_result_path(results_dir, args.task, a, c, s))]
    n_total = len(archs) * len(conditions) * len(args.seeds)
    print(f"task={args.task}  device={device}")
    print(f"runs: {n_total} expected, {n_total - len(missing)} complete, {len(missing)} to run")
    if not missing:
        return 0

    label_map = TASKS[args.task]
    raw = {}
    for split in SPLITS:
        print(f"loading {split}")
        raw[split] = load_split(os.path.join(args.data_root, split), label_map)
    class_weights = compute_class_weights(raw["train"][1], NUM_CLASSES)
    print(f"class weights: {class_weights.numpy().round(4)}")

    train_tf = train_transform()
    amp_ctx, scaler = amp_setup(device)

    # normalized inputs, computed once per needed condition (RAW: per split)
    inputs = {}
    for cond in {c for _, c, _ in missing}:
        window = EXPERIMENT_A_CONDITIONS[cond]
        if window is None:
            inputs[cond] = {s: (raw_minmax_normalize(raw[s][0]), raw[s][1]) for s in SPLITS}
        else:
            inputs[cond] = {s: (apply_window(raw[s][0], *window), raw[s][1]) for s in SPLITS}

    for i, (arch, cond, seed) in enumerate(missing, 1):
        out_path = experiment_a_result_path(results_dir, args.task, arch, cond, seed)
        if is_complete(out_path):
            continue
        cfg = EXPERIMENT_A_CONFIGS[arch]
        if args.grad_scaler != "config":
            cfg = dataclasses.replace(cfg, fresh_grad_scaler=args.grad_scaler == "fresh")
        print(f"\n### run {i}/{len(missing)} | {arch} | {cond} | seed {seed}")
        data = inputs[cond]
        run = train_and_evaluate(
            arch, cfg, seed, data["train"], data["val"], data["test"],
            class_weights=class_weights, device=device, amp_ctx=amp_ctx, scaler=scaler,
            train_tf=train_tf, class_names=class_names, pretrained=not args.no_pretrained,
        )
        write_experiment_a_result(out_path, args.task, arch, cfg, seed, cond,
                                  EXPERIMENT_A_CONDITIONS[cond], run, class_names)
        print(f"  {run['epochs']} epochs, {run['duration']:.1f} s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
