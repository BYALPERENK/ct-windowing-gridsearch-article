"""Collecting per-run result files into arrays for analysis (no PyTorch needed).

A task folder holds one sub-folder per architecture:

    grid search:   <task_dir>/<arch>/seed_<seed>/<WW>_<WC>_windowing.txt
    Experiment A:  <task_dir>/<arch>/<CONDITION>_seed_<seed>.txt

The architecture folders written by scripts/train/ are named after the timm
model (resnet50, ...). The grid-search folders of the original experiments
used other names, which are recognized as well (ORIGINAL_GRID_DIRS).
"""

from __future__ import annotations

import math
import os

import numpy as np

from .constants import ARCHITECTURES, SEEDS, WC_VALUES, WW_VALUES
from .results_io import parse_result_file

ARCH_ORDER = tuple(ARCHITECTURES)  # order used in every table and figure of the paper

# Folder names of the original grid-search archive. Its ischemic folder also
# holds a superseded first run under the plain names; the run reported in the
# paper is the re-run with the suffix "_yenilenmis" ("renewed").
_ORIGINAL_NAMES = {"resnet50": "Resnet-50", "densenet121": "DenseNet-121",
                   "efficientnet_b0": "EfficientNet-B0", "mobilevitv2_175": "MobileVitV2-175"}
ORIGINAL_GRID_DIRS: dict[str, dict[str, str]] = {
    "ischemic": {a: f"{n}_yenilenmis" for a, n in _ORIGINAL_NAMES.items()},
    "hemorrhagic": dict(_ORIGINAL_NAMES),
}


def resolve_arch_dirs(task_dir: str, archs=ARCH_ORDER, task: str | None = None) -> dict[str, str]:
    """Map each architecture to its folder under task_dir.

    The folder is <arch> (repository layout) or, when task is given, the
    folder name of the original archive for that task. Exactly one of them
    must exist, so that runs of different experiments are never mixed.
    """
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"task folder not found: {task_dir}")
    out = {}
    for arch in archs:
        names = [arch] + ([ORIGINAL_GRID_DIRS[task][arch]] if task else [])
        found = [n for n in names if os.path.isdir(os.path.join(task_dir, n))]
        if len(found) != 1:
            raise FileNotFoundError(
                f"{task_dir}: expected exactly one folder for {arch} among {names}, found {found or 'none'}")
        out[arch] = os.path.join(task_dir, found[0])
    return out


def load_grid(task_dir: str, task: str, metrics=(("TEST", "f1_macro"),), archs=ARCH_ORDER, seeds=SEEDS,
              ww_values=WW_VALUES, wc_values=WC_VALUES):
    """Read a complete grid-search task folder.

    Returns ({(split, metric): array[arch, seed, ww, wc]}, {arch: folder}).
    Every result file must exist and every requested value must be finite.
    """
    dirs = resolve_arch_dirs(task_dir, archs, task)
    shape = (len(archs), len(seeds), len(ww_values), len(wc_values))
    out = {key: np.full(shape, np.nan) for key in metrics}
    for a, arch in enumerate(archs):
        for s, seed in enumerate(seeds):
            base = os.path.join(dirs[arch], f"seed_{seed}")
            for i, ww in enumerate(ww_values):
                for j, wc in enumerate(wc_values):
                    path = os.path.join(base, f"{ww}_{wc}_windowing.txt")
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"missing result file: {path}")
                    splits = parse_result_file(path)["splits"]
                    for split, metric in metrics:
                        value = splits.get(split, {}).get(metric)
                        if value is None or not math.isfinite(value):
                            raise ValueError(f"{path}: {split} {metric} is missing or not finite ({value})")
                        out[(split, metric)][a, s, i, j] = value
    return out, dirs


def load_experiment_a(task_dir: str, archs=ARCH_ORDER, seeds=SEEDS,
                      conditions=("RAW", "BRAIN"), required=(("TEST", "f1_macro"), ("TEST", "auroc"))):
    """Parse the Experiment A result files of a task folder.

    Exactly one file per architecture, condition and seed must be present,
    with finite values for the required (split, metric) pairs, so that every
    summary is taken over the same seeds. Returns (records, {arch: folder});
    each record is the parsed file plus "arch" and "path".
    """
    dirs = resolve_arch_dirs(task_dir, archs)
    records = []
    for arch, folder in dirs.items():
        for cond in conditions:
            for seed in seeds:
                path = os.path.join(folder, f"{cond}_seed_{seed}.txt")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"missing result file: {path}")
                rec = parse_result_file(path)
                if rec["condition"] != cond or rec["seed"] != seed:
                    raise ValueError(f"{path}: file says condition={rec['condition']}, seed={rec['seed']}")
                for split, metric in required:
                    value = rec["splits"].get(split, {}).get(metric)
                    if value is None or not math.isfinite(value):
                        raise ValueError(f"{path}: {split} {metric} is missing or not finite ({value})")
                rec["arch"] = arch
                rec["path"] = path
                records.append(rec)
    return records, dirs
