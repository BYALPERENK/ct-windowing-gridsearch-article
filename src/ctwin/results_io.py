"""Per-run result files: writing, completeness check and parsing.

Every run writes one text file with its configuration and the TRAIN / VAL /
TEST metrics of its best checkpoint:

    grid search:   <results>/<task>/<arch>/seed_<seed>/<WW>_<WC>_windowing.txt
    Experiment A:  <results>/<task>/<arch>/<CONDITION>_seed_<seed>.txt

A run whose file exists and is complete is skipped when a script is
restarted. The check looks at the file only; it does not verify that the
file was produced with the current configuration. The parser also reads the
files written by the original scripts, whose labels were in Turkish.
"""

from __future__ import annotations

import math
import os
import re

from .metrics import format_cm
from .windowing import window_bounds

CLASS_DESCRIPTIONS = {
    "ischemic": "0 (normal) vs 1 (ischemic)",
    "hemorrhagic": "0 (normal) vs 2 (hemorrhagic)",
}

SPLIT_TAGS = ("TRAIN", "VAL", "TEST")
_RULE = "=" * 60

# section headers: (new, original)
_FIRST_SECTIONS = ("WINDOWING PARAMETERS", "CONDITION", "WINDOWING PARAMETRELERİ", "KOŞUL BİLGİSİ")
_TRAINING_SECTIONS = ("TRAINING", "EĞİTİM BİLGİSİ")
_BODY_MARKERS = ("--- TRAIN ---", "--- VAL ---", "--- TEST ---", "Confusion Matrix")


def grid_result_path(results_dir: str, task: str, arch: str, seed: int, ww: int, wc: int) -> str:
    return os.path.join(results_dir, task, arch, f"seed_{seed}", f"{ww}_{wc}_windowing.txt")


def experiment_a_result_path(results_dir: str, task: str, arch: str, condition: str, seed: int) -> str:
    return os.path.join(results_dir, task, arch, f"{condition}_seed_{seed}.txt")


def is_complete(path: str) -> bool:
    """True if path is a result file that was written to the end: both header
    sections are present and each of TRAIN / VAL / TEST has all six metrics
    (AUROC may be nan) and a 2x2 confusion matrix."""
    if not os.path.isfile(path) or os.path.getsize(path) < 50:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return False
    if not (any(s in content for s in _FIRST_SECTIONS)
            and any(s in content for s in _TRAINING_SECTIONS)
            and all(m in content for m in _BODY_MARKERS)):
        return False
    splits = parse_result_text(content)["splits"]
    for tag in SPLIT_TAGS:
        m = splits.get(tag)
        if m is None or any(m[k] is None for k in _METRIC_KEYS):
            return False
        if m["cm"] is None or len(m["cm"]) != 2 or any(len(r) != 2 for r in m["cm"]):
            return False
    return True


def _metric_lines(results: dict, class_names) -> list[str]:
    lines = []
    for tag in SPLIT_TAGS:
        m = results[tag]
        lines += [
            "",
            f"--- {tag} ---",
            f"  Accuracy       : {m['acc']:.2f}%",
            f"  F1 (macro)     : {m['f1_macro']:.4f}",
            f"  F1 (weighted)  : {m['f1_weighted']:.4f}",
            f"  Sensitivity    : {m['sensitivity']:.4f}",
            f"  Specificity    : {m['specificity']:.4f}",
            f"  AUROC          : {m['auroc']:.4f}",
            "  Confusion Matrix:",
            format_cm(m["cm"], class_names=list(class_names)),
        ]
    return lines + ["", _RULE]


def _training_lines(arch: str, cfg, seed: int, run: dict) -> list[str]:
    clip = f"max_norm={cfg.grad_clip_norm}" if cfg.grad_clip_norm is not None else "none"
    init = "pretrained" if run.get("pretrained", True) else "random init"
    d = run["duration"]
    return [
        _RULE,
        "TRAINING",
        f"  Total epochs : {run['epochs']}",
        f"  Duration     : {d:.1f} s ({d / 60:.1f} min)",
        f"  Model        : {arch} ({init}, in_chans=1)",
        f"  AMP scaler   : {run.get('grad_scaler', 'n/a')}",
        f"  Optimizer    : AdamW (lr={cfg.learning_rate}, wd={cfg.weight_decay})",
        f"  Scheduler    : ReduceLROnPlateau (factor={cfg.lr_factor}, patience={cfg.lr_patience})",
        f"  Early stop   : patience={cfg.early_stop_patience}",
        f"  Grad clip    : {clip}",
        f"  Max epochs   : {cfg.max_epochs}",
        f"  Batch size   : {cfg.batch_size}",
        f"  Seed         : {seed}",
        _RULE,
    ]


def _write(path: str, lines: list[str]) -> None:
    """Write via a temporary file, so an interrupted write never leaves a
    file that looks complete."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.replace(tmp, path)


def write_grid_result(path: str, task: str, arch: str, cfg, seed: int, ww: int, wc: int,
                      run: dict, class_names) -> None:
    lo, hi = window_bounds(ww, wc)
    lines = [
        _RULE,
        "WINDOWING PARAMETERS",
        f"  WW           : {ww}",
        f"  WC           : {wc}",
        f"  HU range     : [{lo:.0f}, {hi:.0f}]",
        f"  Classes      : {CLASS_DESCRIPTIONS[task]}",
    ]
    _write(path, lines + _training_lines(arch, cfg, seed, run) + _metric_lines(run["results"], class_names))


def write_experiment_a_result(path: str, task: str, arch: str, cfg, seed: int, condition: str,
                              window: tuple[int, int] | None, run: dict, class_names) -> None:
    lines = [_RULE, "CONDITION", f"  Condition    : {condition}"]
    if window is None:
        lines.append("  Windowing    : none (split-wide min-max -> [0,1])")
    else:
        ww, wc = window
        lo, hi = window_bounds(ww, wc)
        lines += [f"  WW           : {ww}", f"  WC           : {wc}", f"  HU range     : [{lo:.0f}, {hi:.0f}]"]
    lines.append(f"  Classes      : {CLASS_DESCRIPTIONS[task]}")
    _write(path, lines + _training_lines(arch, cfg, seed, run) + _metric_lines(run["results"], class_names))


# ---------------------------------------------------------------- parsing

_NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan)"
_METRIC_KEYS = ("acc", "f1_macro", "f1_weighted", "sensitivity", "specificity", "auroc")


def _num(pattern: str, text: str, cast=float):
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    value = m.group(1)
    if value == "nan":
        return math.nan
    return cast(value) if cast is not int else int(float(value))


def parse_result_text(text: str) -> dict:
    """Parse the content of a result file (new or original format).

    Returns a dict with the keys condition, ww, wc, model, seed, epochs,
    duration_s, batch_size, lr, max_epochs (None when absent) and "splits":
    {"TRAIN"|"VAL"|"TEST": {acc, f1_macro, f1_weighted, sensitivity,
    specificity, auroc, cm}} for every split block found.
    """
    m = re.search(r"(?:Condition|Koşul Adı)\s*:\s*(\S+)", text)
    model = re.search(r"Model\s*:\s*(\S+)", text)
    out = {
        "condition": m.group(1) if m else None,
        "ww": _num(rf"^\s*WW\s*:\s*{_NUM}", text),
        "wc": _num(rf"^\s*WC\s*:\s*{_NUM}", text),
        "model": model.group(1) if model else None,
        "seed": _num(r"Seed\s*:\s*(\d+)", text, int),
        "epochs": _num(r"(?:Total epochs|Toplam Epoch)\s*:\s*(\d+)", text, int),
        "duration_s": _num(rf"(?:Duration|Süre)\s*:\s*{_NUM}\s*(?:s|sn)\b", text),
        "batch_size": _num(r"Batch [Ss]ize\s*:\s*(\d+)", text, int),
        "lr": _num(rf"lr={_NUM}", text),
        "max_epochs": _num(r"(?:Max epochs|Max Epoch)\s*:\s*(\d+)", text, int),
        "splits": {},
    }
    for tag in SPLIT_TAGS:
        bm = re.search(rf"---\s*{tag}\s*---(.+?)(?=---\s*(?:TRAIN|VAL|TEST)\s*---|={{3,}}|\Z)",
                       text, re.DOTALL)
        if not bm:
            continue
        block = bm.group(1)
        metrics = {
            "acc": _num(rf"Accuracy\s*:\s*{_NUM}%", block),
            "f1_macro": _num(rf"F1 \(macro\)\s*:\s*{_NUM}", block),
            "f1_weighted": _num(rf"F1 \(weighted\)\s*:\s*{_NUM}", block),
            "sensitivity": _num(rf"Sensitivity\s*:\s*{_NUM}", block),
            "specificity": _num(rf"Specificity\s*:\s*{_NUM}", block),
            "auroc": _num(rf"AUROC\s*:\s*{_NUM}", block),
        }
        rows = re.findall(r"True_\d+:\s+([\d ]+)", block)
        metrics["cm"] = [[int(v) for v in r.split()] for r in rows] or None
        out["splits"][tag] = metrics
    return out


def parse_result_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return parse_result_text(f.read())
