"""Training and evaluation of one run (one architecture, input, task and seed).

Shared by Experiment A and the grid search (paper Sec. 3.5): class-weighted
cross-entropy, AdamW, ReduceLROnPlateau on validation loss, early stopping on
validation loss, AMP on CUDA, optional gradient-norm clipping, and final
evaluation of the checkpoint with the lowest validation loss.
"""

from __future__ import annotations

import gc
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .data import make_loader
from .metrics import format_cm, sensitivity_specificity
from .models import build_model

NUM_CLASSES = 2


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters of one experiment family (one task x architecture).

    fresh_grad_scaler: on CUDA, create a new GradScaler for every run (True)
    or share one GradScaler across all runs of a process (False). Both
    variants occur in the original experiments; see configs.py.
    """

    batch_size: int
    max_epochs: int
    learning_rate: float
    lr_patience: int
    early_stop_patience: int
    grad_clip_norm: float | None = None
    weight_decay: float = 1e-2
    lr_factor: float = 0.5
    fresh_grad_scaler: bool = False

    def __post_init__(self):
        for name in ("batch_size", "max_epochs", "lr_patience", "early_stop_patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 < self.lr_factor < 1:
            raise ValueError("invalid learning_rate, weight_decay or lr_factor")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive or None")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) and request deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def amp_setup(device: torch.device):
    """Return (autocast context, GradScaler) for the device; AMP is used on CUDA only."""
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda"), torch.amp.GradScaler("cuda")
    return nullcontext(), None


def train_one_epoch(model, loader, criterion, optimizer, scaler, amp_ctx, device,
                    grad_clip_norm: float | None = None) -> dict:
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        bs = x.size(0)
        loss_sum += loss.item() * bs
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += bs
        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    P = torch.cat(all_pred).numpy()
    T = torch.cat(all_true).numpy()
    return dict(
        loss=loss_sum / total,
        acc=correct / total * 100,
        f1_macro=f1_score(T, P, average="macro"),
        f1_weighted=f1_score(T, P, average="weighted"),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, amp_ctx, device) -> dict:
    """Loss, accuracy (%), macro/weighted F1, sensitivity, specificity, AUROC
    (from the class-1 softmax probability) and the confusion matrix."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_pred, all_true, all_probs = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with amp_ctx:
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits.float(), dim=1)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += bs
        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())
        all_probs.append(probs.detach().cpu())

    P = torch.cat(all_pred).numpy()
    T = torch.cat(all_true).numpy()
    probs_all = torch.cat(all_probs).numpy()

    cm = confusion_matrix(T, P, labels=list(range(NUM_CLASSES)))
    sens, spec = sensitivity_specificity(cm)
    try:
        auroc = roc_auc_score(T, probs_all[:, 1])
    except Exception:  # e.g. only one class present
        auroc = float("nan")

    return dict(
        loss=loss_sum / total,
        acc=correct / total * 100,
        f1_macro=f1_score(T, P, average="macro"),
        f1_weighted=f1_score(T, P, average="weighted"),
        sensitivity=sens,
        specificity=spec,
        auroc=auroc,
        cm=cm,
    )


def train_and_evaluate(
    arch: str,
    cfg: TrainConfig,
    seed: int,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    class_weights: torch.Tensor,
    device: torch.device,
    amp_ctx,
    scaler,
    train_tf,
    class_names: tuple[str, str] = ("0", "1"),
    pretrained: bool = True,
    start_time: float | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train one model on already-normalized (images, labels) splits and
    evaluate its best-validation-loss checkpoint on all three splits.

    The seed is set right before the model is built, so a run does not depend
    on the runs before it, except through a GradScaler shared across runs
    (cfg.fresh_grad_scaler=False with AMP): its scale carries over from the
    previous run of the process. Skipping completed runs on restart therefore
    does not continue such a process bit for bit.
    Returns {"results": {"TRAIN"|"VAL"|"TEST": metrics}, "epochs", "duration"}.
    """
    t_start = time.time() if start_time is None else start_time
    (tr_img, tr_lab), (vl_img, vl_lab), (te_img, te_lab) = train, val, test

    tr_ld = make_loader(tr_img, tr_lab, cfg.batch_size, shuffle=True, transform=train_tf)
    vl_ld = make_loader(vl_img, vl_lab, cfg.batch_size, shuffle=False)
    te_ld = make_loader(te_img, te_lab, cfg.batch_size, shuffle=False)

    set_seed(seed)
    model = build_model(arch, num_classes=NUM_CLASSES, pretrained=pretrained).to(device)
    log(f"  model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
    )
    if cfg.fresh_grad_scaler and scaler is not None:
        scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    final_epoch = 0

    for ep in range(1, cfg.max_epochs + 1):
        t_ep = time.time()
        trm = train_one_epoch(model, tr_ld, criterion, optimizer, scaler, amp_ctx, device,
                              cfg.grad_clip_norm)
        vlm = evaluate(model, vl_ld, criterion, amp_ctx, device)
        lr_now = optimizer.param_groups[0]["lr"]
        final_epoch = ep

        log(f"  epoch {ep:>2}/{cfg.max_epochs} ({time.time() - t_ep:.1f}s) lr={lr_now:.1e}  "
            f"train loss={trm['loss']:.4f} acc={trm['acc']:.2f}%  "
            f"val loss={vlm['loss']:.4f} acc={vlm['acc']:.2f}% F1m={vlm['f1_macro']:.4f} "
            f"AUROC={vlm['auroc']:.4f}")

        scheduler.step(vlm["loss"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < lr_now:
            log(f"    learning rate reduced: {lr_now:.1e} -> {new_lr:.1e}")

        if vlm["loss"] < best_val_loss:
            best_val_loss = vlm["loss"]
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.early_stop_patience:
            log(f"    early stop at epoch {ep}")
            break

    duration = time.time() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)
    else:  # the validation loss never went below +inf (e.g. it was NaN)
        log("  WARNING: no best checkpoint recorded; evaluating the last epoch")

    tr_eval_ld = make_loader(tr_img, tr_lab, cfg.batch_size, shuffle=False)
    results = {}
    for tag, loader in (("TRAIN", tr_eval_ld), ("VAL", vl_ld), ("TEST", te_ld)):
        m = evaluate(model, loader, criterion, amp_ctx, device)
        results[tag] = m
        log(f"  [{tag:>5}] acc={m['acc']:.2f}% F1m={m['f1_macro']:.4f} F1w={m['f1_weighted']:.4f} "
            f"sens={m['sensitivity']:.4f} spec={m['specificity']:.4f} AUROC={m['auroc']:.4f}")
        log(format_cm(m["cm"], class_names=list(class_names)))

    del tr_ld, vl_ld, te_ld, tr_eval_ld, model, optimizer, scheduler, criterion, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if scaler is None:
        scaler_policy = "off (no AMP)"
    else:
        scaler_policy = "fresh per run" if cfg.fresh_grad_scaler else "shared across runs"
    return {"results": results, "epochs": final_epoch, "duration": duration,
            "pretrained": pretrained, "grad_scaler": scaler_policy}
