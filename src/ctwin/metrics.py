"""Class weights and binary classification metrics (paper Sec. 3.5, 3.6)."""

from __future__ import annotations

import numpy as np


def compute_class_weights(labels: np.ndarray, num_classes: int):
    """Inverse-frequency weights w_c = N / (K * N_c), from the training labels only
    (returned as a float32 torch tensor)."""
    import torch  # imported here so that the analysis scripts do not need PyTorch

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = len(labels) / (num_classes * counts)
    return torch.FloatTensor(weights)


def sensitivity_specificity(cm: np.ndarray) -> tuple[float, float]:
    """Sensitivity and specificity of a 2x2 confusion matrix (rows = true,
    columns = predicted, class 1 = stroke). 0.0 when undefined."""
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def format_cm(cm: np.ndarray, class_names=None) -> str:
    """Render a confusion matrix as text (the layout used in the result files)."""
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    header = "              " + "  ".join(f"Pred_{c:>s}" for c in class_names)
    rows = []
    for i, row in enumerate(cm):
        rows.append(f"    True_{class_names[i]}:  " + "  ".join(f"{v:>6}" for v in row))
    return header + "\n" + "\n".join(rows)
