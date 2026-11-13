import numpy as np
import pytest
import torch

from ctwin.analysis import resolve_arch_dirs
from ctwin.data import train_transform
from ctwin.metrics import compute_class_weights
from ctwin.training import TrainConfig, amp_setup, train_and_evaluate


def test_resolve_arch_dirs_accepts_both_layouts(tmp_path):
    (tmp_path / "resnet50").mkdir()
    (tmp_path / "Resnet-50_yenilenmis").mkdir()
    with pytest.raises(FileNotFoundError, match="exactly one"):
        resolve_arch_dirs(str(tmp_path), ("resnet50",), "ischemic")
    (tmp_path / "resnet50").rmdir()
    assert resolve_arch_dirs(str(tmp_path), ("resnet50",), "ischemic")["resnet50"].endswith("Resnet-50_yenilenmis")
    with pytest.raises(FileNotFoundError):  # the plain name belongs to the hemorrhagic archive only
        resolve_arch_dirs(str(tmp_path), ("resnet50",), "hemorrhagic")


def test_class_weights_are_inverse_frequency():
    w = compute_class_weights(np.array([0, 0, 0, 1]), 2)
    np.testing.assert_allclose(w.numpy(), [4 / 6, 4 / 2])


def test_train_and_evaluate_runs_on_cpu():
    rng = np.random.default_rng(0)

    def split(n):
        x = rng.random((n, 64, 64), dtype=np.float32)
        y = np.array([0, 1] * (n // 2), dtype=np.int64)
        x[y == 1, 20:40, 20:40] += 1.0
        return x, y

    tr, vl, te = split(16), split(8), split(8)
    cfg = TrainConfig(batch_size=8, max_epochs=2, learning_rate=1e-3, lr_patience=1, early_stop_patience=1)
    amp_ctx, scaler = amp_setup(torch.device("cpu"))
    run = train_and_evaluate("efficientnet_b0", cfg, 42, tr, vl, te,
                             class_weights=compute_class_weights(tr[1], 2), device=torch.device("cpu"),
                             amp_ctx=amp_ctx, scaler=scaler, train_tf=train_transform(),
                             pretrained=False, log=lambda s: None)
    assert 1 <= run["epochs"] <= 2
    assert set(run["results"]) == {"TRAIN", "VAL", "TEST"}
    assert run["results"]["TEST"]["cm"].sum() == 8
