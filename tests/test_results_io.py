import math

import numpy as np

from ctwin.results_io import grid_result_path, is_complete, parse_result_text, write_grid_result
from ctwin.training import TrainConfig

CFG = TrainConfig(batch_size=8, max_epochs=3, learning_rate=3e-5, lr_patience=5, early_stop_patience=13,
                  grad_clip_norm=1.0)


def _metrics(acc, auroc=0.75):
    return dict(acc=acc, f1_macro=0.6123, f1_weighted=0.6345, sensitivity=0.7, specificity=0.55,
                auroc=auroc, cm=np.array([[30, 10], [5, 20]]))


def _run():
    return {"results": {"TRAIN": _metrics(90.0), "VAL": _metrics(70.0), "TEST": _metrics(65.5, math.nan)},
            "epochs": 3, "duration": 12.3, "pretrained": False, "grad_scaler": "off (no AMP)"}


def test_round_trip(tmp_path):
    path = grid_result_path(str(tmp_path), "ischemic", "resnet50", 42, 80, 40)
    write_grid_result(path, "ischemic", "resnet50", CFG, 42, 80, 40, _run(), ("0", "1"))
    assert is_complete(path)
    p = parse_result_text(open(path, encoding="utf-8").read())
    assert (p["ww"], p["wc"], p["seed"], p["epochs"], p["max_epochs"]) == (80, 40, 42, 3, 3)
    assert p["lr"] == 3e-5 and p["batch_size"] == 8
    assert p["splits"]["TEST"]["acc"] == 65.5
    assert math.isnan(p["splits"]["TEST"]["auroc"])
    assert p["splits"]["VAL"]["cm"] == [[30, 10], [5, 20]]


def test_truncated_file_is_incomplete(tmp_path):
    path = grid_result_path(str(tmp_path), "ischemic", "resnet50", 42, 80, 40)
    write_grid_result(path, "ischemic", "resnet50", CFG, 42, 80, 40, _run(), ("0", "1"))
    text = open(path, encoding="utf-8").read()
    cut = tmp_path / "cut.txt"
    cut.write_text(text[: text.rfind("    True_1:")], encoding="utf-8")
    assert not is_complete(str(cut))


ORIGINAL_FORMAT = """============================================================
KOŞUL BİLGİSİ
  Koşul Adı    : RAW
  Windowing    : YOK (global min-max normalize → [0,1])
  Sınıflar     : 0 (İnme Yok) vs 1 (İskemik)
============================================================
EĞİTİM BİLGİSİ
  Toplam Epoch : 20
  Süre         : 128.0 sn (2.1 dk)
  Model        : resnet50 (pretrained, in_chans=1)
  Optimizer    : AdamW (lr=0.0001, wd=0.01)
  Scheduler    : ReduceLROnPlateau (factor=0.5, patience=4)
  Early Stop   : patience=7
  Batch Size   : 340
  Max Epoch    : 20
  Seed         : 123
============================================================
""" + "".join(
    f"""
--- {tag} ---
  Accuracy       : 60.27%
  F1 (macro)     : 0.5988
  F1 (weighted)  : 0.6054
  Sensitivity    : 0.6042
  Specificity    : 0.6017
  AUROC          : 0.6279
  Confusion Matrix:
              Pred_0  Pred_1
    True_0:     642     425
    True_1:     302     461
""" for tag in ("TRAIN", "VAL", "TEST")) + "\n" + "=" * 60


def test_parses_original_turkish_format(tmp_path):
    p = parse_result_text(ORIGINAL_FORMAT)
    assert p["condition"] == "RAW" and p["seed"] == 123 and p["epochs"] == 20
    assert p["duration_s"] == 128.0 and p["max_epochs"] == 20 and p["lr"] == 1e-4
    assert p["splits"]["TEST"]["f1_macro"] == 0.5988
    assert p["splits"]["TEST"]["cm"] == [[642, 425], [302, 461]]
    f = tmp_path / "RAW_seed_123.txt"
    f.write_text(ORIGINAL_FORMAT, encoding="utf-8")
    assert is_complete(str(f))
