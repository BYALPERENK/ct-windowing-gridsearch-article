"""Experiment settings (paper Sec. 3.3 and Table 2).

The values are those recorded in the original training scripts and in the
headers of their result files.
"""

from __future__ import annotations

from .constants import BRAIN_WINDOW, SEEDS, WC_VALUES, WW_VALUES  # noqa: F401 (re-exported)
from .training import TrainConfig

# Experiment B. Ischemic: renewed configuration with gradient clipping.
# Hemorrhagic: no clipping. fresh_grad_scaler follows the original scripts:
# the hemorrhagic EfficientNet-B0, DenseNet-121 and MobileViTv2-175 grids were
# run by scripts that built a new GradScaler for every run, the others by
# scripts that kept one GradScaler for the whole process. (Part of the
# hemorrhagic MobileViTv2-175 grid was completed by a resume script of the
# second kind.) The flag only matters with AMP on CUDA.
GRID_CONFIGS: dict[str, dict[str, TrainConfig]] = {
    "ischemic": {
        "resnet50":        TrainConfig(batch_size=340, max_epochs=45, learning_rate=3e-5,
                                       lr_patience=5, early_stop_patience=13, grad_clip_norm=1.0),
        "densenet121":     TrainConfig(batch_size=256, max_epochs=45, learning_rate=3e-5,
                                       lr_patience=5, early_stop_patience=13, grad_clip_norm=1.0),
        "efficientnet_b0": TrainConfig(batch_size=384, max_epochs=70, learning_rate=3e-5,
                                       lr_patience=5, early_stop_patience=13, grad_clip_norm=1.0),
        "mobilevitv2_175": TrainConfig(batch_size=128, max_epochs=35, learning_rate=3e-5,
                                       lr_patience=5, early_stop_patience=13, grad_clip_norm=1.0),
    },
    "hemorrhagic": {
        "resnet50":        TrainConfig(batch_size=340, max_epochs=17, learning_rate=1e-4,
                                       lr_patience=3, early_stop_patience=5),
        "densenet121":     TrainConfig(batch_size=256, max_epochs=15, learning_rate=1e-4,
                                       lr_patience=3, early_stop_patience=5, fresh_grad_scaler=True),
        "efficientnet_b0": TrainConfig(batch_size=256, max_epochs=35, learning_rate=1e-4,
                                       lr_patience=5, early_stop_patience=11, fresh_grad_scaler=True),
        "mobilevitv2_175": TrainConfig(batch_size=128, max_epochs=15, learning_rate=1e-4,
                                       lr_patience=3, early_stop_patience=7, fresh_grad_scaler=True),
    },
}

# Experiment A (RAW vs. BRAIN): same configuration for both tasks; no clipping.
# Dict order is the order in which the original script trained the models.
EXPERIMENT_A_CONFIGS: dict[str, TrainConfig] = {
    "mobilevitv2_175": TrainConfig(batch_size=128, max_epochs=20, learning_rate=1e-4,
                                   lr_patience=4, early_stop_patience=7),
    "densenet121":     TrainConfig(batch_size=256, max_epochs=20, learning_rate=1e-4,
                                   lr_patience=4, early_stop_patience=7),
    "resnet50":        TrainConfig(batch_size=340, max_epochs=20, learning_rate=1e-4,
                                   lr_patience=4, early_stop_patience=7),
    "efficientnet_b0": TrainConfig(batch_size=384, max_epochs=35, learning_rate=1e-4,
                                   lr_patience=4, early_stop_patience=7),
}

# Experiment A conditions: None = RAW (split-wide min-max), else (WW, WC)
EXPERIMENT_A_CONDITIONS: dict[str, tuple[int, int] | None] = {
    "RAW": None,
    "BRAIN": BRAIN_WINDOW,
}
