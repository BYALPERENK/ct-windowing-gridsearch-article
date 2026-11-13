"""Study constants that do not depend on PyTorch (paper Sec. 3.3, 3.4)."""

from __future__ import annotations

SEEDS: tuple[int, ...] = (42, 123, 999, 2026, 7)

# Experiment B grid: WW 10..220 and WC 0..150, both in steps of 10 HU (352 windows)
WW_VALUES: tuple[int, ...] = tuple(range(10, 221, 10))
WC_VALUES: tuple[int, ...] = tuple(range(0, 151, 10))

# clinical brain window (WW, WC)
BRAIN_WINDOW: tuple[int, int] = (80, 40)

# timm model name -> name used in the paper; dict order = order in the paper's tables
ARCHITECTURES: dict[str, str] = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilevitv2_175": "MobileViTv2-175",
}
