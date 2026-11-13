"""The four architectures of the study (paper Sec. 3.4), built with timm.

Every model starts from timm's default ImageNet checkpoint for that name, its
first convolution is adapted to a single input channel (in_chans=1; timm sums
the pretrained RGB filters) and its head outputs two classes.
"""

from __future__ import annotations

import timm
import torch.nn as nn

from .constants import ARCHITECTURES  # noqa: F401 (re-exported)


def build_model(arch: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create a single-channel classifier for one of ARCHITECTURES."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {sorted(ARCHITECTURES)}")
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, in_chans=1)
