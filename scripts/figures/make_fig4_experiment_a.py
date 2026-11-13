"""Figure 4 - schematic of Experiment A: each input (RAW, BRAIN) trains its own copy of
the models; test results are then compared. The CT panels are real renderings of
one example slice given with --dicom.

    python scripts/figures/make_fig4_experiment_a.py --dicom example.dcm --out figures/experiment_A.png
"""
import argparse
_p = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
_p.add_argument("--dicom", required=True, help="example NCCT slice (512x512 DICOM) shown in the figure")
_p.add_argument("--out", default="figures/experiment_A.png")


def _crop(text):
    """'r0:r1,c0:c1' -> ((r0, r1), (c0, c1)) with r0 < r1 and c0 < c1."""
    try:
        rows, cols = (tuple(int(v) for v in part.split(":")) for part in text.split(","))
        assert len(rows) == 2 and len(cols) == 2 and 0 <= rows[0] < rows[1] and 0 <= cols[0] < cols[1]
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(f"expected r0:r1,c0:c1 with r0 < r1 and c0 < c1, got {text!r}")
    return rows, cols


_p.add_argument("--crop", type=_crop, default="45:485,50:490",
                help="rows,cols crop of the slice to the head (default fits the paper's example slice)")
_args = _p.parse_args()
_rows, _cols = _args.crop
import os
import pydicom, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ds = pydicom.dcmread(_args.dicom)
hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
hu = hu[_rows[0]:_rows[1], _cols[0]:_cols[1]]
if hu.ndim != 2 or hu.size == 0 or hu.max() == hu.min():
    raise SystemExit(f"--crop gives an unusable image region (shape {hu.shape})")

def window(hu, ww, wc):
    lo, hi = wc - ww / 2.0, wc + ww / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)

raw_full = (hu - hu.min()) / (hu.max() - hu.min())
brain = window(hu, 80, 40)

SLATE, TEAL, BLUE, INK, GREY, BORDER = "#5f7488", "#1f8a8c", "#2e6da4", "#1a1a1a", "#5b6b7b", "#9fb3c8"
FW, FH = 15.0, 6.6
AR = FH / FW
MASP = FW / FH
fig = plt.figure(figsize=(FW, FH), dpi=200)
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

def rbox(x, y, w, h, ec=BORDER, fc="white", lw=1.6, r=0.016, z=2):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, zorder=z, mutation_aspect=MASP))

def header(x, y, w, h, hh, color, text, fs=11.5):
    bg.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle="round,pad=0,rounding_size=0.015",
                                ec=color, fc=color, lw=0, zorder=3, mutation_aspect=MASP))
    bg.add_patch(plt.Rectangle((x, y + h - hh), w, hh * 0.5, fc=color, ec="none", zorder=3))
    bg.text(x + w / 2, y + h - hh / 2, text, va="center", ha="center", fontsize=fs,
            color="white", weight="bold", zorder=4)

def line(x0, y0, x1, y1, lw=2.6):
    bg.plot([x0, x1], [y0, y1], color=GREY, lw=lw, zorder=3, solid_capstyle="round")

def arrow(x0, y0, x1, y1, lw=2.8):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=20,
                                 lw=lw, color=GREY, zorder=3, shrinkA=0, shrinkB=0))

def image(x, y, ih, arr):
    iw = ih * AR
    ax = fig.add_axes([x, y, iw, ih]); ax.imshow(arr, cmap="gray", vmin=0, vmax=1); ax.axis("off")
    return x + iw

# Source
sx, sy, sw, sh = 0.010, 0.28, 0.135, 0.44
rbox(sx, sy, sw, sh)
bg.text(sx + sw/2, sy + sh - 0.05, "NCCT Slice", ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
bg.text(sx + sw/2, sy + sh - 0.105, "E-NCCT dataset\nin Hounsfield Units (HU)", ha="center", va="center",
        fontsize=9.3, color=GREY, linespacing=1.25)
ih_s = 0.235
image(sx + (sw - ih_s * AR) / 2, sy + 0.035, ih_s, raw_full)

# Input arms
ax0, aw, ah, hh = 0.185, 0.235, 0.40, 0.062
raw_y, brn_y = 0.535, 0.065
raw_mid, brn_mid = raw_y + ah/2, brn_y + ah/2

def arm(y, color, head, img, txt):
    rbox(ax0, y, aw, ah)
    header(ax0, y, aw, ah, hh, color, head)
    ir = image(ax0 + 0.012, y + 0.05, 0.265, img)
    bg.text((ir + ax0 + aw) / 2, y + (ah - hh)/2 + 0.01, txt, ha="center", va="center",
            fontsize=9.6, color=INK, linespacing=1.45)

arm(raw_y, SLATE, "RAW: No Windowing", raw_full, "Full HU range\n\nGlobal min-max\nscaling")
arm(brn_y, TEAL, "BRAIN: Clinical Window", brain, "WW = 80\nWC = 40\n\nMapped to\n[0, 1]")

# One model block per input
mx0, mw = 0.462, 0.205
ARCHS = ["ResNet-50", "DenseNet-121", "EfficientNet-B0", "MobileViTv2-175"]

def models(y, color, head):
    rbox(mx0, y, mw, ah)
    header(mx0, y, mw, ah, hh, color, head)
    cx = mx0 + mw/2
    for i, name in enumerate(ARCHS):
        bg.text(cx, y + ah - hh - 0.055 - i * 0.056, name, ha="center", va="center", fontsize=10.6, color=INK, zorder=4)
    bg.text(cx, y + 0.045, "2 tasks, 5 seeds each", ha="center", va="center", fontsize=9.2, style="italic", color=GREY)

models(raw_y, SLATE, "Training on RAW")
models(brn_y, TEAL, "Training on BRAIN")
bg.text(mx0 + mw/2, (raw_y + brn_y + ah) / 2, "same architectures and protocol, trained separately",
        ha="center", va="center", fontsize=8.6, style="italic", color=GREY)

# Test and comparison
cx0, cy0, cw, ch = 0.708, 0.065, 0.282, 0.87
rbox(cx0, cy0, cw, ch, lw=1.8)
header(cx0, cy0, cw, ch, hh, BLUE, "Testing and Comparison")
cxm = cx0 + cw/2

def chip(y, h, color, title, sub, fc="white"):
    x, w = cx0 + 0.022, cw - 0.044
    rbox(x, y, w, h, ec=color, fc=fc, lw=1.6, r=0.010, z=4)
    bg.text(x + w/2, y + h * 0.66, title, ha="center", va="center", fontsize=11, weight="bold", color=INK, zorder=5)
    bg.text(x + w/2, y + h * 0.30, sub, ha="center", va="center", fontsize=10, color=INK, zorder=5)

chip_h = 0.15
chip(raw_mid - chip_h/2, chip_h, SLATE, "RAW models", "Test F1 and AUROC")
chip(brn_mid - chip_h/2, chip_h, TEAL, "BRAIN models", "Test F1 and AUROC")

mid_h = 0.17
mid_y = 0.5 - mid_h/2
chip(mid_y, mid_h, BLUE, "BRAIN vs. RAW", "Paired difference over five seeds", fc="#eaf1f8")

arrow(cxm, raw_mid - chip_h/2, cxm, mid_y + mid_h, lw=2.4)
arrow(cxm, brn_mid + chip_h/2, cxm, mid_y, lw=2.4)
bg.text(cxm, cy0 + 0.06, "Same held-out test set for both inputs", ha="center", va="center",
        fontsize=9.6, style="italic", color=GREY)

# Arrows
bus = sx + sw + 0.018
line(sx + sw, sy + sh/2, bus, sy + sh/2)
line(bus, brn_mid, bus, raw_mid)
arrow(bus, raw_mid, ax0, raw_mid)
arrow(bus, brn_mid, ax0, brn_mid)
arrow(ax0 + aw, raw_mid, mx0, raw_mid)
arrow(ax0 + aw, brn_mid, mx0, brn_mid)
arrow(mx0 + mw, raw_mid, cx0, raw_mid)
arrow(mx0 + mw, brn_mid, cx0, brn_mid)

os.makedirs(os.path.dirname(_args.out) or ".", exist_ok=True)
fig.savefig(_args.out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print(f"saved {_args.out}")
