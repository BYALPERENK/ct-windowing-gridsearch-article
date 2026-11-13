"""Figure 5 - schematic of Experiment B (systematic grid search with testing and
comparison), in the same visual language as Figure 4. All CT panels are real windowed
renderings of one example slice given with --dicom; the 3x3 contact sheet shows what
the grid produces, and the grid schematic plots the true 352-point (22 WW x 16 WC)
sampling with the clinical window marked.

    python scripts/figures/make_fig5_experiment_b.py --dicom example.dcm --out figures/experiment_B.png
"""
import argparse
_p = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
_p.add_argument("--dicom", required=True, help="example NCCT slice (512x512 DICOM) shown in the figure")
_p.add_argument("--out", default="figures/experiment_B.png")


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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle

# ------------------------------------------------------------------ load + HU
ds = pydicom.dcmread(_args.dicom)
hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
hu = hu[_rows[0]:_rows[1], _cols[0]:_cols[1]]
if hu.ndim != 2 or hu.size == 0 or hu.max() == hu.min():
    raise SystemExit(f"--crop gives an unusable image region (shape {hu.shape})")

def window(hu, ww, wc):
    lo, hi = wc - ww / 2.0, wc + ww / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)

raw_disp = (hu - hu.min()) / (hu.max() - hu.min())

# ------------------------------------------------------------------ palette
BLUE   = "#2e6da4"
TEAL   = "#1f8a8c"
SRC    = "#3f5266"
GOLD   = "#f0b400"
INK    = "#1a1a1a"
GREY   = "#5b6b7b"
ZINK   = "#3f4a55"    # neutral (colourless) tone for the zoom square + detail card
BORDER = "#9fb3c8"
FW, FH = 15.6, 7.5
AR = FH / FW
MASP = FW / FH

PAGE_W = 19.6
XMAX = PAGE_W / FW
fig = plt.figure(figsize=(PAGE_W, FH), dpi=200)
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, XMAX); bg.set_ylim(0, 1); bg.axis("off")

def rbox(x, y, w, h, ec=BORDER, fc="white", lw=1.6, r=0.02, z=2):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        ec=ec, fc=fc, lw=lw, zorder=z, mutation_aspect=MASP))

def header(x, y, w, h, hh, color, text, fs=12):
    bg.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle="round,pad=0,rounding_size=0.018",
        ec=color, fc=color, lw=0, zorder=3, mutation_aspect=MASP))
    bg.add_patch(Rectangle((x, y + h - hh), w, hh * 0.5, fc=color, ec="none", zorder=3))
    bg.text(x + w/2, y + h - hh/2, text, va="center", ha="center", fontsize=fs,
            color="white", weight="bold", zorder=4)

def arrow(x0, y0, x1, y1, lw=3.0):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=22, lw=lw, color=GREY, zorder=3, shrinkA=0, shrinkB=0))

def image(x, y, ih, arr, hl=None):
    iw = ih * AR
    ax = fig.add_axes([x / XMAX, y, iw / XMAX, ih]); ax.imshow(arr, cmap="gray", vmin=0, vmax=1); ax.axis("off")
    if hl:
        bg.add_patch(Rectangle((x, y), iw, ih, fill=False, ec=hl, lw=2.6, zorder=6))
    return iw

# common vertical extent
BY, BH = 0.07, 0.86
midy = BY + BH/2
hh = 0.075

# ================================================================ (1) SOURCE
s_x, s_w = 0.010, 0.165
rbox(s_x, BY, s_w, BH)
header(s_x, BY, s_w, BH, hh, SRC, "Raw NCCT Image", fs=11.5)
iw_s = s_w - 0.028; ih_s = iw_s / AR      # fit the image to the box width
img_x = s_x + (s_w - iw_s)/2
image(img_x, midy - ih_s/2 - 0.01, ih_s, raw_disp)
bg.text(s_x + s_w/2, BY + 0.09, "Preserved\nHounsfield Units", ha="center", va="center",
        fontsize=9.2, color=GREY, linespacing=1.2)

# ================================================================ (2) GRID SEARCH
g_x, g_w = 0.226, 0.220
rbox(g_x, BY, g_w, BH)
header(g_x, BY, g_w, BH, hh, BLUE, "Systematic WW–WC Grid Search", fs=11.5)
bg.text(g_x + g_w/2, BY + BH - hh - 0.055,
        "WW: 10–220 (step 10)  ·  WC: 0–150 (step 10)",
        ha="center", va="center", fontsize=9.2, color=INK)
bg.text(g_x + g_w/2, BY + BH - hh - 0.095, "352 window combinations",
        ha="center", va="center", fontsize=10, color=BLUE, weight="bold")
# real 16x22 sampling grid.  Axes SWAPPED (x = WC, y = WW) so the tall box is
# filled by a tall grid. Axes width:height set to x_range:y_range (in inches) so a
# 10-HU step spans the same physical distance on both axes -> equal gaps.
XR, YR = 170.0, 230.0                       # x = WC range, y = WW range
gh_ax = 0.40
gw_ax = gh_ax * (FH / FW) * (XR / YR)
gax_x, gax_y = g_x + 0.052, midy - gh_ax/2 - 0.02
GX0, GX1, GY0, GY1 = -10.0, 160.0, 0.0, 235.0
gax = fig.add_axes([gax_x / XMAX, gax_y, gw_ax / XMAX, gh_ax])
WWv = np.arange(10, 221, 10); WCv = np.arange(0, 151, 10)
X, Y = np.meshgrid(WCv, WWv)                 # X = WC, Y = WW
gax.scatter(X, Y, s=26, marker="s", c="#93b8dd", edgecolors="#5f88b5",
            linewidth=0.4, zorder=2)
# diverse example windows the zoom card shows (enlarged) so the visual effect of
# windowing is clearly different from cell to cell
selWW = [20, 80, 200]; selWC = [10, 40, 80]
gax.scatter([40], [80], s=60, marker="s", facecolor=GOLD, edgecolor="#333",
            linewidth=1.3, zorder=5)
gax.set_xlim(GX0, GX1); gax.set_ylim(GY0, GY1)
gax.set_xlabel("WC (HU)", fontsize=9); gax.set_ylabel("WW (HU)", fontsize=9)
gax.set_xticks([0, 50, 100, 150]); gax.set_yticks([10, 110, 220])
gax.tick_params(labelsize=8)
for sp in ["top", "right"]:
    gax.spines[sp].set_visible(False)
# clean clinical legend below the grid (gold square + label), centred in the box
leg_y = BY + 0.045
seg = 0.166
lx0 = g_x + (g_w - seg)/2
bg.add_patch(Rectangle((lx0, leg_y - 0.013), 0.013, 0.026, fc=GOLD, ec="#333",
                       lw=0.9, zorder=4))
bg.text(lx0 + 0.021, leg_y, "clinical window: WW=80, WC=40",
        va="center", ha="left", fontsize=8.3, color=INK)

# ============================= (3) ZOOM / INSPECTION CARD  (deliberately NOT a stage)
# A magnifier-style detail card: the enlarged examples the grid produces. Sharp-cornered
# and colourless (with a drop shadow + lens icon) so it reads as an inspection card
# lifted above the pipeline, not another rounded/coloured processing box.
c_x, c_w = 0.497, 0.252
cY, cH = 0.21, 0.58                                # shorter than the other boxes,
cTop = cY + cH                                     # centred on the pipeline midline
bg.add_patch(Rectangle((c_x + 0.006, cY - 0.010), c_w, cH,         # drop shadow
    ec="none", fc="#c4cfda", lw=0, zorder=1.4, alpha=0.55))
bg.add_patch(Rectangle((c_x, cY), c_w, cH,                         # the card (sharp)
    ec=ZINK, fc="white", lw=2.0, zorder=2))
# lens icon (round, drawn in a square inset) + title
licon = fig.add_axes([(c_x + 0.020) / XMAX, cTop - 0.075, 0.030 / XMAX, 0.030*MASP])
licon.set_xlim(0, 1); licon.set_ylim(0, 1); licon.axis("off")
licon.add_patch(Circle((0.42, 0.58), 0.30, ec=ZINK, fc="none", lw=2.6))
licon.plot([0.63, 0.92], [0.37, 0.08], color=ZINK, lw=3.2, solid_capstyle="round")
bg.text(c_x + 0.058, cTop - 0.048, "Windowed CT Inputs", ha="left", va="center",
        fontsize=12, weight="bold", color=ZINK, zorder=4)
bg.text(c_x + c_w/2, cTop - 0.098,
        "example windowed inputs across the search grid", ha="center",
        va="center", fontsize=8.6, style="italic", color=GREY)
# --- 3x3 block of real windowed images (columns = WW, rows = WC) ---
gap = 0.008
cell_w = 0.048
cell_h = cell_w / AR                                     # square cells
grid_w = 3*cell_w + 2*gap
grid_h = 3*cell_h + 2*gap
left_room, right_room = 0.062, 0.020                     # margins for labels / dots
block_left = c_x + left_room + (c_w - left_room - right_room - grid_w)/2
block_cy   = cTop - 0.19 - grid_h/2                       # sits below the title, hugs card
block_top  = block_cy + grid_h/2
block_bot  = block_cy - grid_h/2
block_right = block_left + grid_w
cc_x = block_left + (cell_w + gap) + cell_w/2             # centre-column x
for i, wc in enumerate(selWC):
    ry = block_top - i*(cell_h + gap) - cell_h
    bg.text(block_left - 0.038, ry + cell_h/2, f"{wc}", ha="center", va="center",
            fontsize=9, color=INK)
    for j, ww in enumerate(selWW):
        cx = block_left + j*(cell_w + gap)
        hl = GOLD if (ww == 80 and wc == 40) else None
        image(cx, ry, cell_h, window(hu, ww, wc), hl=hl)
for j, ww in enumerate(selWW):                            # WW tick values above columns
    cx = block_left + j*(cell_w + gap) + cell_w/2
    bg.text(cx, block_top + 0.036, f"{ww}", ha="center", va="center",
            fontsize=9, color=INK)
bg.text(block_left + grid_w/2, block_top + 0.062, "WW (HU)", ha="center", va="center",
        fontsize=9.2, weight="bold", color=INK)
bg.text(block_left - 0.060, block_cy, "WC (HU)", ha="center", va="center",
        fontsize=9.2, weight="bold", color=INK, rotation=90)
# continuation dots on all four sides — equal PHYSICAL spacing (x-offsets scaled by AR
# so horizontal and vertical dots match), kept inside the card
dv = [0.010 + k*0.008 for k in range(3)]          # vertical offsets (y-fraction)
dh = [v*AR for v in dv]                            # horizontal offsets (equal in inches)
for v, h in zip(dv, dh):
    bg.scatter([cc_x, cc_x], [block_top + v, block_bot - v], s=7, color=GREY, zorder=4)
    bg.scatter([block_left - h, block_right + h], [block_cy, block_cy], s=7,
               color=GREY, zorder=4)

# ================================================================ (4) EVALUATION
e_x, e_w = 0.800, 0.190
rbox(e_x, BY, e_w, BH)
header(e_x, BY, e_w, BH, hh, BLUE, "Training on\nEach Window", fs=11.5)
chip_c = BY + 0.42                      # centre of the 4-chip stack
for i, name in enumerate(["ResNet-50", "DenseNet-121", "EfficientNet-B0", "MobileViTv2-175"]):
    yy = chip_c + (1.5 - i)*0.10
    bg.add_patch(FancyBboxPatch((e_x + 0.022, yy - 0.032), e_w - 0.044, 0.064,
        boxstyle="round,pad=0,rounding_size=0.012", ec=BLUE, fc="white", lw=1.4,
        zorder=3, mutation_aspect=MASP))
    bg.text(e_x + e_w/2, yy, name, ha="center", va="center", fontsize=9.8,
            color=INK, zorder=4)

bg.text(e_x + e_w/2, BY + 0.085, "2 tasks, 5 seeds each", ha="center", va="center",
        fontsize=9.6, style="italic", color=GREY)

# ================================================================ (5) COMPARISON
k_x, k_w = 1.040, 0.205
rbox(k_x, BY, k_w, BH, lw=1.8)
header(k_x, BY, k_w, BH, hh, BLUE, "Testing and\nComparison", fs=11.5)
def chip(y, h, ec, title, sub, fc="white"):
    x, w = k_x + 0.016, k_w - 0.032
    rbox(x, y, w, h, ec=ec, fc=fc, lw=1.6, r=0.012, z=4)
    bg.text(x + w/2, y + h*0.68, title, ha="center", va="center", fontsize=11.5, weight="bold", color=INK, zorder=5)
    bg.text(x + w/2, y + h*0.30, sub, ha="center", va="center", fontsize=10, color=INK, zorder=5, linespacing=1.25)
ch = 0.15
top_y, bot_y = 0.655, 0.195
mid_h = 0.19
mid_y = (top_y + bot_y + ch) / 2 - mid_h / 2
chip(top_y, ch, "#5f88b5", "Each window", "Test F1 and AUROC")
chip(bot_y, ch, GOLD, "Clinical window", "Test F1 and AUROC")
chip(mid_y, mid_h, BLUE, "Window vs. clinical", "Paired difference over five seeds\nCorrected across all windows", fc="#eaf1f8")
kc = k_x + k_w/2
arrow(kc, top_y, kc, mid_y + mid_h, lw=2.4)
arrow(kc, bot_y + ch, kc, mid_y, lw=2.4)
bg.text(kc, BY + 0.06, "Same held-out test set for all windows", ha="center", va="center",
        fontsize=9.6, style="italic", color=GREY)

# ============================================ ZOOM CALLOUT  (grid -> real images)
# Box 2 and box 3 depict the SAME grid: box 2 as abstract sampled points, box 3 as
# the images those points produce. So instead of an arrow, draw ONE square that selects
# a 3x3 block of grid cells (centred on the clinical window) and expand it into box 3.
def g2fig(xd, yd):                              # grid data coords -> figure coords
    fx = gax_x + (xd - GX0) / (GX1 - GX0) * gw_ax
    fy = gax_y + (yd - GY0) / (GY1 - GY0) * gh_ax
    return fx, fy
# 3x3 block of cells centred on clinical (WC 30-50, WW 70-90); +5 HU so edges fall
# between cells and the square cleanly encloses exactly nine little squares
rx0, ry0 = g2fig(30 - 5, 70 - 5)
rx1, ry1 = g2fig(50 + 5, 90 + 5)
# Draw the square + the two "zoom" leader lines in a TOP overlay axes so the
# connection is never hidden behind box 2's grid or the detail card. The lines run
# from the square's right corners to the card's left corners (they stop at the card
# edge, so nothing is drawn over the card interior).
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, XMAX); ov.set_ylim(0, 1)
ov.axis("off"); ov.patch.set_alpha(0)
ov.plot([rx1, c_x], [ry1, block_top], color=ZINK, lw=1.3, alpha=0.7)
ov.plot([rx1, c_x], [ry0, block_bot], color=ZINK, lw=1.3, alpha=0.7)
ov.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, ec=ZINK, fc="none", lw=1.9))

# ================================================================ ARROWS
for (xa, xb) in [(s_x + s_w, g_x), (c_x + c_w, e_x), (e_x + e_w, k_x)]:
    arrow(xa + 0.006, midy, xb - 0.006, midy)

os.makedirs(os.path.dirname(_args.out) or ".", exist_ok=True)
fig.savefig(_args.out, dpi=200,
            bbox_inches="tight", facecolor="white", pad_inches=0.12)
print(f"saved {_args.out}")
