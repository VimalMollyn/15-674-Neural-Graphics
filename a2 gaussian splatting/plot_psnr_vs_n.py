#!/usr/bin/env python3
"""PSNR against the number of Gaussians for every image fitted by a2.py: one curve per
image on one pair of axes, a dot per N joined by a line.

    direnv exec .. python plot_psnr_vs_n.py                 (from this directory; the venv lives in ..)
    direnv exec .. python plot_psnr_vs_n.py --ns 128 256 1024 --images coffee cat astronaut

The points are read off the file names a2.py writes to images/,
``<image>_opt_N<count>_psnr<db>.png``: the count is the Gaussians the run ended with and
the PSNR is the fitted image against the resized target, as the training loop printed
it. Only the counts in --ns are drawn (128, 256 and 1024 by default), so files from
densification runs that ended at other counts are left out. Every image must have a
file for every N, or this stops and says which is missing.

Writes ``figures/psnr_vs_n.png`` next to this script.

Drawn in the style of plot_sensor_power.py (CamWatchGestures) by way of the a1 figures
(plot_psnr.py, plot_size_psnr.py): bold Helvetica, left/bottom spines only, hand-drawn
grid, the axis lines ending on the last tick, the legend to the right of the plot, and
that figure's pastels, one hue per image. N is on a log axis with powers-of-two ticks,
as in plot_size_psnr.py, so 128 and 256 are not crushed together by 1024; PSNR is
linear, on ticks every 10 dB, or every 5 dB when the PSNRs span less than 20.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
OUT = HERE / "figures" / "psnr_vs_n.png"
DPI = 400  # print quality, as plot_sensor_power.py

# the laptop has Helvetica Neue, the linux box does not — fall through rather than warn
FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = FONTS

# the images as fitted, and the Gaussian counts of P5
IMAGES = ("coffee", "cat", "astronaut")
NS = (128, 256, 1024)
# one hue per image, from plot_sensor_power.VENDOR_COLORS by way of plot_psnr.ARCH_COLORS:
# the analogous run pink -> purple -> blue; an image not named here takes the next spare
IMAGE_COLORS = {
    "coffee": "#F590B8",
    "cat": "#B3A7FB",
    "astronaut": "#95BEE6",
}
SPARE_COLORS = ("#82C167", "#EFCF53")   # green, yellow, for a fourth and fifth image
FILE_RE = re.compile(r"^(?P<image>[A-Za-z0-9]+)_opt_N(?P<n>\d+)_psnr(?P<psnr>[\d.]+)\.png$")

PAD_LO_X = 0.10   # decades of axis before the first tick: the "origin" corner
PAD_HI_X = 0.04   # decades of axes box past the last tick, so the end dots are not clipped
PAD_LO_Y = 0.15   # tick spacings, likewise, on the linear axis
PAD_HI_Y = 0.05
PANEL_W, PANEL_H = 4.6, 3.4              # inches, the axes box
LW = 2.0                                 # three points per curve: a heavier line than the training curves
MS = 7                                   # marker size, as the dots of plot_size_psnr.py


def _bold(size: float) -> FontProperties:
    return FontProperties(family=FONTS, weight="bold", size=size)


def _style_axis(ax, tick_size: float = 16, spine_w: float = 2) -> None:
    """Paper axis chrome: left/bottom spines only, bold tick labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(spine_w)
        ax.spines[side].set_color("black")
    ax.tick_params(width=spine_w, color="black")
    ax.set_axisbelow(True)
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_fontproperties(_bold(tick_size))


def colour(image: str, images) -> str:
    if image in IMAGE_COLORS:
        return IMAGE_COLORS[image]
    extra = [im for im in images if im not in IMAGE_COLORS]
    return SPARE_COLORS[extra.index(image) % len(SPARE_COLORS)]


def label(image: str) -> str:
    """The image as its file name in images/."""
    return f"{image}.png"


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #
Key = tuple[str, int]                    # (image, N)


def load(images_dir: Path, images, ns) -> dict[Key, float]:
    """``{(image, N): psnr}`` from the file names in images/. Every (image, N) asked for
    must be there: a missing run fails here, not in the figure, where it would just be a
    missing dot. If a run was repeated, the latest file wins."""
    found: dict[Key, tuple[float, float]] = {}      # (mtime, psnr)
    for path in images_dir.glob("*.png"):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        key = (m.group("image"), int(m.group("n")))
        stamp = path.stat().st_mtime
        if key not in found or stamp > found[key][0]:
            found[key] = (stamp, float(m.group("psnr")))
    pts: dict[Key, float] = {}
    for im in images:
        for n in ns:
            if (im, n) not in found:
                raise SystemExit(f"no {im}_opt_N{n}_psnr*.png in {images_dir}: run a2.py on {im} with N = {n}")
            pts[(im, n)] = found[(im, n)][1]
    return pts


# --------------------------------------------------------------------------- #
# plot                                                                        #
# --------------------------------------------------------------------------- #
def xticks_for(ns) -> tuple[int, ...]:
    """Every power of two from the smallest N to the largest."""
    lo, hi = int(np.floor(np.log2(min(ns)))), int(np.ceil(np.log2(max(ns))))
    return tuple(2 ** k for k in range(lo, hi + 1))


def yticks_for(pts) -> tuple[int, ...]:
    """Ticks every 10 dB spanning the PSNRs, or every 5 dB when they span less than 20."""
    lo, hi = min(pts.values()), max(pts.values())
    step = 10 if hi - lo >= 20 else 5
    return tuple(range(int(np.floor(lo / step)) * step, int(np.ceil(hi / step)) * step + 1, step))


def _draw(ax, pts, images, ns) -> None:
    """One curve per image: a dot at each N, joined by a line in the image's hue. The
    best image at the largest N is drawn first and the worst last, so where two curves
    coincide the paler one shows over the deeper one instead of vanishing under it."""
    for depth, im in enumerate(images):
        ys = [pts[(im, n)] for n in ns]
        ax.plot(ns, ys, color=colour(im, images), linewidth=LW, marker="o", markersize=MS,
                solid_capstyle="round", zorder=4 + 0.1 * depth)


def _finish_axis(ax, xticks, yticks, tick_size: float) -> None:
    """Log x, linear y: ticks and limits, the two axis lines and the hand-drawn grid."""
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:g}" for v in xticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    # The axis lines meet at the axes corner, a step before the first tick, and end
    # exactly on the last tick — no overhang; the axes box keeps a small margin past it so
    # the end dots are not clipped. The grid is drawn by hand for the same reason: the
    # built-in one would span the whole box.
    dy = yticks[1] - yticks[0]
    x0, x1 = xticks[0] / 10 ** PAD_LO_X, xticks[-1]
    y0, y1 = yticks[0] - PAD_LO_Y * dy, yticks[-1]
    ax.set_xlim(x0, x1 * 10 ** PAD_HI_X)
    ax.set_ylim(y0, y1 + PAD_HI_Y * dy)
    ax.spines["left"].set_bounds(y0, y1)
    ax.spines["bottom"].set_bounds(x0, x1)
    _style_axis(ax, tick_size=tick_size)
    ax.grid(False)
    ax.hlines(yticks, x0, x1, color="0.85", linewidth=1, zorder=0)


def draw(pts, images, ns, out: Path) -> None:
    """The curves on one pair of axes, the legend of the images to the right."""
    xticks, yticks = xticks_for(ns), yticks_for(pts)
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    _draw(ax, pts, images, ns)
    _finish_axis(ax, xticks, yticks, tick_size=16)
    ax.set_xlabel("Number of Gaussians", fontproperties=_bold(18))
    ax.set_ylabel("PSNR (dB)", fontproperties=_bold(18))

    # to the right of the plot, centred on the y axis line, left edge a fixed distance
    # past the axis end
    corner = offset_copy(ax.transData, fig, x=18, y=0, units="points")
    y_mid = (ax.get_ylim()[0] + yticks[-1]) / 2
    handles = [Line2D([], [], color=colour(im, images), linewidth=LW, marker="o", markersize=MS,
                      label=label(im)) for im in images]
    ax.legend(handles=handles, prop=_bold(16), frameon=False, loc="center left",
              bbox_to_anchor=(xticks[-1], y_mid), bbox_transform=corner,
              borderaxespad=0, handlelength=1.5, labelspacing=0.45)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def table(pts, images, ns) -> str:
    width = max(len(label(im)) for im in images)
    lines = [f"  {'image':<{width}s}  colour   " + "  ".join(f"{f'N={n}':>8s}" for n in ns)]
    for im in images:
        lines.append(f"  {label(im):<{width}s}  {colour(im, images)}  "
                     + "  ".join(f"{pts[(im, n)]:8.2f}" for n in ns))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", nargs="+", default=list(IMAGES), metavar="NAME",
                   help="the images to draw, one curve each (names in images/, without .png)")
    p.add_argument("--ns", nargs="+", type=int, default=list(NS), metavar="N",
                   help="the Gaussian counts to draw, one dot each")
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR, help="the folder of a2.py's outputs")
    p.add_argument("--out", type=Path, default=OUT, help="where the figure goes")
    args = p.parse_args()
    ns = tuple(sorted(args.ns))
    pts = load(args.images_dir, args.images, ns)
    # legend and drawing order: best at the largest N first, so the legend reads top to
    # bottom as the curves do at their right-hand ends
    images = tuple(sorted(args.images, key=lambda im: -pts[(im, ns[-1])]))
    print(table(pts, images, ns))
    print()
    draw(pts, images, ns, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
