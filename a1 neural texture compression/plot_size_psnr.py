#!/usr/bin/env python3
"""Representation size against PSNR for every method on every texture: one panel per
texture, side by side on one axis, with a labelled dot per method in each.

    direnv exec .. python plot_size_psnr.py                  the three textures of P1-P5
    direnv exec .. python plot_size_psnr.py --textures vimal body many_freqs --out figures/size_vs_psnr_p8.png
                                              (from this directory; the venv lives in ..)

The methods are S3TC (a1_p1p2.py) and the three neural textures of a1_p3p4p5.py, large,
medium and small. A dot's x is the bytes the method stores for a 512x512 texture — 4
bits per texel for S3TC (two RGB565 colours and sixteen 2-bit indices per 4x4 block),
and every parameter at 4 bytes (float32) for a model, read off its checkpoint in
models/ — and its y is the PSNR of the decoded texture against the original. The model
PSNRs are read from final_results.txt (as a1_p3p4p5.py printed them); the S3TC ones are
what a1_p1p2.py printed, kept in ``S3TC_PSNR``.

The three models are dots in the colours plot_psnr.py gives them (pink large, purple
medium, blue small), joined by a light line as one family — the rate-distortion curve
of the neural texture — and S3TC is a dark square, the reference off that curve. Every
dot has its name beside it, so there is no legend. Writes ``figures/size_vs_psnr.png``
next to this script for the default textures, and ``--out`` for another set.

Drawn in the style of plot_sensor_power.py (CamWatchGestures, sensor power vs
framerate): bold Helvetica, left/bottom spines only, hand-drawn grid, the axis lines
ending on the last tick, pastel colours. Size is on a log axis with powers-of-two
ticks, so the 50-60 KB models are not crushed against the axis by the 360 KB one; PSNR
is linear, on ticks every 10 dB, or every 5 dB when the set's PSNRs span less than 20.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
RESULTS = HERE / "final_results.txt"
OUT = HERE / "figures" / "size_vs_psnr.png"
DPI = 400  # print quality, as plot_sensor_power.py

# the laptop has Helvetica Neue, the linux box does not — fall through rather than warn
FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = FONTS

# the panels, in order: the textures as trained, each titled "<name>.png"; --textures
# picks another set
TEXTURES = ("gradient", "bricks", "clouds")
TEXTURE_W = TEXTURE_H = 512

# the methods: the colour and marker of the dot, and where its label sits — the offset
# in points from the dot and the text's anchor corner. The models take plot_psnr.py's
# colours (plot_sensor_power's pastels); S3TC, the reference, is a dark grey square.
# Labels go where the neighbours and the line are not: the small and medium models are
# a hair apart in size, so the small one is labelled on its left; the medium one on its
# right but a touch low, under the line leaving it for the large one; the large one
# above its dot, off both the line coming in and the panel's right edge.
METHODS = {
    "Small": ("#95BEE6", "o", (-8, 0, "right", "center")),
    "Medium": ("#B3A7FB", "o", (8, -5, "left", "center")),
    "Large": ("#F590B8", "o", (0, 8, "center", "bottom")),
    "S3TC": ("0.35", "s", (9, 0, "left", "center")),
}
# where a label sits in one panel only, when the default runs into something there: on
# clouds the S3TC square is right of and level with the medium model, so its label goes
# above-left of the dot instead
LABEL_OVERRIDES = {
    ("clouds", "Medium"): (-2, 7, "right", "bottom"),
}
MLP = ("Small", "Medium", "Large")      # the family joined by a line, in size order
ARCH_OF = {"Large": "large", "Medium": "medium", "Small": "small"}
LINE_COLOR = "0.8"                       # the line through the models
# a1_p1p2.py's printout for textures/<texture>.png
S3TC_PSNR = {
    "gradient": 42.58418667024381, "bricks": 41.13052571422686, "clouds": 35.879420620871045,
    "vimal": 38.2279300354141, "body": 35.95029993303073, "many_freqs": 35.020564388933856,
}
S3TC_BITS_PER_TEXEL = 4

KB = 1024
XTICKS = (32, 64, 128, 256, 512)         # KB
PAD_LO_X = 0.10   # decades of axis before the first tick: the "origin" corner
PAD_HI_X = 0.04   # decades of axes box past the last tick, so the end dots are not clipped
PAD_LO_Y = 0.15   # tick spacings, likewise, on the linear axis
PAD_HI_Y = 0.05
PANEL_W, PANEL_H = 4.6, 3.4              # inches, each panel's axes box


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


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #
Key = tuple[str, str]                    # (method, texture)


def model_bytes(arch: str, tex: str) -> int:
    """The bytes of the checkpoint's parameters, at their stored precision."""
    path = MODELS_DIR / arch / f"{tex}.png".replace(".png", ".pt")
    if not path.is_file():
        raise SystemExit(f"no checkpoint for {tex} / {arch}: {path}")
    sd = torch.load(path, map_location="cpu")
    return sum(t.numel() * t.element_size() for t in sd.values())


def model_psnrs(path: Path) -> dict[tuple[str, str], float]:
    """``{(arch, texture): psnr}`` from final_results.txt's ``<arch> textures/<tex>.png
    <psnr>`` lines. The number is taken by pattern, so a stray character after it (the
    file is hand-kept) does not lose the row."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(\w+)\s+textures/(\w+)\.png\s+([\d.]+)", line)
        if m:
            out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def load(textures) -> dict[Key, tuple[float, float]]:
    """``{(method, texture): (bytes, psnr)}`` for every method on every texture."""
    psnr = model_psnrs(RESULTS)
    pts: dict[Key, tuple[float, float]] = {}
    for tex in textures:
        if tex not in S3TC_PSNR:
            raise SystemExit(f"no S3TC PSNR for {tex}: run a1_p1p2.py on it and add it to S3TC_PSNR")
        pts[("S3TC", tex)] = (TEXTURE_W * TEXTURE_H * S3TC_BITS_PER_TEXEL / 8, S3TC_PSNR[tex])
        for method in MLP:
            arch = ARCH_OF[method]
            if (arch, tex) not in psnr:
                raise SystemExit(f"no PSNR for {arch} / {tex} in {RESULTS}")
            pts[(method, tex)] = (model_bytes(arch, tex), psnr[(arch, tex)])
    return pts


# --------------------------------------------------------------------------- #
# plot                                                                        #
# --------------------------------------------------------------------------- #
def _draw_panel(ax, fig, pts, tex: str, ms: float, label_size: float) -> None:
    """The texture's methods as dots, the models joined by a line, each dot named."""
    xs = [pts[(m, tex)][0] / KB for m in MLP]
    ys = [pts[(m, tex)][1] for m in MLP]
    ax.plot(xs, ys, color=LINE_COLOR, linewidth=1.5, zorder=3)
    for method, (colour, marker, place) in METHODS.items():
        dx, dy, ha, va = LABEL_OVERRIDES.get((tex, method), place)
        x, y = pts[(method, tex)]
        x /= KB
        ax.plot([x], [y], linestyle="none", marker=marker,
                markersize=ms * (0.88 if marker == "s" else 1), color=colour, zorder=4)
        ax.text(x, y, method, fontproperties=_bold(label_size), color="black", ha=ha, va=va,
                transform=offset_copy(ax.transData, fig, x=dx, y=dy, units="points"),
                zorder=5)


def yticks_for(pts) -> tuple[int, ...]:
    """Ticks every 10 dB spanning the PSNRs, or every 5 dB when they span less than 20."""
    lo, hi = min(p for _, p in pts.values()), max(p for _, p in pts.values())
    step = 10 if hi - lo >= 20 else 5
    return tuple(range(int(np.floor(lo / step)) * step, int(np.ceil(hi / step)) * step + 1, step))


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


def draw(pts, textures, out: Path) -> None:
    """The textures' panels side by side, sharing the y axis."""
    n = len(textures)
    yticks = yticks_for(pts)
    fig, axes = plt.subplots(1, n, figsize=(PANEL_W * n, PANEL_H), sharey=True)
    for ax, tex in zip(axes, textures):
        _draw_panel(ax, fig, pts, tex, ms=7, label_size=13)
        _finish_axis(ax, XTICKS, yticks, tick_size=16)
        ax.set_title(f"{tex}.png", fontproperties=_bold(18), pad=10)
        ax.set_xlabel("Representation Size (KB)", fontproperties=_bold(18))
    axes[0].set_ylabel("PSNR (dB)", fontproperties=_bold(18))
    fig.subplots_adjust(wspace=0.12)            # the shared y axis needs no label gap

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def table(pts, textures) -> str:
    width = max(len(m) for m in METHODS)
    lines = [f"  {'method':<{width}s}  {'bytes':>7s}  {'KB':>6s}  " + "  ".join(f"{t:>10s}" for t in textures)]
    for method in METHODS:
        b = pts[(method, textures[0])][0]
        lines.append(f"  {method:<{width}s}  {b:7.0f}  {b / KB:6.1f}  "
                     + "  ".join(f"{pts[(method, t)][1]:10.2f}" for t in textures))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--textures", nargs="+", default=list(TEXTURES), metavar="NAME",
                   help="the textures to draw, one panel each (names in textures/, without .png)")
    p.add_argument("--out", type=Path, default=None,
                   help="where the figure goes (default: figures/size_vs_psnr.png, or "
                        "figures/size_vs_psnr_<names>.png for a chosen set)")
    args = p.parse_args()
    textures = tuple(args.textures)
    out = args.out or (OUT if textures == TEXTURES else OUT.with_name(f"{OUT.stem}_{'_'.join(textures)}.png"))
    pts = load(textures)
    print(table(pts, textures))
    print()
    draw(pts, textures, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
