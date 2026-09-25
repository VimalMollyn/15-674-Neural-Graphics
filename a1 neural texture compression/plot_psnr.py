#!/usr/bin/env python3
"""PSNR over training for the nine (texture, architecture) runs: one panel per texture,
side by side on one y axis, with one line per architecture in each.

    direnv exec .. python plot_psnr.py        (from this directory; the venv lives in ..)

Reads ``training/<architecture>/<texture>/training.csv`` as a1_p3p4p5.py writes it: one
``step,psnr`` row per optimiser step, 2000 steps, the PSNR of that step's 16384-texel
minibatch (-10 log10 of its MSE on [0, 1] colours). Each panel is titled with the
texture's file name in textures/, and its three lines are the architectures, one hue
each: pink for large (4 grids 16-128, 4 features), purple for medium (3 grids 16-64,
2 features), blue for small (1 grid of 64, 2 features).

Writes ``figures/psnr_training.png`` next to this script.

Drawn in the style of plot_sensor_power.py (CamWatchGestures, sensor power vs framerate):
bold Helvetica, left/bottom spines only, hand-drawn grid, the axis lines ending on the
last tick, the panels sharing the y axis with one legend to the right of the last one.
The three hues are that figure's pastels (``ARCH_COLORS``), the warm-to-cool run of
pink, purple and blue so the sizes read as a progression. Both axes are linear; the
axes start a step before the first tick so the first points sit inside the plot, not on
its lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

HERE = Path(__file__).resolve().parent
TRAINING_DIR = HERE / "training"
OUT = HERE / "figures" / "psnr_training.png"
DPI = 400  # print quality, as plot_sensor_power.py

# the laptop has Helvetica Neue, the linux box does not — fall through rather than warn
FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = FONTS

# the panels, in order: the textures as trained, each titled "<name>.png"
TEXTURES = ("gradient", "bricks", "clouds")
# one hue per architecture, in legend order (largest first), from
# plot_sensor_power.VENDOR_COLORS (the pastel outline tones of the CamWatchGestures
# system diagram): the analogous run pink -> purple -> blue, warmest for the largest
ARCH_COLORS = {
    "large": "#F590B8",
    "medium": "#B3A7FB",
    "small": "#95BEE6",
}
ARCHS = tuple(ARCH_COLORS)
STEPS = 2000                            # rows expected per file, as the training loop

XTICKS = (0, 500, 1000, 1500, 2000)    # training step
YTICKS = (10, 20, 30, 40, 50, 60)      # PSNR, dB
PAD_LO = 0.15   # tick spacings of axis before the first tick: the "origin" corner
PAD_HI = 0.05   # tick spacings of axes box past the last tick, so the line ends are not clipped
END_STEPS = 50  # the "end" PSNR of a run, for the table: its mean over these
PANEL_W, PANEL_H = 4.6, 3.4             # inches, each panel's axes box


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


def colour(arch: str) -> str:
    return ARCH_COLORS[arch]


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #
Key = tuple[str, str]                    # (texture, architecture)


def load(root: Path) -> dict[Key, np.ndarray]:
    """Every run as ``{(texture, architecture): (step, psnr) array}``, rows in step order.
    All nine files must be there and complete: a missing run fails here, not in the
    figure, where it would just be a missing line."""
    series: dict[Key, np.ndarray] = {}
    for arch in ARCHS:
        for tex in TEXTURES:
            path = root / arch / tex / "training.csv"
            if not path.is_file():
                raise SystemExit(f"no training log for {tex} / {arch}: {path}")
            data = np.loadtxt(path, delimiter=",", ndmin=2)
            if data.shape != (STEPS, 2):
                raise SystemExit(f"{path}: expected {STEPS} rows of step,psnr, got {data.shape}")
            series[(tex, arch)] = data[np.argsort(data[:, 0])]
    return series


def end_psnr(run: np.ndarray) -> float:
    return float(run[-END_STEPS:, 1].mean())


def title(tex: str) -> str:
    """The texture as its file name in textures/."""
    return f"{tex}.png"


def label(arch: str) -> str:
    return arch.capitalize()


# --------------------------------------------------------------------------- #
# plot                                                                        #
# --------------------------------------------------------------------------- #
def _draw_panel(ax, series, tex: str, lw: float) -> None:
    """The texture's runs, one line through each one's 2000 steps. The largest model is
    drawn first and the smallest last, so the paler lines show over the deep one where
    the runs coincide (the three gradient runs sit within a decibel of each other for
    most of training) instead of vanishing under it."""
    for depth, arch in enumerate(ARCHS):
        run = series[(tex, arch)]
        ax.plot(run[:, 0], run[:, 1], color=colour(arch), linewidth=lw,
                solid_capstyle="round", zorder=4 + 0.1 * depth)


def _finish_axis(ax, xticks, yticks, tick_size: float) -> None:
    """Ticks and limits, the two axis lines and the hand-drawn grid."""
    ax.minorticks_off()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:g}" for v in xticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    # The axis lines meet at the axes corner, a step before the first tick, and end
    # exactly on the last tick — no overhang; the axes box keeps a small margin past it so
    # the line ends are not clipped. The grid is drawn by hand for the same reason: the
    # built-in one would span the whole box.
    dx, dy = xticks[1] - xticks[0], yticks[1] - yticks[0]
    x0, y0 = xticks[0] - PAD_LO * dx, yticks[0] - PAD_LO * dy
    x1, y1 = xticks[-1], yticks[-1]
    ax.set_xlim(x0, x1 + PAD_HI * dx)
    ax.set_ylim(y0, y1 + PAD_HI * dy)
    ax.spines["left"].set_bounds(y0, y1)
    ax.spines["bottom"].set_bounds(x0, x1)
    _style_axis(ax, tick_size=tick_size)
    ax.grid(False)
    ax.hlines(yticks, x0, x1, color="0.85", linewidth=1, zorder=0)


def draw(series, out: Path, lw: float = 1.5) -> None:
    """The textures' panels side by side, sharing the y axis, one legend of the
    architectures to the right of the last panel."""
    n = len(TEXTURES)
    fig, axes = plt.subplots(1, n, figsize=(PANEL_W * n, PANEL_H), sharey=True)
    for ax, tex in zip(axes, TEXTURES):
        _draw_panel(ax, series, tex, lw=lw)
        _finish_axis(ax, XTICKS, YTICKS, tick_size=16)
        ax.set_title(title(tex), fontproperties=_bold(18), pad=10)
        ax.set_xlabel("Training Step", fontproperties=_bold(18))
    axes[0].set_ylabel("PSNR (dB)", fontproperties=_bold(18))
    fig.subplots_adjust(wspace=0.12)            # the shared y axis needs no label gap

    # to the right of the last panel, centred on the y axis line, left edge a fixed
    # distance past the axis end
    last = axes[-1]
    corner = offset_copy(last.transData, fig, x=18, y=0, units="points")
    y_mid = (last.get_ylim()[0] + YTICKS[-1]) / 2
    handles = [Line2D([], [], color=colour(arch), linewidth=lw, label=label(arch))
               for arch in ARCHS]
    last.legend(handles=handles, prop=_bold(16), frameon=False, loc="center left",
                bbox_to_anchor=(XTICKS[-1], y_mid), bbox_transform=corner,
                borderaxespad=0, handlelength=1.5, labelspacing=0.45)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def table(series) -> str:
    width = max(len(f"{title(t)}, {label(a)}") for t in TEXTURES for a in ARCHS)
    lines = [f"  {'run':<{width}s}  colour   PSNR (dB): step 0 -> best -> mean of last {END_STEPS}"]
    for tex in TEXTURES:
        for arch in ARCHS:
            run = series[(tex, arch)]
            lines.append(f"  {f'{title(tex)}, {label(arch)}':<{width}s}  {colour(arch)}  "
                         f"{run[0, 1]:5.1f} -> {run[:, 1].max():5.1f} -> {end_psnr(run):5.1f}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--training-dir", type=Path, default=TRAINING_DIR,
                   help="the folder of <architecture>/<texture>/training.csv logs")
    p.add_argument("--out", type=Path, default=OUT, help="where the figure goes")
    args = p.parse_args()
    series = load(args.training_dir)
    print(table(series))
    print()
    draw(series, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
