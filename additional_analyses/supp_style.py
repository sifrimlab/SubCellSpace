"""Shared style for the A4 supplementary pages drawn in these notebooks.

Every page is ONE matplotlib figure drawn from cached data and saved as editable SVG
(text kept as text for the final Inkscape pass), plus a PNG preview. Visual language
follows the paper's other supplementary pages: white background, matplotlib defaults,
bold lowercase panel letters, boxed caption with a bold title.

    from supp_style import new_page, panel_letter, caption_box, save_page, COL
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from palette import pattern_palette  # noqa: E402

COL = pattern_palette()
OUT_DIR = paths.figures("supplementary_pages")
A4 = (8.27, 11.69)


def new_page():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "svg.fonttype": "none",          # editable text in Inkscape
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt.figure(figsize=A4)


def panel_letter(fig, ax, letter, dx=-0.045, dy=0.012):
    """Bold lowercase letter just above-left of an axes (figure coordinates)."""
    b = ax.get_position()
    fig.text(b.x0 + dx, b.y1 + dy, letter, fontsize=12, weight="bold", va="bottom", ha="left")


def caption_box(fig, y_top, title, caption, width_chars=118, fontsize=8.5):
    """Boxed caption whose top edge sits at figure-y `y_top`. Returns the box bottom."""
    body = textwrap.fill(caption, width_chars)
    n = body.count("\n") + 1
    line = fontsize * 1.4 / 72 / A4[1]           # one text line as figure fraction
    h = 0.018 + line * (n + 1.25)
    y0 = y_top - h
    fig.patches.append(Rectangle((0.025, y0), 0.95, h, transform=fig.transFigure,
                                 fill=False, lw=0.8, color="black"))
    fig.text(0.037, y_top - 0.009, title, fontsize=fontsize, weight="bold", va="top")
    fig.text(0.037, y_top - 0.009 - line * 1.25, body, fontsize=fontsize, va="top", linespacing=1.4)
    return y0


def save_page(fig, stem):
    """Write <stem>.svg (final, editable) and <stem>.png (preview) into OUT_DIR."""
    out = OUT_DIR / stem
    fig.savefig(str(out) + ".svg")
    fig.savefig(str(out) + ".png", dpi=150)
    return out
