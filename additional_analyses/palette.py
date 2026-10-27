"""The project's canonical pattern palette, extended to the two additional simulated patterns.

`make_paper_figures/src/plotting.py::get_pattern_palette()` is the published mapping
and is used unchanged, so a pattern keeps the colour it has in the paper's figures.
The two additional simulated categories are not in it, so they get colours chosen
to sit as far as possible from all ten canonical hues in CIELAB: black (min dE 49.9)
and dark forest green (45.9) -- both comfortably clear of `intranuclear`'s brighter
green and of every other canonical colour.

    from palette import pattern_palette
    COL = pattern_palette()
"""
import sys
from pathlib import Path

_PAPER_SRC = Path(__file__).resolve().parents[1] / "make_paper_figures" / "src"

NEW_PATTERNS = {"celledge": "#000000", "cytoplasm": "#00441b"}


def pattern_palette(extra=None):
    """{pattern: hex}. Canonical names keep their published colour."""
    import matplotlib.colors as mc
    if str(_PAPER_SRC) not in sys.path:
        sys.path.insert(0, str(_PAPER_SRC))
    from plotting import get_pattern_palette
    pal = {k: mc.to_hex(v) for k, v in get_pattern_palette().items()}
    pal.update(NEW_PATTERNS)
    if extra:
        pal.update(extra)
    return pal


def composite_colour(a, b, palette=None):
    """Colour for a two-pattern composite: the CIELAB midpoint of its parents.

    Composites are not new categories needing a maximally-distant colour the way
    `celledge` and `cytoplasm` did -- they are literally mixtures, so the honest
    encoding is the mixture of the two parents' published colours. Mixing in CIELAB
    rather than sRGB keeps the midpoint perceptually between the two instead of
    drifting dark, which matters because several canonical hues are saturated.
    """
    import numpy as np
    import matplotlib.colors as mc
    from skimage.color import rgb2lab, lab2rgb
    pal = palette or pattern_palette()
    lab = [rgb2lab(np.array(mc.to_rgb(pal[p]), float).reshape(1, 1, 3)) for p in (a, b)]
    mid = (lab[0] + lab[1]) / 2.0
    return mc.to_hex(np.clip(lab2rgb(mid).reshape(3), 0, 1))


def composite_palette(pairs, palette=None):
    """{f'{a}x{b}': hex} for an iterable of (a, b) parent-pattern tuples."""
    pal = palette or pattern_palette()
    return {f"{a}x{b}": composite_colour(a, b, pal) for a, b in pairs}
