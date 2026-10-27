"""Cell geometry: raster masks -> the polygons a boundary-based method needs.

Every observation in this project lives in a per-cell 100x100 crop-and-resize
("zoomed") frame, and the cell outline is stored there as a *raster* --
``obsm/cell_mask`` on the h5ads, ``cell_mask_zoomed/<stem>_cell-mask.tif`` in the
simulated trees. ELLA refuses a raster (``AGENTS.md``: "cell_poly is mandatory;
raster masks are not a substitute"), so the mask has to become an ``(M, 2)``
vertex array before it can be fed anything.

Two conventions fixed here and used by everything downstream:

* **A point is ``(x, y) = (col, row)``.** The point tables in ``controls.py``
  are ``row``/``col`` image indices; ELLA speaks ``x``/``y``. The swap happens
  once, here.
* **The radius is measured from the nucleus centroid**, not the cell centroid,
  because that is the origin ELLA's cellular coordinate system is anchored on.

``marching_squares`` rather than ``alphashape``
-----------------------------------------------
ELLA lists ``alphashape`` as the dependency for "build cell polygons from raster
masks". Alpha shapes reconstruct a boundary from a *point cloud*; a filled
binary mask already has an exact boundary, and marching squares
(``skimage.measure.find_contours``) recovers it without an alpha parameter to
tune and without the concavity failures alpha shapes have on thin protrusions --
which are precisely the shapes the ``protrusion`` and ``cell-edge`` patterns are
made of. ``polygon_mask_iou`` is the check that this is true rather than
asserted.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

RENDER_SHAPE = (100, 100)


# --- mask -> polygon ---------------------------------------------------------

def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of an ``(M, 2)`` polygon (always positive)."""
    x, y = np.asarray(poly)[:, 0], np.asarray(poly)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def mask_to_polygon(mask: np.ndarray, tolerance: float = 0.0,
                    min_vertices: int = 8) -> Optional[np.ndarray]:
    """Largest outer contour of a binary mask, as ``(M, 2)`` ``(x, y)`` vertices.

    Returns None for an empty mask or one whose boundary degenerates to fewer
    than ``min_vertices`` points -- the caller decides whether to drop the cell
    or fall back, rather than being handed a polygon that ray-casting will
    silently mis-answer.

    ``tolerance`` > 0 applies Douglas-Peucker simplification. Left at 0 by
    default: the contour of a 100x100 mask is at most a few hundred vertices,
    which ELLA handles comfortably, and simplification would move the boundary
    the radius is normalised against.
    """
    from skimage import measure

    m = np.asarray(mask)
    if m.ndim == 1:
        m = m.reshape(RENDER_SHAPE)
    binary = (m > 0)
    if not binary.any():
        return None

    # Pad by one pixel so a cell touching the frame edge still yields a closed
    # contour; find_contours otherwise leaves it open along that border.
    padded = np.pad(binary.astype(float), 1, mode="constant", constant_values=0.0)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None

    best = max(contours, key=lambda c: polygon_area(c[:, ::-1]))
    if tolerance > 0:
        best = measure.approximate_polygon(best, tolerance)
    if len(best) < min_vertices:
        return None

    # undo the pad, and swap (row, col) -> (x, y)
    poly = np.empty((len(best), 2), dtype=np.float64)
    poly[:, 0] = best[:, 1] - 1.0   # x = col
    poly[:, 1] = best[:, 0] - 1.0   # y = row
    return poly


def rasterize_polygon(poly: np.ndarray, shape: Tuple[int, int] = RENDER_SHAPE
                      ) -> np.ndarray:
    """Fill a polygon back into a boolean raster. Used only to check the
    round-trip, never in the analysis path."""
    from skimage.draw import polygon as draw_polygon

    out = np.zeros(shape, dtype=bool)
    rr, cc = draw_polygon(np.asarray(poly)[:, 1], np.asarray(poly)[:, 0], shape=shape)
    out[rr, cc] = True
    return out


def polygon_mask_iou(poly: np.ndarray, mask: np.ndarray) -> float:
    """Intersection-over-union of a polygon against the mask it came from."""
    m = np.asarray(mask)
    if m.ndim == 1:
        m = m.reshape(RENDER_SHAPE)
    a = (m > 0)
    b = rasterize_polygon(poly, a.shape)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


# --- the origin --------------------------------------------------------------

def centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """``(x, y)`` centroid of a binary mask, or None if empty."""
    m = np.asarray(mask)
    if m.ndim == 1:
        m = m.reshape(RENDER_SHAPE)
    rr, cc = np.nonzero(m > 0)
    if rr.size == 0:
        return None
    return float(cc.mean()), float(rr.mean())


def point_in_polygon(point: Sequence[float], poly: np.ndarray) -> bool:
    """Even-odd ray-crossing test."""
    x, y = float(point[0]), float(point[1])
    p = np.asarray(poly, dtype=float)
    x1, y1 = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.sum(straddles & (x_cross > x)) % 2 == 1)


def nucleus_center(dapi_mask: Optional[np.ndarray], cell_mask: np.ndarray,
                   poly: Optional[np.ndarray] = None
                   ) -> Tuple[Tuple[float, float], str]:
    """The origin of the radial coordinate, plus which rule produced it.

    Preference order: DAPI centroid -> cell centroid -> polygon vertex mean.
    A centroid that lands *outside* its own polygon (a strongly concave or
    horseshoe cell) is rejected and the next rule tried, because ray-casting
    from outside the boundary returns a radius that is not on [0, 1].

    The rule is returned rather than logged so it can be counted per dataset:
    "how many cells did not use their nucleus" is a number the write-up needs.
    """
    candidates = []
    if dapi_mask is not None:
        c = centroid(dapi_mask)
        if c is not None:
            candidates.append((c, "dapi_centroid"))
    c = centroid(cell_mask)
    if c is not None:
        candidates.append((c, "cell_centroid"))
    if poly is not None and len(poly):
        candidates.append(((float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))),
                           "polygon_mean"))
    if not candidates:
        raise ValueError("no mask carries a single foreground pixel")
    if poly is None:
        return candidates[0]
    for point, rule in candidates:
        if point_in_polygon(point, poly):
            return point, rule
    point, rule = candidates[-1]
    return point, rule + "_outside_polygon"


# --- our own ray-cast, for cross-checking ELLA's -----------------------------

def register_radius(points_xy: np.ndarray, poly: np.ndarray,
                    center_xy: Sequence[float]) -> np.ndarray:
    """Relative radius r = |p - c| / |boundary along that ray - c|.

    This duplicates what ELLA's ``register_cells`` does internally (its
    ``d_c_s`` column). It exists to *check* that number, not to replace it:
    lane 2's profiles are built from ELLA's own registration, and an independent
    implementation agreeing to floating-point noise is what makes that
    defensible.

    r > 1 for a molecule outside the polygon; returned unclamped so the caller
    can count them.
    """
    p = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    v = p - np.array([cx, cy])
    d = np.hypot(v[:, 0], v[:, 1])

    poly = np.asarray(poly, dtype=float)
    a = poly - np.array([cx, cy])
    b = np.roll(poly, -1, axis=0) - np.array([cx, cy])

    out = np.full(len(p), np.nan)
    nonzero = d > 0
    # Ray c + t*u, t >= 0. Segment a + s*(b - a), s in [0, 1]. Solve the 2x2
    # system per (point, edge) pair in one vectorised sweep -- 100x100 frames
    # give a few hundred edges, so the full outer product is small and beats a
    # Python loop over molecules.
    u = np.zeros_like(v)
    u[nonzero] = v[nonzero] / d[nonzero, None]
    e = b - a
    denom = u[:, None, 0] * (-e[None, :, 1]) - u[:, None, 1] * (-e[None, :, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a[None, :, 0] * (-e[None, :, 1]) - a[None, :, 1] * (-e[None, :, 0])) / denom
        s = (u[:, None, 0] * a[None, :, 1] - u[:, None, 1] * a[None, :, 0]) / denom
    hit = np.isfinite(t) & (t > 1e-12) & (s >= 0.0) & (s <= 1.0)
    t_hit = np.where(hit, t, np.inf)
    t_min = t_hit.min(axis=1)
    good = np.isfinite(t_min) & (t_min > 0)
    out[good] = d[good] / t_min[good]
    out[~nonzero] = 0.0
    return out


# --- the zoomed frame vs native coordinates ----------------------------------

def zoom_affine(local_xy: np.ndarray, subcell_xy: np.ndarray
                ) -> Optional[Tuple[float, float, float, float]]:
    """Per-axis least-squares map native -> zoomed: ``(sx, ox, sy, oy)``.

    The zoomed frame is a crop of the cell's bounding box resized to 100x100, so
    the map is axis-aligned and separable. Recovering it from a cell's own
    transcripts -- which carry both ``local_row``/``local_col`` and
    ``subcell_row``/``subcell_col`` -- is what lets the polygon be pushed back
    into native coordinates, and what makes the anisotropy ``sx / sy``
    measurable per cell instead of assumed.

    Returns None when the transcripts do not span both axes (fewer than two
    distinct values on either), where the fit is undetermined.
    """
    lx = np.asarray(local_xy, dtype=float).reshape(-1, 2)
    sx = np.asarray(subcell_xy, dtype=float).reshape(-1, 2)
    if len(lx) < 2:
        return None
    out = []
    for axis in (0, 1):
        a, b = lx[:, axis], sx[:, axis]
        if np.unique(a).size < 2:
            return None
        design = np.stack([a, np.ones_like(a)], axis=1)
        coef, *_ = np.linalg.lstsq(design, b, rcond=None)
        out.extend([float(coef[0]), float(coef[1])])
    return out[0], out[1], out[2], out[3]


def anisotropy(affine: Tuple[float, float, float, float]) -> float:
    """``max(|sx|, |sy|) / min(|sx|, |sy|)`` -- 1.0 is an isotropic resize."""
    sx, _, sy, _ = affine
    a, b = abs(sx), abs(sy)
    lo = min(a, b)
    return float(max(a, b) / lo) if lo > 0 else float("inf")


def unzoom_polygon(poly: np.ndarray, affine: Tuple[float, float, float, float]
                   ) -> np.ndarray:
    """Push a zoomed-frame polygon back into native coordinates."""
    sx, ox, sy, oy = affine
    out = np.asarray(poly, dtype=float).copy()
    out[:, 0] = (out[:, 0] - ox) / sx
    out[:, 1] = (out[:, 1] - oy) / sy
    return out
