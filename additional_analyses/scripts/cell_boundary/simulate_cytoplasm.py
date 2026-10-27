#!/usr/bin/env python
# Provenance for additional_analyses/cell_boundary_vs_cytoplasmic.ipynb.
# Simulates the two additional patterns (data keys `cytoplasm` and `celledge`) on the 64
# held-out cell templates, writing gene_coords/, rotated/, rotated_dapi/ ... per pattern.
# The example coordinate files in <data>/cell_boundary/examples/gene_coords come from here.
# Expects the original simulation tree (cell and nucleus masks); paths are the original ones.
"""Simulate 2D `cytoplasm` and `celledge` patterns, strictly excluding the nucleus.

    /mnt/data/miniconda3/envs/cvae_david3/bin/python \
        ~/SubCellSpace/additional_analyses/scripts/cell_boundary/simulate_cytoplasm.py --pattern cytoplasm
    /mnt/data/miniconda3/envs/cvae_david3/bin/python \
        ~/SubCellSpace/additional_analyses/scripts/cell_boundary/simulate_cytoplasm.py --pattern celledge

Why this exists
---------------
The question is whether SubCellSpace can separate RNAs distributed randomly
throughout the cytoplasm but not within the nucleus from RNAs at the cell edge,
as distinct from the existing `random` category. simfish has
no such pattern and `simulate_localization_pattern` whitelists its pattern names,
so this builds it directly.

What "cytoplasm" means here, precisely
--------------------------------------
simfish samples in 3D and the model sees 2D max-projections. Because the cell is
roughly twice as thick over the nucleus as elsewhere, a 3D-uniform draw projects a
lot of spots onto the nuclear footprint: measured on the real `in_dapi` column,
`random` is ~46% in-nucleus and even `extranuclear` -- which zeroes the nuclear
*voxels* -- is ~19%.

Both patterns here are **2D-strict**: zero spots on the nuclear footprint.

  cytoplasm  100% uniform over the 2D cell-minus-nucleus mask (prop 1.00)
  celledge   90% uniform over a hard 2px band inside the cell boundary,
             10% uniform over the rest of the cytoplasm (prop 0.90)

They are deliberately a nested pair: the same cytoplasmic background, differing only
in rim enrichment, so the comparison isolates "enriched at the boundary" from
"uniform in the cytoplasm".

`celledge` is 0.90 rather than a pure rim because a hard band at 1.00 would be about
as explicit as `protrusion`, the most detectable class in the set, and separating
that from cytoplasm would prove little. 0.90 matches every existing pattern's
adherence. Its 10% goes over cell-minus-nucleus, NOT over the whole cell as simfish
does -- simfish's whole-cell background is one reason its own `cell-edge` still puts
30% of spots on the nuclear footprint.

Note both differ from `random` in two ways at once, no nuclear footprint *and* no
thickness weighting, which any write-up must say.

Output format matches `even_less_spots` exactly: binary float64 100x100 TIFFs,
sub-pixel float coordinates, `in_dapi` by rounding, raster by `int()` truncation,
and the on-disk filename grammar (which repeats `{it}it` and uses `icell`).
"""
import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from skimage.filters import gaussian

REPO = Path(__file__).resolve().parents[3]   # the SubCellSpace repository
sys.path.insert(0, str(REPO / "training_subcellspace" / "create_simulations"))
import rotationally_augment_simulations as ras  # noqa: E402

SRC = Path("/mnt/data/david/simulated_cells/with_dapi/even_less_spots")
OUT_ROOT = Path("/path/to/work")
TEST_IDS = REPO / "training_subcellspace/create_simulations/test_cell_ids.csv"
GEOMETRY_FROM = "random"          # geometry is byte-identical across patterns
SHAPE = (100, 100)
DEFAULT_PROP = {"cytoplasm": 1.00, "celledge": 0.90}
SIGMA = 1.5
SEED = 101


def assert_patched_skimage():
    from skimage.transform import rotate
    if not isinstance(rotate(np.zeros(SHAPE), 37, resize=True), tuple):
        raise RuntimeError(
            "stock scikit-image: rotate() returned one array. Stage 2 needs the "
            "patched version that returns (warped, translation).\n"
            f"interpreter: {sys.executable}\n"
            "use /mnt/data/miniconda3/envs/cvae_david3/bin/python")


def region_masks(cell, nuc, pattern, band_px):
    """(target, background) boolean masks to sample from.

    `celledge`'s band is the set of cell pixels within `band_px` of a non-cell
    pixel -- the 2D analogue of simfish's `cell_map <= 2` rule, but computed on the
    projected mask, so nothing above or below the nucleus can leak into it.
    """
    from scipy import ndimage as ndi
    cyto = cell & ~nuc
    if pattern == "cytoplasm":
        return cyto, None
    band = cyto & (ndi.distance_transform_edt(cell) <= band_px)
    return band, cyto & ~band


def sample_spots(target, background, nuc, n_spots, prop, rng):
    """`n_spots` sub-pixel positions: `prop` of them in `target`, the rest in
    `background` (or all in `target` when `background` is None).

    A position is kept only when `int()` *and* `round()` of it both land in the
    region it was drawn from: the raster truncates and `in_dapi` rounds, so both
    rules must hold for the pattern to be strictly nucleus-free in the image as
    well as in the coordinate table. Rejection only bites at the region boundary.
    """
    def draw(mask, k):
        rows, cols = np.nonzero(mask)
        got = []
        while len(got) < k:
            j = rng.integers(0, len(rows))
            r = rows[j] + rng.random()
            c = cols[j] + rng.random()
            ri, ci = int(r), int(c)
            rr, cc = int(np.rint(r)), int(np.rint(c))
            if not (0 <= rr < SHAPE[0] and 0 <= cc < SHAPE[1]):
                continue
            if not (mask[ri, ci] and mask[rr, cc]):
                continue
            got.append((r, c, bool(nuc[rr, cc] > 0)))
        return got

    if background is None:
        return draw(target, n_spots)
    n_t = int(n_spots * prop)
    return draw(target, n_t) + draw(background, n_spots - n_t)


def stage1(cell_ids, n_it, rng, pattern, prop, band_px, out):
    for d in ("cell_filtered", "nuclei_filtered", "gene_expression", "gene_coords"):
        (out / d).mkdir(parents=True, exist_ok=True)
    stems = []
    for cid in cell_ids:
        src_cell = SRC / "cell_filtered" / f"{GEOMETRY_FROM}_icell{cid}_cell.tif"
        src_nuc = SRC / "nuclei_filtered" / f"{GEOMETRY_FROM}_icell{cid}_dapi.tif"
        # byte-copy: these are float64 binary TIFFs written under skimage 0.17.2,
        # and re-saving them under 0.19 would silently write bool instead.
        shutil.copyfile(src_cell, out / "cell_filtered" / f"{pattern}_icell{cid}_cell.tif")
        shutil.copyfile(src_nuc, out / "nuclei_filtered" / f"{pattern}_icell{cid}_dapi.tif")

        cell = io.imread(src_cell) > 0
        nuc = io.imread(src_nuc) > 0
        target, background = region_masks(cell, nuc, pattern, band_px)
        for it in range(n_it):
            n_spots = int(rng.integers(10, 101))       # flat 10..100, as on disk
            spots = sample_spots(target, background, nuc, n_spots, prop, rng)
            stem = (f"{pattern}_{it}it_{n_spots}spots_icell{cid}_{it}it_"
                    f"prop{prop:.2f}")
            df = pd.DataFrame(spots, columns=["row", "col", "in_dapi"])
            df.to_csv(out / "gene_coords" / f"{stem}_coords.csv", index=False)
            img = np.zeros(SHAPE, dtype=np.float64)
            img[df["row"].astype(int).values, df["col"].astype(int).values] = 1.0
            io.imsave(out / "gene_expression" / f"{stem}_gene_expression.tif", img,
                      check_contrast=False)
            stems.append(stem)
    return stems


def regenerate(gene_img, angle, dapi=None, cell_mask=None):
    """Stage 2 at a chosen angle -- the published primitives, in published order."""
    kwargs, _ = ras.calc_kwargs(gene_img.shape, SHAPE, angle, first_resize_step=False)
    kwargs = kwargs[:-1]
    coords = ras.convert_image_to_coords(gene_img)
    new = [ras.point2int(ras.register_point(c, kwargs)) for c in coords]
    new = [c for c in new if all(e >= 1 for e in c)]
    new, bbox = ras.cut_out_coords_after_rotation(new, SHAPE)
    zoom = ras.calculate_resize_zoom_factors((bbox[2] - bbox[0], bbox[3] - bbox[1]), SHAPE)
    final = [ras.resize_point_with_transform(c, zoom) for c in new]
    out = {"gene": gaussian(ras.coords_to_img(final, SHAPE), SIGMA), "coords": final}
    for name, img in (("dapi", dapi), ("cell", cell_mask)):
        if img is not None:
            rotated, _ = ras.rotate(img, angle, resize=True)
            out[name] = ras.resize(ras.cut_bbox_from_image(bbox, rotated), SHAPE)
    return out


def stage2(stems, n_rot, rng, pattern, out):
    for d in ("rotated", "rotated_dapi", "rotated_cell_mask", "rotated_coords"):
        (out / d).mkdir(parents=True, exist_ok=True)
    made = 0
    for stem in stems:
        cid = re.search(r"icell(\d+)", stem).group(1)
        g = io.imread(out / "gene_expression" / f"{stem}_gene_expression.tif")
        d = io.imread(out / "nuclei_filtered" / f"{pattern}_icell{cid}_dapi.tif")
        m = io.imread(out / "cell_filtered" / f"{pattern}_icell{cid}_cell.tif")
        for _ in range(n_rot):
            angle = int(rng.integers(1, 361))
            r = regenerate(g, angle, dapi=d, cell_mask=m)
            base = f"{stem}_gene_expression_rotated{angle}_blurred{SIGMA:1.1f}"
            io.imsave(out / "rotated" / f"{base}.tif", r["gene"], check_contrast=False)
            io.imsave(out / "rotated_dapi" / f"{base}_DAPI.tif", r["dapi"],
                      check_contrast=False)
            io.imsave(out / "rotated_cell_mask" / f"{base}_cell-mask.tif", r["cell"],
                      check_contrast=False)
            pd.DataFrame(r["coords"], columns=["row", "col"]).to_csv(
                out / "rotated_coords" / f"{base}_gene-coords.csv", index=False)
            made += 1
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-it", type=int, default=20, help="simulations per cell")
    ap.add_argument("--n-rot", type=int, default=5, help="rotations per simulation")
    ap.add_argument("--limit-cells", type=int, default=None, help="smoke run")
    ap.add_argument("--pattern", choices=("cytoplasm", "celledge"), default="cytoplasm")
    ap.add_argument("--band-px", type=float, default=2.0,
                    help="celledge only: rim thickness in the 100x100 frame")
    ap.add_argument("--prop", type=float, default=None,
                    help="fraction of spots in the target region; default per pattern")
    ap.add_argument("--out", default=None, help="output dir; default /mnt/.../<pattern>/sim")
    args = ap.parse_args()
    prop = args.prop if args.prop is not None else DEFAULT_PROP[args.pattern]
    out = Path(args.out) if args.out else OUT_ROOT / args.pattern / "sim"

    assert_patched_skimage()
    cell_ids = [int(x) for x in TEST_IDS.read_text().split()]
    if args.limit_cells:
        cell_ids = cell_ids[:args.limit_cells]
    print(f"pattern {args.pattern} | prop {prop:.2f}"
          + (f" | band {args.band_px:g}px" if args.pattern == "celledge" else ""),
          flush=True)
    print(f"{len(cell_ids)} held-out cells x {args.n_it} it x {args.n_rot} rot "
          f"= {len(cell_ids)*args.n_it*args.n_rot} observations", flush=True)

    rng = np.random.default_rng(SEED)
    stems = stage1(cell_ids, args.n_it, rng, args.pattern, prop, args.band_px, out)
    print(f"stage 1: {len(stems)} base simulations -> {out}", flush=True)
    made = stage2(stems, args.n_rot, rng, args.pattern, out)
    print(f"stage 2: {made} rotated observations", flush=True)


if __name__ == "__main__":
    main()
