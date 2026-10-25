"""Build ELLA's input from a ``ControlSet``, and read its output back.

Runs in ``subcellspace_env``. It never imports ELLA -- the model lives in its
own environment behind ``adapters/ella.py``, and these two halves talk through
files.

Why the points come from a ``ControlSet``
-----------------------------------------
``expr`` is built from the same ``controls.ControlSet`` that feeds SubCellSpace
and PointNet, for the real arm as well as the shuffled one. That is not a
convenience: it means ELLA is fed literally the same molecules as every other
method, and the set carries a ``fingerprint`` that is stamped into the export
and travels into every results row. Rule 2 of the fairness contract becomes a
checkable property rather than a promise.

The interchange format is parquet + npz + JSON rather than a pickle, because the
two environments are numpy 1.23 / pandas 1.x on one side and numpy 2.4 /
pandas 3.0 on the other, and a pickle does not survive that.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .geometry import RENDER_SHAPE, mask_to_polygon, nucleus_center, polygon_mask_iou


# --- where the masks come from -----------------------------------------------

class H5adMasks:
    """Per-cell 100x100 rasters out of ``obsm/cell_mask`` / ``obsm/dapi_mask``.

    Reads one row at a time through h5py. Never ``read_h5ad(backed='r')`` and
    never a whole-array ``read_elem``: those masks are 12.3 GB each on the APEX
    file and materialising them is the 88 GB blow-up this project has already
    hit once.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._index = None

    def _row_index(self) -> Dict[str, int]:
        if self._index is None:
            import h5py
            from anndata.experimental import read_elem
            with h5py.File(self.path, "r") as f:
                obs = read_elem(f["obs"])
            self._index = {str(k): i for i, k in enumerate(obs.index.astype(str))}
        return self._index

    def iter_masks(self, cells: Sequence[str], raw: bool = False
                   ) -> Iterable[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
        """Yield ``(cell, cell_mask, dapi_mask)`` in row order, block by block.

        ``raw=False`` (the default) yields boolean masks, which is what the
        geometry path wants. ``raw=True`` yields the stored float64 values
        untouched: ``dapi_mask`` is *not* binary (599 distinct values in the
        first three APEX rows), and the CVAE's second channel is that graded
        image, so thresholding it would feed the model something it was never
        trained on.

        The masks are stored gzip-compressed with chunk shape ``(603, 79)``, so
        one row spans 127 chunks and pulling it decompresses ~48 MB: measured
        74 ms per cell, which is 7.7 hours over the whole file. A *block* of 603
        consecutive rows costs those same 127 decompressions, so reading whole
        chunk-rows and picking out the wanted cells turns the full 154,296-cell
        pass into ~40 s. Only blocks containing a requested cell are touched, and
        one block (48 MB) is resident at a time.
        """
        import h5py
        idx = self._row_index()
        rows = sorted((idx[c], str(c)) for c in cells if str(c) in idx)
        if not rows:
            return
        with h5py.File(self.path, "r") as f:
            cm = f["obsm/cell_mask"]
            has_dapi = "obsm/dapi_mask" in f
            dm = f["obsm/dapi_mask"] if has_dapi else None
            block = (cm.chunks[0] if cm.chunks else 1024)
            i = 0
            while i < len(rows):
                start = (rows[i][0] // block) * block
                stop = min(start + block, cm.shape[0])
                j = i
                while j < len(rows) and rows[j][0] < stop:
                    j += 1
                cblk = np.asarray(cm[start:stop])
                dblk = np.asarray(dm[start:stop]) if has_dapi else None
                if not raw:
                    cblk = cblk > 0
                    dblk = (dblk > 0) if dblk is not None else None
                for k in range(i, j):
                    r, cell = rows[k]
                    yield (cell,
                           cblk[r - start].reshape(RENDER_SHAPE),
                           dblk[r - start].reshape(RENDER_SHAPE) if dblk is not None else None)
                del cblk, dblk
                i = j


class TreeMasks:
    """Per-observation masks from a simulated coordinate tree.

    ``<tree>/cell_mask_zoomed/<stem>_cell-mask.tif`` and
    ``<tree>/dapi_zoomed/<stem>_DAPI.tif``. Paths are constructed rather than
    listed -- the directories hold 805,549 entries and walking one to find a
    subset costs more than the reads do.
    """

    def __init__(self, tree, cell_dir: str = "cell_mask_zoomed",
                 dapi_dir: str = "dapi_zoomed"):
        self.tree = Path(tree)
        self.cell_dir = self.tree / cell_dir
        self.dapi_dir = self.tree / dapi_dir

    def iter_masks(self, cells: Sequence[str], raw: bool = False
                   ) -> Iterable[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
        """Yield ``(stem, cell_mask, dapi)`` as stored in the TIFs.

        ``raw`` is accepted for interface parity with
        :meth:`H5adMasks.iter_masks` and is a no-op: the TIFs are yielded
        unchanged either way, which is what every current caller already
        gets. A caller that needs a binary nucleus thresholds it itself --
        see :func:`build_geometry_with_nucleus` -- rather than relying on
        this flag.
        """
        from skimage import io
        for stem in cells:
            cp = self.cell_dir / (stem + "_cell-mask.tif")
            if not cp.is_file():
                continue
            cell_mask = np.asarray(io.imread(cp))
            dp = self.dapi_dir / (stem + "_DAPI.tif")
            dapi = np.asarray(io.imread(dp)) if dp.is_file() else None
            yield stem, cell_mask, dapi


# --- geometry for a set of cells ---------------------------------------------

def build_geometry(masks, cells: Sequence[str], iou_sample: int = 200,
                   seed: int = 101) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """``{cell: polygon}`` plus a per-cell report.

    The report carries the centre rule, the vertex count and -- for a random
    sample -- the polygon/mask IoU, so the quality of the raster-to-polygon step
    is a measured distribution rather than an assumption.
    """
    rng = np.random.RandomState(seed)
    cells = [str(c) for c in cells]
    check = set(rng.choice(len(cells), size=min(iou_sample, len(cells)),
                           replace=False).tolist()) if cells else set()
    position = {c: i for i, c in enumerate(cells)}

    polys: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for cell, cell_mask, dapi in masks.iter_masks(cells):
        poly = mask_to_polygon(cell_mask)
        if poly is None:
            rows.append({"cell": cell, "ok": False, "reason": "no_contour",
                         "n_vertices": 0, "center_rule": "", "center_x": np.nan,
                         "center_y": np.nan, "iou": np.nan,
                         "cell_area_px": int((np.asarray(cell_mask) > 0).sum())})
            continue
        (cx, cy), rule = nucleus_center(dapi, cell_mask, poly)
        polys[cell] = poly
        rows.append({
            "cell": cell, "ok": True, "reason": "", "n_vertices": int(len(poly)),
            "center_rule": rule, "center_x": cx, "center_y": cy,
            "iou": (polygon_mask_iou(poly, cell_mask)
                    if position.get(cell, -1) in check else np.nan),
            "cell_area_px": int((np.asarray(cell_mask) > 0).sum()),
        })
    return polys, pd.DataFrame(rows)


def build_geometry_with_nucleus(masks, cells, iou_sample: int = 200,
                                seed: int = 101, dapi_threshold: float = 0.0,
                                min_vertices: int = 8, clip_to_cell: bool = True):
    """``({cell: cell_poly}, {cell: nucleus_poly}, report)`` for a set of cells.

    The nucleus sibling of :func:`build_geometry`, which builds the *cell*
    polygon only -- it hands the DAPI raster to :func:`nucleus_center` and never
    contours it. A boundary-based method that scores transcripts against the
    nuclear envelope (Bento's RNAforest asks for exactly this: molecule
    coordinates, a cell boundary and a nucleus boundary) needs the second
    polygon, so it is built here rather than in a notebook.

    Two decisions are made explicit rather than inherited:

    * **The DAPI threshold.** ``obsm/dapi_mask`` is *not* binary -- it holds
      graded float64 values (599 distinct in the first three APEX rows) -- and
      ``mask_to_polygon`` binarises with a hardcoded ``> 0``. Reading the masks
      with ``raw=True`` and thresholding here makes the cut a recorded
      parameter, and ``dapi_threshold=0.0`` reproduces the ``> 0`` convention
      every other caller already uses.
    * **The nucleus is clipped to the cell** (``clip_to_cell``, on by default).
      The DAPI raster genuinely spills outside the cell raster for **67.6% of
      ileum cells** (median 9 px, max 488) -- the two masks are produced
      independently and are not nested. A nucleus cannot lie outside its own
      cell, and a boundary-based consumer will not accept one that does: Bento's
      ``_sjoin_shapes`` maps a nucleus to a cell with ``predicate="covers"`` and
      a 1e-5 px slack, so an overhanging nucleus is silently dropped and *five of
      its thirteen features go NaN* -- measured at 67.24% of ileum pairs before
      this clip. Clipping on the raster, before contouring, is exact and costs a
      median 1.10% of nucleus area (mean 1.94%, max 14.04%). The per-cell cost is
      recorded as ``nuc_clipped_px`` so the write-up quotes a measurement.

    * **A missing nucleus is not a missing cell.** ``mask_to_polygon`` returns
      None for an empty mask or a boundary under ``min_vertices``, which for a
      small nucleus is a real outcome. Such a cell keeps its entry in
      ``cell_polys`` and its pairs downstream; only ``nucleus_polys`` lacks it,
      and the report says why. Dropping the cell instead would shrink the
      scored pair set and move every other method's numbers.

    The report carries both IoUs on a seeded random sample, so the quality of
    each raster-to-polygon step is a measured distribution rather than a claim.
    """
    rng = np.random.RandomState(seed)
    cells = [str(c) for c in cells]
    check = set(rng.choice(len(cells), size=min(iou_sample, len(cells)),
                           replace=False).tolist()) if cells else set()
    position = {c: i for i, c in enumerate(cells)}

    cell_polys: Dict[str, np.ndarray] = {}
    nuc_polys: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for cell, cell_mask, dapi in masks.iter_masks(cells, raw=True):
        sampled = position.get(cell, -1) in check
        cm = np.asarray(cell_mask)
        rec: Dict[str, Any] = {
            "cell": cell,
            "cell_area_px": int((cm > 0).sum()),
            "dapi_threshold": float(dapi_threshold),
        }

        poly = mask_to_polygon(cm, min_vertices=min_vertices)
        if poly is None:
            rec.update({"ok": False, "reason": "no_contour", "n_vertices": 0,
                        "center_rule": "", "center_x": np.nan, "center_y": np.nan,
                        "iou": np.nan, "nuc_ok": False,
                        "nuc_reason": "cell_dropped", "nuc_n_vertices": 0,
                        "nuc_area_px": 0, "nuc_iou": np.nan,
                        "nuc_clipped_px": 0})
            rows.append(rec)
            continue

        (cx, cy), rule = nucleus_center(
            None if dapi is None else (np.asarray(dapi) > dapi_threshold),
            cm, poly)
        cell_polys[cell] = poly
        rec.update({"ok": True, "reason": "", "n_vertices": int(len(poly)),
                    "center_rule": rule, "center_x": cx, "center_y": cy,
                    "iou": polygon_mask_iou(poly, cm) if sampled else np.nan})

        if dapi is None:
            rec.update({"nuc_ok": False, "nuc_reason": "no_dapi_channel",
                        "nuc_n_vertices": 0, "nuc_area_px": 0,
                        "nuc_iou": np.nan, "nuc_clipped_px": 0})
        else:
            nmask = np.asarray(dapi) > dapi_threshold
            n_raw = int(nmask.sum())
            if clip_to_cell:
                nmask = nmask & (cm > 0)
            rec["nuc_area_px"] = int(nmask.sum())
            rec["nuc_clipped_px"] = n_raw - int(nmask.sum())
            npoly = mask_to_polygon(nmask, min_vertices=min_vertices)
            if npoly is None:
                rec.update({
                    "nuc_ok": False, "nuc_n_vertices": 0, "nuc_iou": np.nan,
                    "nuc_reason": ("empty_after_threshold" if not nmask.any()
                                   else "under_min_vertices")})
            else:
                nuc_polys[cell] = npoly
                rec.update({"nuc_ok": True, "nuc_reason": "",
                            "nuc_n_vertices": int(len(npoly)),
                            "nuc_iou": (polygon_mask_iou(npoly, nmask)
                                        if sampled else np.nan)})
        rows.append(rec)

    return cell_polys, nuc_polys, pd.DataFrame(rows)


# --- the export --------------------------------------------------------------

def split_keys(obs_keys: Sequence[str], kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """``obs_key`` -> ``(cell, gene)``.

    ``cell_gene``  ``"<cell>__<gene>"``  (real data: many genes per cell)
    ``image_stem`` the stem itself       (simulated: one gene per cell, so the
                                          gene is a constant and the *cell* is
                                          the observation)
    """
    keys = np.asarray([str(k) for k in obs_keys], dtype=object)
    if kind == "cell_gene":
        parts = np.array([k.rsplit("__", 1) for k in keys], dtype=object)
        return parts[:, 0].astype(object), parts[:, 1].astype(object)
    if kind == "image_stem":
        return keys, np.full(len(keys), "sim", dtype=object)
    raise ValueError("kind must be 'cell_gene' or 'image_stem'")


def write_input(points: pd.DataFrame, masks, out_dir, *, index_kind: str,
                dataset: str, arm: str, fingerprint: str = "",
                cell_type: str = "cell", max_cells: Optional[int] = None,
                gene_panel: Optional[Sequence[str]] = None,
                min_umi_per_cell: int = 1, seed: int = 101,
                iou_sample: int = 200,
                extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write ``expr.parquet`` + ``cell_poly.npz`` + ``meta.json`` for the adapter.

    ``points`` is a ``ControlSet.points`` table (``obs_key``, ``row``, ``col``,
    ``in_dapi``). ``arm`` is ``"real"`` or ``"control"`` and is recorded, not
    inferred, so a mislabelled export is visible in the metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = points.copy()
    cell, gene = split_keys(df["obs_key"].values, index_kind)
    df["cell"], df["gene"] = cell, gene

    n_before = len(df)
    if gene_panel is not None:
        # The Xenium panel carries control codewords alongside the 277 real
        # genes (a 300-cell APEX slice shows 541 distinct `gene` strings). They
        # are not in the ground truth, nothing else is scored on them, and
        # leaving them in would both inflate ELLA's BY-FDR denominator and pay
        # nhpp_prepare's per-gene cost twice over for nothing.
        df = df[df["gene"].isin(set(map(str, gene_panel)))].copy()
        if df.empty:
            raise ValueError("no molecule left after restricting to the gene panel")

    counts = df.groupby("cell", sort=False).size()
    keep_cells = counts[counts >= min_umi_per_cell].index
    if max_cells is not None and len(keep_cells) > max_cells:
        # deterministic subset: sorted, then strided, so it spans the file
        # rather than taking the first block of one tile
        ordered = np.sort(np.asarray(keep_cells, dtype=object))
        step = max(1, len(ordered) // max_cells)
        keep_cells = ordered[::step][:max_cells]
    df = df[df["cell"].isin(set(map(str, keep_cells)))].copy()

    polys, report = build_geometry(masks, sorted(df["cell"].unique()),
                                   iou_sample=iou_sample, seed=seed)
    df = df[df["cell"].isin(polys.keys())].copy()
    if df.empty:
        raise ValueError("no cell survived geometry construction")

    centers = report.set_index("cell")[["center_x", "center_y"]]
    sc_total = df.groupby("cell", sort=False).size().rename("sc_total")

    expr = pd.DataFrame({
        "type": cell_type,
        "cell": df["cell"].astype(str).values,
        "gene": df["gene"].astype(str).values,
        "x": df["col"].to_numpy(dtype=np.float64),   # x = col, y = row
        "y": df["row"].to_numpy(dtype=np.float64),
        "umi": np.ones(len(df), dtype=np.int64),
    })
    expr["centerX"] = centers.loc[expr["cell"], "center_x"].to_numpy()
    expr["centerY"] = centers.loc[expr["cell"], "center_y"].to_numpy()
    expr["sc_total"] = sc_total.loc[expr["cell"]].to_numpy()
    expr["obs_key"] = df["obs_key"].astype(str).values

    expr.to_parquet(out_dir / "expr.parquet", index=False)

    cells_sorted = sorted(polys)
    verts = np.concatenate([polys[c] for c in cells_sorted]).astype(np.float32)
    offsets = np.cumsum([0] + [len(polys[c]) for c in cells_sorted]).astype(np.int64)
    np.savez_compressed(out_dir / "cell_poly.npz",
                        cells=np.asarray(cells_sorted, dtype=np.str_),
                        offsets=offsets, verts=verts)
    report.to_parquet(out_dir / "geometry_report.parquet", index=False)

    iou = report["iou"].dropna()
    meta = {
        "schema": 1, "dataset": dataset, "arm": arm, "index_kind": index_kind,
        "cell_type": cell_type, "frame": "zoomed_100x100",
        "control_fingerprint": fingerprint,
        "n_cells": len(cells_sorted), "n_genes": int(expr["gene"].nunique()),
        "n_molecules": int(len(expr)),
        "n_molecules_dropped_off_panel": int(n_before - len(df)),
        "gene_panel_applied": gene_panel is not None,
        "n_cells_dropped_no_contour": int((~report["ok"]).sum()),
        "center_rules": report.loc[report["ok"], "center_rule"].value_counts().to_dict(),
        "iou_n_checked": int(len(iou)),
        "iou_median": float(iou.median()) if len(iou) else None,
        "iou_min": float(iou.min()) if len(iou) else None,
        "seed": seed,
    }
    meta.update(extra or {})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return meta


# --- reading the adapter's output back ---------------------------------------

def read_gene_table(path) -> pd.DataFrame:
    """ELLA's per-gene results, in the shape ``sweep.py`` consumes.

    Adds ``ella_neglog10p`` as the score column and ``passes`` as ELLA's own
    call (``pv_fdr <= 0.05``), so the fixed-threshold arm reproduces ELLA's
    decision rule rather than one we chose for it.
    """
    df = pd.read_csv(path)
    df["gene_name"] = df["gene_name"].astype(str)
    with np.errstate(divide="ignore"):
        df["ella_neglog10p"] = -np.log10(np.clip(df["pv_fdr"].to_numpy(dtype=float),
                                                 1e-300, None))
    df["passes"] = df["pv_fdr"].to_numpy(dtype=float) <= 0.05
    return df


# --- the native-coordinate handicap check ------------------------------------

def zoom_affines(local_points: pd.DataFrame, zoomed_points: pd.DataFrame
                 ) -> Tuple[Dict[str, tuple], pd.DataFrame]:
    """Per-cell map from native to zoomed coordinates, fitted from transcripts.

    ``detected_transcripts`` carries each molecule twice -- once as
    ``local_row``/``local_col`` in the tile's native pixel grid and once as
    ``subcell_row``/``subcell_col`` in the per-cell 100x100 crop -- so the crop's
    scale and offset can be recovered per cell by least squares, with no access
    to the cropping code. Two things fall out: the anisotropy of the resize
    becomes a measurable distribution, and the polygon can be pushed back into
    native coordinates so ELLA can be run the way its authors intended.

    Both frames must be given in the same row order (two ``extract_from_h5ad``
    calls over the same cells return exactly that).
    """
    from .geometry import anisotropy, zoom_affine

    df = pd.DataFrame({
        "cell": local_points["obs_key"].astype(str).str.rsplit("__", n=1).str[0],
        "lx": local_points["col"].to_numpy(dtype=float),
        "ly": local_points["row"].to_numpy(dtype=float),
        "zx": zoomed_points["col"].to_numpy(dtype=float),
        "zy": zoomed_points["row"].to_numpy(dtype=float)})
    affines, rows = {}, []
    for cell, g in df.groupby("cell", sort=False):
        aff = zoom_affine(np.stack([g["lx"], g["ly"]], 1), np.stack([g["zx"], g["zy"]], 1))
        if aff is None:
            rows.append({"cell": cell, "n": len(g), "ok": False, "anisotropy": np.nan,
                         "residual_px": np.nan})
            continue
        sx, ox, sy, oy = aff
        resid = float(np.max(np.abs(np.concatenate(
            [sx * g["lx"] + ox - g["zx"], sy * g["ly"] + oy - g["zy"]]))))
        affines[cell] = aff
        rows.append({"cell": cell, "n": len(g), "ok": True,
                     "anisotropy": anisotropy(aff), "residual_px": resid})
    return affines, pd.DataFrame(rows)


def write_native_input(local_points: pd.DataFrame, masks, affines: Dict[str, tuple],
                       out_dir, *, index_kind: str, dataset: str, arm: str,
                       cell_type: str = "cell", gene_panel: Optional[Sequence[str]] = None,
                       seed: int = 101, extra: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """The same export, in native pixel coordinates instead of the zoomed frame.

    Molecules are their native ``local_row``/``local_col``; the polygon and the
    nucleus centre are the zoomed ones pushed back through the inverse affine.
    There is **no shuffled control in native coordinates**, so an export made
    here carries no fingerprint and must never enter the cross-method
    comparison -- it exists only to answer "was ELLA handicapped by the frame?"
    """
    from .geometry import unzoom_polygon

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = local_points.copy()
    cell, gene = split_keys(df["obs_key"].values, index_kind)
    df["cell"], df["gene"] = cell, gene
    if gene_panel is not None:
        df = df[df["gene"].isin(set(map(str, gene_panel)))].copy()
    df = df[df["cell"].isin(set(affines))].copy()

    polys_z, report = build_geometry(masks, sorted(df["cell"].unique()),
                                     iou_sample=0, seed=seed)
    centers = report.set_index("cell")[["center_x", "center_y"]]

    polys, cx_n, cy_n = {}, {}, {}
    for c, poly in polys_z.items():
        aff = affines.get(c)
        if aff is None:
            continue
        polys[c] = unzoom_polygon(poly, aff)
        sx, ox, sy, oy = aff
        cx_n[c] = (float(centers.loc[c, "center_x"]) - ox) / sx
        cy_n[c] = (float(centers.loc[c, "center_y"]) - oy) / sy

    df = df[df["cell"].isin(polys)].copy()
    if df.empty:
        raise ValueError("no cell survived the native-frame construction")
    sc_total = df.groupby("cell", sort=False).size().rename("sc_total")

    expr = pd.DataFrame({
        "type": cell_type,
        "cell": df["cell"].astype(str).values,
        "gene": df["gene"].astype(str).values,
        "x": df["col"].to_numpy(dtype=np.float64),
        "y": df["row"].to_numpy(dtype=np.float64),
        "umi": np.ones(len(df), dtype=np.int64),
    })
    expr["centerX"] = expr["cell"].map(cx_n).to_numpy()
    expr["centerY"] = expr["cell"].map(cy_n).to_numpy()
    expr["sc_total"] = sc_total.loc[expr["cell"]].to_numpy()
    expr["obs_key"] = df["obs_key"].astype(str).values
    expr.to_parquet(out_dir / "expr.parquet", index=False)

    cells_sorted = sorted(polys)
    verts = np.concatenate([polys[c] for c in cells_sorted]).astype(np.float32)
    offsets = np.cumsum([0] + [len(polys[c]) for c in cells_sorted]).astype(np.int64)
    np.savez_compressed(out_dir / "cell_poly.npz",
                        cells=np.asarray(cells_sorted, dtype=np.str_),
                        offsets=offsets, verts=verts)
    report.to_parquet(out_dir / "geometry_report.parquet", index=False)

    meta = {"schema": 1, "dataset": dataset, "arm": arm, "index_kind": index_kind,
            "cell_type": cell_type, "frame": "native_pixels",
            "control_fingerprint": "",
            "comparison_safe": False,
            "why_not_comparable": "no shuffled control exists in native coordinates",
            "n_cells": len(cells_sorted), "n_genes": int(expr["gene"].nunique()),
            "n_molecules": int(len(expr)), "seed": seed}
    meta.update(extra or {})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return meta


# --- streaming the whole file, one cell block at a time ----------------------

def count_cells(path) -> int:
    """Number of cells in a h5ad, without reading anything but ``obs``."""
    import h5py
    from anndata.experimental import read_elem
    with h5py.File(Path(path), "r") as f:
        return int(len(read_elem(f["obs"])))


def iter_point_blocks(path, fields: Sequence[str] = ("subcell_row", "subcell_col"),
                      block_cells: int = 5000, cell_chunk: int = 2048,
                      max_cells: Optional[int] = None, start_block: int = 0,
                      with_in_dapi: bool = False):
    """Yield ``(block_index, cell_names, points)`` for one cell block at a time.

    ``controls.extract_from_h5ad`` builds the whole point table in memory. That
    is right for 2,031 cells (4.3M points) and wrong for all 150,151: the APEX
    file holds 321,668,970 transcript records, and a single DataFrame of those
    -- with an ``obs_key`` Python string per row -- is tens of gigabytes before
    anything is computed with it.

    So this reuses the same awkward-array readers ``controls`` already has, and
    yields per block. Nothing here duplicates the tricky part; the ragged
    ``ListOffsetArray`` / ``IndexedOptionArray`` handling stays in one place.

    ``with_in_dapi`` defaults False: ELLA takes its origin from the DAPI mask
    centroid, so the per-molecule nucleus flag -- the expensive half -- is not
    needed and is skipped.
    """
    import h5py
    from anndata.experimental import read_elem

    from .controls import (POINT_COLUMNS, _dapi_lookup, _form_fields, _read_chunk)

    path = Path(path)
    fields = tuple(fields)
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        cell_names = np.asarray(obs.index.astype(str))
        n_cells = len(cell_names) if max_cells is None else min(max_cells, len(cell_names))
        dt = f["obsm/detected_transcripts"]
        specs = _form_fields(dt)
        missing = [x for x in fields if x not in specs]
        if missing:
            raise KeyError("fields {} not in detected_transcripts".format(missing))
        field_specs = [specs[x] for x in fields]

        starts = list(range(0, n_cells, block_cells))
        for bi, bstart in enumerate(starts):
            if bi < start_block:
                continue
            bstop = min(bstart + block_cells, n_cells)
            keys, rows, cols, flags = [], [], [], []
            for s in range(bstart, bstop, cell_chunk):
                e = min(s + cell_chunk, bstop)
                genes, rr, cc = _read_chunk(dt, field_specs, specs["gene"], s, e)
                for local in range(e - s):
                    r, c, g = rr[local], cc[local], genes[local]
                    if r is None or c is None or len(r) == 0:
                        continue
                    cell = cell_names[s + local]
                    gs = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                                     for x in g])
                    keys.append(np.char.add(np.char.add(cell, "__"), gs))
                    rows.append(np.asarray(r))
                    cols.append(np.asarray(c))
                    flags.append(_dapi_lookup(f, s + local, r, c) if with_in_dapi
                                 else np.zeros(len(r), dtype=bool))
            points = (pd.DataFrame({"obs_key": np.concatenate(keys),
                                    "row": np.concatenate(rows).astype(np.int16),
                                    "col": np.concatenate(cols).astype(np.int16),
                                    "in_dapi": np.concatenate(flags)})
                      if keys else pd.DataFrame(columns=POINT_COLUMNS))
            yield bi, cell_names[bstart:bstop], points


def block_fingerprint(points: pd.DataFrame) -> str:
    """sha1 over one block's canonically-sorted points.

    NOT a ``ControlSet.fingerprint``: that one sorts the *whole* point set, and
    at 316M rows a global sort is not worth its cost. The driver folds these
    per-block digests into one hash and records
    ``fingerprint_scheme="blockwise_sha1_v1"`` so the two are never mistaken for
    each other. Comparing a blockwise digest to a ControlSet fingerprint is
    meaningless; comparing two blockwise digests over the same blocks is not.
    """
    import hashlib
    cols = ["obs_key", "row", "col"]
    df = points[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)
    h = hashlib.sha1()
    h.update(",".join(cols).encode())
    h.update(np.ascontiguousarray(df["obs_key"].astype(str).values.astype("U")).tobytes())
    h.update(np.ascontiguousarray(df["row"].to_numpy(dtype=np.int16)).tobytes())
    h.update(np.ascontiguousarray(df["col"].to_numpy(dtype=np.int16)).tobytes())
    return h.hexdigest()
