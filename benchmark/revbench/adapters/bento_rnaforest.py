"""Bento / RNAforest adapter -- runs in Bento's own environment, not ours.

    /mnt/data/miniconda3/envs/bento_env/bin/python bento_rnaforest.py \
        --points       /mnt/.../work/apex_pc/real/block_0000.parquet \
        --cell-poly    /mnt/.../artifacts/bento/geom/apex_hek/real/block_0000_cell.npz \
        --nucleus-poly /mnt/.../artifacts/bento/geom/apex_hek/real/block_0000_nuc.npz \
        --out          /mnt/.../artifacts/bento/apex_hek/real/block_0000.npz \
        --labels       /mnt/.../artifacts/bento/apex_hek/real/block_0000_labels.parquet \
        --dataset apex_hek --index-kind cell_gene

Like ``adapters/pointcloud_vae.py`` and ``adapters/ella.py`` this imports
**nothing** from ``revbench``: it sees python 3.11 / pandas 2.3 / sklearn 1.5 /
xgboost 2.0.3, while the harness sees python 3.9 / pandas 1.x / sklearn 1.2. The
two halves agree only on the file formats, and the ``schema`` field in the
sidecar is what keeps the duplicated ``write_dump`` below honest against
``representations.py``'s original.

Three things about this tool that the call site cannot guess
------------------------------------------------------------

**1. The shipped model needs xgboost 2.0.3.** ``bento/models/rf_calib_20220514.pkl``
is a ``CalibratedClassifierCV`` over a ``OneVsRestClassifier`` of five
``Pipeline(StandardScaler, XGBRFClassifier)`` -- despite the ``rf_`` name it is
XGBoost's random-forest mode, and xgboost >= 2.1 refuses the 2022 serialization
buffer with "Invalid serialization file". Bento's own ``lp`` also needs the
``cls.estimator = cls.base_estimator`` shim, reproduced here for the label lane.

**2. Every cell in this project shares one coordinate frame.** Observations live
in a per-cell 100x100 crop-and-resize frame, so cell A's polygon and cell B's
polygon occupy the *same* square. Bento indexes points to shapes with a real
geometric ``sjoin``, which in that frame would assign every point to every cell.
Each cell is therefore translated onto its own tile of a grid before Bento sees
it (``--pitch``, default 160 > 100 + margin). All thirteen features are measured
relative to the cell's own boundary, centroid, radius, span and area, so a rigid
translation cannot move any of them -- ``--selftest`` proves that rather than
assuming it, by scoring the same cells at two different pitches.

**3. ``num_workers`` is not the lever.** Measured on Bento's own sample:
35.9 pairs/s at ``num_workers=1``, 49.4 at 4, 48.9 at 16. It plateaus, because
``analyze_points`` builds one dask bag partition per *cell* and the per-partition
work is small. The parallelism therefore belongs at the block level -- many of
these processes at once, one worker each -- which is exactly how
``00_prepare`` drives the PointNet adapter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

DUMP_SCHEMA = 1
INDEX_KINDS = ("obs_names", "canonical_key", "template", "cell_gene", "image_stem")

#: The thirteen features RNAforest consumes, in the order the shipped model was
#: fitted on. Copied from ``bento/tools/_lp.py``, whose own comment reads "Do not
#: change order of features!" -- reproduced here so a Bento version bump that
#: reorders them is caught by the assertion in `extract_features`, not silently
#: scored against a permuted matrix.
PATTERN_FEATURES = (
    "cell_boundaries_inner_proximity",
    "nucleus_boundaries_inner_proximity",
    "nucleus_boundaries_outer_proximity",
    "cell_boundaries_inner_asymmetry",
    "nucleus_boundaries_inner_asymmetry",
    "nucleus_boundaries_outer_asymmetry",
    "l_max",
    "l_max_gradient",
    "l_min_gradient",
    "l_monotony",
    "l_half_radius",
    "point_dispersion_norm",
    "nucleus_boundaries_dispersion_norm",
)

CELL_KEY = "cell_boundaries"
NUC_KEY = "nucleus_boundaries"
POINTS_KEY = "transcripts"
FEATURE_KEY = "feature_name"


# --- the dump format ---------------------------------------------------------

def write_dump(path, X, obs_index, index_kind, method, dataset, extra=None):
    """Mirror of ``revbench.representations.write_dump``.

    Duplicated deliberately: this module runs where revbench is not importable.
    Keep the arrays and the ``schema`` field in step with that function.
    """
    if index_kind not in INDEX_KINDS:
        raise ValueError("index_kind must be one of %s" % (INDEX_KINDS,))
    path = str(path)
    if not path.endswith(".npz"):
        path = path + ".npz"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    keys = np.asarray([str(k) for k in obs_index], dtype=np.str_)
    if keys.shape[0] != X.shape[0]:
        raise ValueError("obs_index has %d entries, X has %d rows"
                         % (keys.shape[0], X.shape[0]))
    # .npz on the temp name on purpose: savez_compressed appends it otherwise,
    # the rename fails, and every resume silently redoes the work.
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, X=X, obs_index=keys)
    os.replace(tmp, path)
    meta = {"schema": DUMP_SCHEMA, "method": method, "dataset": dataset,
            "index_kind": index_kind, "n": int(X.shape[0]),
            "latent_dim": int(X.shape[1]),
            "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "producer": "revbench.adapters.bento_rnaforest",
            "feature_order": list(PATTERN_FEATURES)}
    meta.update(extra or {})
    with open(path[:-4] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    return path


# --- inputs ------------------------------------------------------------------

def read_polys(path):
    """``{cell: (M, 2) vertices}`` from the cells/offsets/verts npz layout."""
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as npz:
        cells = [str(c) for c in npz["cells"]]
        offsets = np.asarray(npz["offsets"], dtype=np.int64)
        verts = np.asarray(npz["verts"], dtype=np.float64)
    return {c: verts[offsets[i]:offsets[i + 1]] for i, c in enumerate(cells)}


def split_obs_key(obs_keys, index_kind, gene_for_stems):
    """``obs_key`` -> ``(cell, gene)``.

    ``cell_gene`` keys are ``"<cell>__<gene>"`` on a **double** underscore, which
    is the convention every dump and gene table on disk uses. ``image_stem``
    keys are one simulated cell carrying one gene, so the gene is a supplied
    constant and the pair key is the stem itself.
    """
    keys = np.asarray([str(k) for k in obs_keys])
    if index_kind == "cell_gene":
        parts = pd.Series(keys).str.rsplit("__", n=1, expand=True)
        if parts.shape[1] != 2 or parts[1].isna().any():
            bad = keys[pd.Series(keys).str.count("__").to_numpy() == 0][:3]
            raise ValueError("keys without a '__' separator, e.g. %s" % (list(bad),))
        return parts[0].to_numpy(), parts[1].to_numpy()
    return keys, np.full(len(keys), str(gene_for_stems))


# --- the grid ----------------------------------------------------------------

def grid_offsets(cells, pitch):
    """One ``(dx, dy)`` tile per cell, so no two 100x100 frames overlap.

    Deterministic in the sorted cell order, so re-running a block reproduces the
    same layout and the same floating-point arithmetic.
    """
    order = sorted(cells)
    ncol = max(1, int(np.ceil(np.sqrt(len(order)))))
    return {c: ((i % ncol) * pitch, (i // ncol) * pitch)
            for i, c in enumerate(order)}


def build_sdata(points, cell_polys, nuc_polys, pitch):
    """A SpatialData Bento's public ``prep`` will accept, cells tiled on a grid.

    ``points`` needs ``cell``, ``gene``, ``row``, ``col``. ``(x, y) = (col, row)``
    -- the point tables are image indices, Bento speaks x/y, and the swap happens
    once, here, the same way ``revbench.geometry`` does it.
    """
    import geopandas as gpd
    import spatialdata as sd
    from shapely.geometry import Polygon

    have = set(cell_polys)
    keep = points["cell"].isin(have).to_numpy()
    dropped = int((~keep).sum())
    points = points.loc[keep].copy()
    cells = sorted(set(points["cell"].astype(str)) & have)

    off = grid_offsets(cells, pitch)
    dx = points["cell"].map(lambda c: off[c][0]).to_numpy(dtype=float)
    dy = points["cell"].map(lambda c: off[c][1]).to_numpy(dtype=float)

    nuc_col = ("nucleus" if "nucleus" in points.columns
               else ("in_dapi" if "in_dapi" in points.columns else None))
    pts = pd.DataFrame({
        "x": points["col"].to_numpy(dtype=float) + dx,
        "y": points["row"].to_numpy(dtype=float) + dy,
        FEATURE_KEY: pd.Categorical(points["gene"].astype(str)),
    })
    if nuc_col is not None:
        # Carried through so the nucleus assignment can be checked against OUR
        # flag after `prep`. `_sync_points` drops points outside the cell, so a
        # positional comparison against the input frame silently length-mismatches
        # and the check is skipped -- which is how it went unreported on APEX.
        pts["ours_in_nucleus"] = points[nuc_col].to_numpy(dtype=bool)

    cell_geom, cell_idx = [], []
    nuc_geom, nuc_idx = [], []
    for c in cells:
        ox, oy = off[c]
        p = np.asarray(cell_polys[c], dtype=float) + (ox, oy)
        cell_geom.append(Polygon(p))
        cell_idx.append(c)
        if c in nuc_polys:
            q = np.asarray(nuc_polys[c], dtype=float) + (ox, oy)
            nuc_geom.append(Polygon(q))
            nuc_idx.append(c)

    shapes = {CELL_KEY: sd.models.ShapesModel.parse(
        gpd.GeoDataFrame(geometry=cell_geom, index=pd.Index(cell_idx, dtype=str)))}
    if nuc_geom:
        shapes[NUC_KEY] = sd.models.ShapesModel.parse(
            gpd.GeoDataFrame(geometry=nuc_geom, index=pd.Index(nuc_idx, dtype=str)))

    sdata = sd.SpatialData(
        points={POINTS_KEY: sd.models.PointsModel.parse(
            pts, coordinates={"x": "x", "y": "y"}, feature_key=FEATURE_KEY)},
        shapes=shapes)
    return sdata, points, dropped, len(nuc_geom)


# --- running the tool --------------------------------------------------------

def run_lp(sdata, num_workers):
    """``bt.io.prep`` then ``bt.tl.lp`` -- the public API, nothing bypassed."""
    import bento as bt
    shape_keys = [CELL_KEY] + ([NUC_KEY] if NUC_KEY in sdata.shapes else [])
    sdata = bt.io.prep(sdata, points_key=POINTS_KEY, feature_key=FEATURE_KEY,
                       instance_key=CELL_KEY, shape_keys=shape_keys)
    bt.tl.lp(sdata, instance_key=CELL_KEY, nucleus_key=NUC_KEY,
             groupby=FEATURE_KEY, num_workers=num_workers)
    return sdata


def extract_features(sdata):
    """The 13-column matrix, its keys, and the label frames.

    Asserts the feature names and their order against ``PATTERN_FEATURES``: a
    Bento release that renames or reorders them must fail loudly here rather
    than hand back a permuted matrix the shipped model would happily score.
    """
    from bento._constants import PATTERN_NAMES
    tab = sdata.tables["table"]
    fk = "%s_%s_features" % (CELL_KEY, FEATURE_KEY)
    if fk not in tab.uns:
        raise KeyError("no feature table %r; have %s" % (fk, sorted(tab.uns)))
    F = tab.uns[fk]
    missing = [f for f in PATTERN_FEATURES if f not in F.columns]
    if missing:
        raise KeyError("Bento did not produce %d expected feature(s): %s"
                       % (len(missing), missing))
    X = F[list(PATTERN_FEATURES)].to_numpy(dtype=np.float64)
    cells = F[CELL_KEY].astype(str).to_numpy()
    genes = F[FEATURE_KEY].astype(str).to_numpy()
    lp = tab.uns.get("lp")
    lpp = tab.uns.get("lpp")
    return X, cells, genes, lp, lpp, list(PATTERN_NAMES)


def selftest(points, cell_polys, nuc_polys, num_workers, pitch):
    """Are the 13 features invariant to the grid translation? Measure, don't assume."""
    cells = sorted(set(points["cell"].astype(str)) & set(cell_polys))[:12]
    sub = points[points["cell"].isin(cells)].copy()
    print("selftest on %d cells / %d points" % (len(cells), len(sub)))
    got = []
    for p in (pitch, pitch * 3 + 7):
        sdata, _, _, _ = build_sdata(sub, cell_polys, nuc_polys, p)
        X, c, g, _, _, _ = extract_features(run_lp(sdata, num_workers))
        order = np.lexsort((g, c))
        got.append((X[order], c[order], g[order]))
        print("  pitch %-6g -> %s" % (p, X.shape))
    (X1, c1, g1), (X2, c2, g2) = got
    assert list(c1) == list(c2) and list(g1) == list(g2), "key sets differ"
    both = np.isfinite(X1) & np.isfinite(X2)
    nan_same = (np.isnan(X1) == np.isnan(X2)).all()
    delta = np.abs(X1[both] - X2[both])
    print("\n  NaN pattern identical : %s" % bool(nan_same))
    print("  finite cells compared : %d of %d" % (int(both.sum()), X1.size))
    print("  max |delta|           : %.3e" % (delta.max() if delta.size else 0.0))
    ok = bool(nan_same) and (delta.size == 0 or delta.max() < 1e-9)
    print("\n=== TRANSLATION_%s ===" % ("INVARIANT" if ok else "DEPENDENT"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", required=True)
    ap.add_argument("--cell-poly", required=True)
    ap.add_argument("--nucleus-poly", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--method", default="bento_rnaforest")
    ap.add_argument("--index-kind", default="cell_gene", choices=list(INDEX_KINDS))
    ap.add_argument("--gene-for-stems", default="sim",
                    help="feature_name for image_stem inputs (one gene per cell)")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--pitch", type=float, default=160.0)
    ap.add_argument("--selftest", action="store_true",
                    help="check translation invariance on a few cells, then exit")
    a = ap.parse_args(argv)

    t_start = time.time()
    pts = pd.read_parquet(a.points)
    for col in ("obs_key", "row", "col"):
        if col not in pts.columns:
            raise KeyError("points parquet lacks %r; has %s" % (col, list(pts.columns)))
    nuc_flag = ("nucleus" if "nucleus" in pts.columns
                else ("in_dapi" if "in_dapi" in pts.columns else None))
    if nuc_flag is None:
        raise KeyError("points parquet carries neither 'nucleus' nor 'in_dapi'; "
                       "the nucleus channel would be silently dropped")
    cell, gene = split_obs_key(pts["obs_key"].to_numpy(), a.index_kind,
                               a.gene_for_stems)
    pts["cell"], pts["gene"] = cell, gene

    cell_polys = read_polys(a.cell_poly)
    nuc_polys = read_polys(a.nucleus_poly)
    print("points %d | cells with a polygon %d | nucleus polygons %d"
          % (len(pts), len(cell_polys), len(nuc_polys)))

    if a.selftest:
        return selftest(pts, cell_polys, nuc_polys, a.num_workers, a.pitch)

    sdata, used, dropped, n_nuc = build_sdata(pts, cell_polys, nuc_polys, a.pitch)
    if dropped:
        print("dropped %d points on cells with no cell polygon" % dropped)

    t0 = time.time()
    sdata = run_lp(sdata, a.num_workers)
    t_lp = time.time() - t0

    X, cells, genes, lp, lpp, pattern_names = extract_features(sdata)
    if a.index_kind == "cell_gene":
        keys = np.char.add(np.char.add(cells.astype(str), "__"), genes.astype(str))
    else:
        keys = cells.astype(str)

    n_nan_rows = int(np.isnan(X).any(axis=1).sum())
    print("features %s in %.1f s (%.1f pairs/s) | rows with any NaN %d (%.2f%%)"
          % (X.shape, t_lp, X.shape[0] / max(t_lp, 1e-9), n_nan_rows,
             100.0 * n_nan_rows / max(X.shape[0], 1)))

    # --- does Bento's own geometric sjoin agree with our nucleus flag? -------
    # Our in_dapi is authoritative (recomputed at the shuffled position from the
    # DAPI raster); Bento's comes from a point-in-polygon test against the
    # contour of that same raster. They should agree except at the boundary, and
    # the disagreement rate is the honest measure of the polygonisation step.
    agree, n_agree = None, 0
    try:
        p_out = sdata.points[POINTS_KEY].compute()
        if NUC_KEY in p_out.columns and "ours_in_nucleus" in p_out.columns:
            theirs = (p_out[NUC_KEY].astype(str) != "").to_numpy()
            ours = p_out["ours_in_nucleus"].to_numpy(dtype=bool)
            n_agree = int(len(ours))
            agree = float((theirs == ours).mean()) if n_agree else None
            print("nucleus assignment agreement with %s: %.6f (%d points)"
                  % (nuc_flag, agree, n_agree))
        else:
            print("nucleus agreement check unavailable (columns: %s)"
                  % sorted(p_out.columns))
    except Exception as exc:                      # never fail the run on a check
        print("nucleus agreement check skipped: %s" % exc)

    if a.out:
        write_dump(a.out, X, keys, a.index_kind, a.method, a.dataset, extra={
            "n_pairs": int(X.shape[0]), "n_rows_with_nan": n_nan_rows,
            "n_cells": int(len(set(cells))), "n_nucleus_polygons": int(n_nuc),
            "n_points": int(len(used)), "n_points_dropped_no_polygon": int(dropped),
            "points_file": str(a.points), "grid_pitch": float(a.pitch),
            "frame": "zoomed_100x100_gridded",
            "nucleus_channel": "explicit_%s" % nuc_flag,
            "nucleus_sjoin_agreement": agree,
            "nucleus_sjoin_agreement_n": int(n_agree),
            "imputation": "none -- NaN preserved; the caller declares its policy",
            "num_workers": int(a.num_workers),
            "lp_seconds": round(t_lp, 1),
            "bento_version": __import__("bento").__version__,
            "xgboost_version": __import__("xgboost").__version__,
        })
        print("wrote %s" % a.out)

    if a.labels and lp is not None:
        out = pd.DataFrame({"obs_key": keys})
        for p in pattern_names:
            out[p] = np.asarray(lp[p], dtype=np.uint8)
            if lpp is not None:
                out[p + "_p"] = np.asarray(lpp[p], dtype=np.float32)
        out["n_labels"] = out[list(pattern_names)].sum(axis=1).astype(np.uint8)
        os.makedirs(os.path.dirname(os.path.abspath(a.labels)) or ".", exist_ok=True)
        out.to_parquet(a.labels, index=False)
        print("wrote %s (unlabelled pairs %d)"
              % (a.labels, int((out.n_labels == 0).sum())))

    print("total %.1f s" % (time.time() - t_start))
    return 0


if __name__ == "__main__":
    # The main guard is load-bearing, not decoration: analyze_points' dask pool
    # uses spawn, which re-imports this module in every worker.
    sys.exit(main())
