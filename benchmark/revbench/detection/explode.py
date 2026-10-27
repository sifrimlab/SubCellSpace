"""Explode 3-D per-cell embedding layers into a matched real/control pair.

The detection test compares each gene's real embeddings against its shuffled
control, and the published ``classify_all_genes`` asserts the two tables share
an index *in the same order*. No file on disk ships such a pair: every exploded
h5ad carries the real embedding only, with an empty ``layers`` group. So the
pair has to be built from the per-cell source.

Doing that with the published
``make_paper_figures/src/adata_utils.py::reshape_and_filter_anndata_with_index_and_mapping``
works for the 408-cell ileum file but not for the 154296-cell APEX file, where
each 3-D layer is 10 GB and the function's ``np.repeat(adata.obs.values,
n_genes, axis=0)`` would materialise an 83.5M-row object-dtype frame. This
module keeps the published function as the fast path for small inputs and
chunks over cells for large ones.

Retention: a row is kept when the *real* embedding is not all-zero. An all-zero
row means the (cell, gene) pair fell outside the ``10 < n_spots < 101`` window,
i.e. was never measured. The mask is taken from the real layer and applied
verbatim to the control, so the two stay aligned by construction; any
divergence is counted and reported rather than assumed away.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ..config import (CACHE_DIR, N_SPOTS_BINS, N_SPOTS_LABELS_FIXED,
                      N_SPOTS_LABELS_PUBLISHED, code_version)

#: Cells per chunk. The APEX layers are gzip with HDF5 chunks (2411, 17, 1), so
#: cell slices that are multiples of 2411 are chunk-aligned; 4822 cells holds
#: both layers in ~626 MB.
HDF5_CELL_CHUNK = 2411
DEFAULT_CHUNK_CELLS = 2 * HDF5_CELL_CHUNK

#: Below this many cell-gene pairs, delegate to the published function.
DIRECT_MAX_PAIRS = 5_000_000

#: obs columns the exploded pair carries by default. Deliberately minimal: at
#: 8.4M rows every extra int64 column costs ~67 MB.
DEFAULT_OBS_COLS = ("cell_id",)


def _read_metadata(path):
    """Read only ``obs`` and ``var``.

    ``anndata.read_h5ad(path, backed="r")`` leaves ``X`` on disk but eagerly
    materialises ``obsm`` -- and this file's ``obsm`` holds ``cell_mask`` and
    ``dapi_mask`` at 12.3 GB each. Reading the two tables directly keeps the
    metadata step at a few hundred MB instead of 25 GB.
    """
    import h5py
    from anndata.experimental import read_elem
    with h5py.File(path, "r") as f:
        return read_elem(f["obs"]), read_elem(f["var"])


def _n_spots_intervals(n_spots: np.ndarray) -> Dict[str, pd.Categorical]:
    published = pd.cut(n_spots.astype(float), bins=N_SPOTS_BINS,
                       labels=N_SPOTS_LABELS_PUBLISHED)
    fixed = pd.cut(n_spots.astype(float), bins=N_SPOTS_BINS,
                   labels=N_SPOTS_LABELS_FIXED)
    return {"n_spots_interval": published, "n_spots_interval_fixed": fixed}


def _build_obs(cell_idx, gene_idx, n_spots, cell_names, gene_names,
               source_obs: Optional[pd.DataFrame], obs_cols: Sequence[str]):
    """Assemble the exploded obs from integer codes, without repeating values."""
    obs = pd.DataFrame({
        "original_cell_index": cell_idx.astype(np.int32),
        "original_gene_index": gene_idx.astype(np.int32),
        "n_spots": n_spots,
    })
    obs["original_obs_index"] = pd.Categorical.from_codes(cell_idx, categories=cell_names)
    obs["gene_name"] = pd.Categorical.from_codes(gene_idx, categories=gene_names)
    obs["original_gene_name"] = obs["gene_name"]
    for col in obs_cols:
        if source_obs is not None and col in source_obs.columns:
            values = source_obs[col]
            if isinstance(values.dtype, pd.CategoricalDtype):
                codes = values.cat.codes.values[cell_idx]
                obs[col] = pd.Categorical.from_codes(codes, categories=values.cat.categories)
            else:
                obs[col] = values.values[cell_idx]
        elif col == "cell_id":
            obs["cell_id"] = obs["original_obs_index"]
    for key, value in _n_spots_intervals(n_spots).items():
        obs[key] = value
    obs.index = (obs["original_obs_index"].astype(str) + "_"
                 + obs["gene_name"].astype(str))
    return obs


def _finalise(X_real, X_ctrl, obs, source: str, report: Dict[str, Any]):
    real = ad.AnnData(X=X_real, obs=obs.copy())
    ctrl = ad.AnnData(X=X_ctrl, obs=obs.copy())
    vc = real.obs["gene_name"].value_counts()
    for a in (real, ctrl):
        a.obs["n_obs"] = a.obs["gene_name"].map(vc).astype(np.int32)
        a.uns["explode"] = dict(report, source=source)
    return real, ctrl


def _explode_direct(adata, real_layer, control_layer, obs_cols):
    """Small inputs: reuse the published function and check the pair aligns."""
    import sys
    from ..config import PUBLISHED_SRC
    src = str(PUBLISHED_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from adata_utils import reshape_and_filter_anndata_with_index_and_mapping as reshape

    real = reshape(adata, layer=real_layer)
    ctrl = reshape(adata, layer=control_layer)
    if not real.obs_names.equals(ctrl.obs_names):
        raise ValueError(
            "real and control explode to different row sets ({} vs {}); the "
            "control layer was not computed on the same spot-count window"
            .format(real.n_obs, ctrl.n_obs))
    vc = real.obs["gene_name"].value_counts()
    for a in (real, ctrl):
        a.obs["n_obs"] = a.obs["gene_name"].map(vc).astype(np.int32)
    report = {"path": "direct", "n_rows": int(real.n_obs),
              "real_not_control": 0, "control_not_real": 0}
    for a in (real, ctrl):
        a.uns["explode"] = report
    return real, ctrl


def _explode_chunked(path, real_layer, control_layer, obs_cols,
                     chunk_cells, verbose):
    import h5py

    source_obs, source_var = _read_metadata(path)
    cell_names = list(map(str, source_obs.index))
    gene_names = (list(map(str, source_var["gene_name"]))
                  if "gene_name" in source_var.columns
                  else list(map(str, source_var.index)))

    reals, ctrls, cells, genes, spots = [], [], [], [], []
    real_not_ctrl = ctrl_not_real = 0

    with h5py.File(path, "r") as f:
        dr = f["layers/{}".format(real_layer)]
        dc = f["layers/{}".format(control_layer)]
        if dr.shape != dc.shape:
            raise ValueError("layer shapes differ: {} vs {}".format(dr.shape, dc.shape))
        n_cells, n_genes, dim = dr.shape
        counts = f["X"]
        gene_ar = np.arange(n_genes, dtype=np.int32)

        for start in range(0, n_cells, chunk_cells):
            stop = min(start + chunk_cells, n_cells)
            R = np.asarray(dr[start:stop]).reshape(-1, dim)
            C = np.asarray(dc[start:stop]).reshape(-1, dim)
            keep = np.any(R != 0, axis=1)
            keep_c = np.any(C != 0, axis=1)
            real_not_ctrl += int((keep & ~keep_c).sum())
            ctrl_not_real += int((keep_c & ~keep).sum())
            if not keep.any():
                continue
            reals.append(R[keep].astype(np.float32))
            ctrls.append(C[keep].astype(np.float32))
            n = stop - start
            cells.append(np.repeat(np.arange(start, stop, dtype=np.int32), n_genes)[keep])
            genes.append(np.tile(gene_ar, n)[keep])
            spots.append(np.asarray(counts[start:stop]).reshape(-1)[keep])
            if verbose:
                print("  cells {:>7d}-{:<7d} kept {:>9d}".format(
                    start, stop, int(keep.sum())), flush=True)

    X_real = np.concatenate(reals); del reals
    X_ctrl = np.concatenate(ctrls); del ctrls
    cell_idx = np.concatenate(cells)
    gene_idx = np.concatenate(genes)
    n_spots = np.concatenate(spots)

    obs = _build_obs(cell_idx, gene_idx, n_spots, cell_names, gene_names,
                     source_obs, obs_cols)
    report = {"path": "chunked", "n_rows": int(X_real.shape[0]),
              "real_not_control": real_not_ctrl,
              "control_not_real": ctrl_not_real,
              "chunk_cells": chunk_cells,
              "n_cells": int(n_cells), "n_genes": int(n_genes)}
    return X_real, X_ctrl, obs, report


def _cache_key(path: Path, real_layer: str, control_layer: str) -> str:
    st = path.stat()
    raw = "|".join([path.name, str(st.st_size), str(int(st.st_mtime)),
                    real_layer, control_layer, code_version()])
    import hashlib
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def explode_matched(source, real_layer: str = "subcell_embedding",
                    control_layer: str = "random_subcell_embedding",
                    obs_cols: Sequence[str] = DEFAULT_OBS_COLS,
                    cache_dir: Optional[Path] = CACHE_DIR,
                    chunk_cells: int = DEFAULT_CHUNK_CELLS,
                    force: bool = False, verbose: bool = False
                    ) -> Tuple[ad.AnnData, ad.AnnData]:
    """Return ``(real, control)`` exploded to one row per measured (cell, gene).

    Both tables share an index, in the same order, as
    ``classify_all_genes`` requires. Small sources go through the published
    reshape; large ones are chunked over cells in bounded memory and cached.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        with __import__("h5py").File(path, "r") as f:
            shape = f["layers/{}".format(real_layer)].shape
        n_pairs = shape[0] * shape[1]
        small = n_pairs <= DIRECT_MAX_PAIRS
    else:
        path = None
        small = True

    if small:
        adata = ad.read_h5ad(path) if path is not None else source
        return _explode_direct(adata, real_layer, control_layer, obs_cols)

    key = _cache_key(path, real_layer, control_layer)
    cache = Path(cache_dir) / "explode_{}.npz".format(key) if cache_dir else None

    if cache is not None and cache.is_file() and not force:
        with np.load(cache, allow_pickle=False) as z:
            X_real, X_ctrl = z["X_real"], z["X_ctrl"]
            cell_idx, gene_idx, n_spots = z["cell_idx"], z["gene_idx"], z["n_spots"]
            cell_names = [str(v) for v in z["cell_names"]]
            gene_names = [str(v) for v in z["gene_names"]]
        report = json.loads(cache.with_suffix(".json").read_text())
        source_obs, _ = _read_metadata(path)
        obs = _build_obs(cell_idx, gene_idx, n_spots, cell_names, gene_names,
                         source_obs, obs_cols)
        report["path"] = "cache"
        return _finalise(X_real, X_ctrl, obs, str(path), report)

    X_real, X_ctrl, obs, report = _explode_chunked(
        path, real_layer, control_layer, obs_cols, chunk_cells, verbose)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X_real=X_real, X_ctrl=X_ctrl,
                 cell_idx=obs["original_cell_index"].values,
                 gene_idx=obs["original_gene_index"].values,
                 n_spots=obs["n_spots"].values,
                 cell_names=np.asarray(
                     obs["original_obs_index"].cat.categories, dtype=np.str_),
                 gene_names=np.asarray(
                     obs["gene_name"].cat.categories, dtype=np.str_))
        cache.with_suffix(".json").write_text(
            json.dumps(dict(report, source=str(path)), indent=2, default=str))

    return _finalise(X_real, X_ctrl, obs, str(path), report)
