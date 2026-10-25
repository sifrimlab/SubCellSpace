"""Build + score one real dataset with Bento, block by block, resumable.

Prototype for the `00_prepare` stage, and shaped like that notebook's
`stage_pointnet_apex`: a single producer streams point blocks out of the h5ad
while a thread pool of subprocesses consumes them, with a semaphore for
backpressure because the pool queue is unbounded and every queued block holds
its points in memory.

Reading and scoring are partitioned separately
----------------------------------------------
`iter_point_blocks` costs far more per cell at small block sizes: it runs a
per-cell Python loop that decodes every gene string (~2,100 per APEX cell, 321M
records in the file), plus a fixed per-`_read_chunk` cost that small blocks
multiply. Measured: ~1 min per 800-cell block, i.e. 6+ hours to read both arms,
against ~1 min per 5,000-cell block in `00_prepare`'s proven configuration.

So the file is read in 5,000-cell blocks and each is *split* into 1,000-cell
scoring units, keyed by `global_cell_index // sub_cells`. That keying is
independent of the read size, so changing `--read-block` never invalidates a
shard, while `--sub-cells` is what a resume must keep fixed.

Two things this shape exists to avoid
-------------------------------------
* **A quadratic read.** `EIO.iter_point_blocks` streams the whole file per call,
  so calling it once per output block would re-read every point N times. The
  partition is therefore *defined* by that iterator and consumed in one pass
  per arm.
* **Bento's own `num_workers`.** Measured 35.9 pairs/s at 1 worker, 49.4 at 4,
  48.9 at 16, 49.4 at 32 -- `analyze_points` makes one dask partition per cell,
  so extra workers have nothing to take. Each block is one subprocess with one
  worker, and many run at once.

Polygons are arm-independent: the shuffled control relocates transcripts inside
the same cell, so the cell and nucleus masks are identical across arms. They are
built on whichever arm reaches the block first and reused by the other.
"""
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

# resolved from this file, not hardcoded, so the script moves with the repo
_REVISION = Path(__file__).resolve().parent.parent
for _p in (_REVISION / "src", _REVISION / "notebooks" / "reduce"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import h5py
from anndata.experimental import read_elem
import paths
from revbench import ella_io as EIO
from revbench.detection import explode_matched
from revbench.geometry import RENDER_SHAPE

ARM_FIELDS = {"real": ("subcell_row", "subcell_col"),
              "ctrl": ("subcell_row_shuffled", "subcell_col_shuffled")}
ADAPTER = Path(paths.REVISION) / "src" / "revbench" / "adapters" / "bento_rnaforest.py"
# A Jupyter kernel exports MPLBACKEND=module://matplotlib_inline.backend_inline,
# which subprocesses inherit and bento_env cannot import. setdefault does not fix
# it -- the variable is already set, and wrongly. The thread pins stop 40 child
# processes each spawning 128 BLAS threads.
CHILD_ENV = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "NUMEXPR_NUM_THREADS": "1"}

ap = argparse.ArgumentParser()
ap.add_argument("dataset", choices=["ileum", "apex_hek"])
ap.add_argument("--read-block", type=int, default=5000,
                help="cells per iter_point_blocks yield; 5000 is the proven size")
ap.add_argument("--sub-cells", type=int, default=1000,
                help="cells per scoring unit; must divide --read-block")
ap.add_argument("--concurrency", type=int, default=40)
ap.add_argument("--limit-blocks", type=int, default=None)
A = ap.parse_args()

H5 = {"ileum": paths.ILEUM, "apex_hek": paths.APEX}[A.dataset]
OUT = paths.ARTIFACTS / "bento" / A.dataset
GEOM, SHARD = OUT / "geom", OUT / "shards"
PTS = paths.WORK / "bento_pc" / A.dataset
for d in (OUT, GEOM, SHARD, PTS):
    d.mkdir(parents=True, exist_ok=True)
MF_PATH = paths.WORK / ("bento_manifest_%s.json" % A.dataset)
_lock = threading.Lock()
MF = json.loads(MF_PATH.read_text()) if MF_PATH.is_file() else {"units": {}}
MF.setdefault("units", {})
_geom_lock = threading.Lock()


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def done(key, out=None):
    rec = MF["units"].get(key)
    if not rec or not rec.get("done"):
        return False
    if out is not None and not Path(out).exists():
        log("manifest says %s done but %s missing -- redoing" % (key, out))
        return False
    return True


def mark(key, info=None, out=None):
    with _lock:
        MF["units"][key] = {"done": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "output": str(out) if out else None, "info": info or {}}
        tmp = MF_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(MF, indent=1, sort_keys=True, default=str))
        tmp.replace(MF_PATH)


def save_polys(polys, path):
    """Atomic; the temp name ends in .npz because savez_compressed appends it."""
    cells = sorted(polys)
    verts = (np.concatenate([polys[c] for c in cells]).astype(np.float32)
             if cells else np.zeros((0, 2), np.float32))
    offsets = np.cumsum([0] + [len(polys[c]) for c in cells]).astype(np.int64)
    tmp = Path(str(path)[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, cells=np.asarray(cells, dtype=np.str_),
                        offsets=offsets, verts=verts)
    tmp.replace(path)


# ------------------------------------------------------------- the pair set
t0 = time.time()
real, ctrl = explode_matched(H5)
pair_key = (real.obs["original_obs_index"].astype(str) + "__"
            + real.obs["gene_name"].astype(str)).to_numpy()
N_PAIRS = len(pair_key)
n_usable = int(real.obs["original_obs_index"].nunique())
log("exploded: %d pairs / %d usable cells in %.1f s" % (N_PAIRS, n_usable, time.time() - t0))
pd.DataFrame({"obs_key": pair_key}).to_parquet(OUT / "pair_keys.parquet", index=False)
WANT = set(pair_key)
del real, ctrl

with h5py.File(H5, "r") as f:
    obs_names = list(read_elem(f["obs"]).index.astype(str))
OBS_POS = {c: i for i, c in enumerate(obs_names)}
MASKS = EIO.H5adMasks(H5)


def attach_in_dapi(p, cells_arr, ci=None):
    """in_dapi from ONE chunk-aligned slice, verified bit-identical to
    controls._dapi_lookup on 578,320 points (which is 283x slower)."""
    if ci is None:
        ci = pd.Series(cells_arr).map(OBS_POS).to_numpy().astype(np.int64)
    lo, hi = int(ci.min()), int(ci.max()) + 1
    with h5py.File(H5, "r") as f:
        blk = np.asarray(f["obsm/dapi_mask"][lo:hi]) > 0
    r = np.clip(p["row"].to_numpy(dtype=int), 0, RENDER_SHAPE[0] - 1)
    c = np.clip(p["col"].to_numpy(dtype=int), 0, RENDER_SHAPE[1] - 1)
    flat = (ci - lo) * (RENDER_SHAPE[0] * RENDER_SHAPE[1]) + r * RENDER_SHAPE[1] + c
    p = p.copy()
    p["in_dapi"] = blk.reshape(-1)[flat]
    p["nucleus"] = p["in_dapi"]
    del blk
    return p


def ensure_geometry(bi, cells):
    """Cell + nucleus polygons for one block. Arm-independent, so built once."""
    cp_path = GEOM / ("block_%04d_cell.npz" % bi)
    np_path = GEOM / ("block_%04d_nuc.npz" % bi)
    with _geom_lock:
        if cp_path.exists() and np_path.exists():
            return cp_path, np_path, None
        cp, npl, rep = EIO.build_geometry_with_nucleus(
            MASKS, cells, iou_sample=min(50, len(cells)), seed=101,
            dapi_threshold=0.0, clip_to_cell=True)
        save_polys(cp, cp_path)
        save_polys(npl, np_path)
        rep["block"] = bi
        rep.to_parquet(GEOM / ("report_%04d.parquet" % bi), index=False)
        return cp_path, np_path, rep


def run_block(arm, bi, points):
    """Stage one block's points and score it. Runs in a worker thread."""
    key = "bento:feat:%s:%s:%04d" % (A.dataset, arm, bi)
    out = SHARD / arm / ("block_%04d.npz" % bi)
    lab = SHARD / arm / ("block_%04d_labels.parquet" % bi)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cells_arr, _ = EIO.split_keys(points["obs_key"].to_numpy(), "cell_gene")
    cells = sorted(set(str(c) for c in cells_arr))
    cp_path, np_path, _ = ensure_geometry(bi, cells)
    if done(key, out):
        return bi, arm, 0, 0.0, "reused"
    pq = PTS / arm / ("block_%04d.parquet" % bi)
    pq.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(pq) + ".tmp")
    points.to_parquet(tmp, index=False)
    tmp.replace(pq)
    try:
        cmd = [paths.BENTO_ENV, str(ADAPTER),
               "--points", str(pq), "--cell-poly", str(cp_path),
               "--nucleus-poly", str(np_path), "--out", str(out),
               "--labels", str(lab), "--dataset", A.dataset,
               "--index-kind", "cell_gene", "--num-workers", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=CHILD_ENV)
        if r.returncode != 0:
            raise RuntimeError("bento %s block %d exit %d\n%s"
                               % (arm, bi, r.returncode, r.stderr[-2000:]))
    finally:
        pq.unlink(missing_ok=True)
    with np.load(out, allow_pickle=False) as z:
        n = int(z["X"].shape[0])
    mark(key, {"n_pairs": n, "s": round(time.time() - t0, 1)}, out)
    return bi, arm, n, time.time() - t0, "built"


# --------------------------------------------------- one producer pass per arm
t_all = time.time()
n_units = n_pairs_done = 0
failures = []
gate = threading.Semaphore(A.concurrency + 8)   # queued blocks hold their points in RAM
with ThreadPoolExecutor(max_workers=A.concurrency) as ex:
    futs = []
    for arm, fields in ARM_FIELDS.items():
        n_read = n_sub = 0
        for bi, blk_cells, points in EIO.iter_point_blocks(
                H5, fields=fields, block_cells=A.read_block, with_in_dapi=False):
            if A.limit_blocks and n_read >= A.limit_blocks:
                break
            n_read += 1
            if points.empty:
                continue
            keep = pd.Series(points["obs_key"].to_numpy()).isin(WANT).to_numpy()
            if not keep.any():
                continue
            p = points.loc[keep].copy()
            cells_arr, _ = EIO.split_keys(p["obs_key"].to_numpy(), "cell_gene")
            ci = pd.Series(cells_arr).map(OBS_POS).to_numpy().astype(np.int64)
            p = attach_in_dapi(p, cells_arr, ci)
            # scoring unit id from the GLOBAL cell index, so it does not depend
            # on how the file happened to be read
            p["_sub"] = ci // A.sub_cells
            for sub, chunk in p.groupby("_sub", sort=True):
                n_sub += 1
                gate.acquire()

                def job(arm=arm, sub=int(sub), chunk=chunk.drop(columns=["_sub"])):
                    try:
                        return run_block(arm, sub, chunk)
                    finally:
                        gate.release()
                futs.append(ex.submit(job))
            del p
        log("%s: %d read blocks -> %d scoring units submitted" % (arm, n_read, n_sub))
    for f in futs:
        try:
            bi, arm, n, dt, how = f.result()
        except Exception as exc:
            failures.append(str(exc)[:500])
            continue
        n_units += 1
        n_pairs_done += n
        if n_units % 20 == 0 or n_units == len(futs):
            el = time.time() - t_all
            log("  %d/%d units | %d pairs | %.0f pairs/s | %.1f min"
                % (n_units, len(futs), n_pairs_done, n_pairs_done / max(el, 1e-9), el / 60))
if failures:
    for m in failures[:3]:
        log("FAILURE: %s" % m)
    raise RuntimeError("%d block(s) failed -- completed blocks are kept, re-run resumes"
                       % len(failures))
log("features done in %.1f min" % ((time.time() - t_all) / 60))

reps = sorted(GEOM.glob("report_*.parquet"))
if reps:
    R = pd.concat([pd.read_parquet(p) for p in reps], ignore_index=True)
    R.to_parquet(OUT / "geometry_report.parquet", index=False)
    log("geometry: %d cells | cell IoU median %.6f | nucleus IoU median %.6f | "
        "clipped %d (%.1f%%) | nucleus missing %d"
        % (len(R), R.iou.median(), R[R.nuc_ok].nuc_iou.median(),
           int((R.nuc_clipped_px > 0).sum()), 100.0 * (R.nuc_clipped_px > 0).mean(),
           int((~R.nuc_ok).sum())))

if A.limit_blocks:
    log("--limit-blocks set: skipping reassembly (partial run)")
    raise SystemExit(0)

# ------------------------------------------------------------- reassemble
for arm in ARM_FIELDS:
    out = OUT / ("%s_%s.npz" % (A.dataset, arm))
    key = "bento:concat:%s:%s" % (A.dataset, arm)
    if done(key, out):
        log("skip concat %s" % arm)
        continue
    Xs, keys = [], []
    for p in sorted((SHARD / arm).glob("block_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            Xs.append(np.asarray(z["X"], dtype=np.float32))
            keys.extend(str(k) for k in z["obs_index"])
    X = np.vstack(Xs)
    dup = len(keys) - len(set(keys))
    if dup:
        raise ValueError("%d duplicate keys across blocks in %s" % (dup, arm))
    labs = pd.concat([pd.read_parquet(p) for p in
                      sorted((SHARD / arm).glob("block_*_labels.parquet"))],
                     ignore_index=True)
    tmp = Path(str(out)[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, X=X, obs_index=np.asarray(keys, dtype=np.str_))
    tmp.replace(out)
    labs.to_parquet(OUT / ("labels_%s.parquet" % arm), index=False)
    nan_rows = int(np.isnan(X).any(axis=1).sum())
    allzero = int((np.nan_to_num(X) == 0).all(axis=1).sum())
    Path(str(out)[:-4] + ".meta.json").write_text(json.dumps({
        "schema": 1, "method": "bento_rnaforest", "dataset": A.dataset,
        "index_kind": "cell_gene", "n": int(X.shape[0]),
        "latent_dim": int(X.shape[1]), "arm": arm,
        "n_rows_with_nan": nan_rows, "n_all_zero_rows": allzero,
        "n_pairs_expected": N_PAIRS, "sub_cells": A.sub_cells, "read_block": A.read_block,
        "frame": "zoomed_100x100_gridded", "clip_to_cell": True,
        "feature_order_source": "bento.tools._lp.pattern_features",
    }, indent=2))
    mark(key, {"n": len(keys), "d": int(X.shape[1]), "n_nan_rows": nan_rows}, out)
    log("assembled %s: %s | NaN rows %d (%.3f%%) | all-zero %d | coverage %d/%d %s"
        % (arm, X.shape, nan_rows, 100.0 * nan_rows / len(keys), allzero,
           len(keys), N_PAIRS, "OK" if len(keys) == N_PAIRS else "!! SHORT"))
log("=== BENTO_%s_DONE ===" % A.dataset.upper())
