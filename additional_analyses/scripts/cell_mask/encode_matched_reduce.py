#!/usr/bin/env python
# Provenance only: this script produced the real-data embeddings of the matched 300-epoch checkpoints, read by score_matched_detection.py. Needs the checkpoints, the real datasets and a GPU.
# Paths are those of the original run; it is not meant to run unchanged.
"""Encode ANY 2- or 3-channel CVAE over the `reduce/` real datasets, at full scale.

A parameterised copy of `encode_cvae3_reduce.py`, which is hardwired to the one
3-channel ablation checkpoint. This one takes `--ckpt`, `--out-dir` and `--method`
so a matched pair of checkpoints (e.g. the 2ch and 3ch 300-epoch runs) can be
encoded through an identical path and compared without the channel count or the
training run being confounded with the encoding. The original is left untouched
because the benchmark track depends on it.

    conda activate subcellspace_env      # or paths.CVAE_PY
    cd ~/SubCellSpace/additional_analyses/scripts/cell_mask
    nohup python -u encode_matched_reduce.py -device 1 \\
        > /path/to/work/cvae_encode.log 2>&1 &

Re-running resumes: every arm (and every APEX block) writes an atomic output plus
a manifest entry, and a unit whose output is on disk is skipped.

Why a script and not a cell in 00_prepare
-----------------------------------------
Same reason PointNet's adapter and the CNN baseline are scripts: the encode is
long, GPU-bound and worth resuming, and `00_prepare` shells out to it through
`run_script()` so the notebook stays a ledger of what was built.

`reduce/` is full-scale only -- no demos, no subsampling of a dataset. So every
stage here covers its dataset entirely, and `04_summary`'s scale gate rejects
anything that does not:

    artifact A   805,454 matched stems, both arms
    validation    44,706 rows, in `simval_obs.parquet`'s order
    ileum          3,358 (cell, gene) pairs, both arms
    APEX       8,449,113 pairs, both arms, streamed in cell blocks

Two things about the encode itself, both measured rather than assumed
--------------------------------------------------------------------
**The image source differs by dataset.** Simulated reads the original rendered
TIFFs; APEX and ileum render the stored coordinates. That is not a stylistic
choice: the simulated coordinate trees were recovered from the blurred images by
peak detection, so re-rendering them reproduces the shipped model's stored latent
only to cosine 0.9995 with a minimum of 0.82, while reading the TIFFs it trained
on gives median 1.000000, minimum 0.999996. APEX is the other way round -- its
stored integer coordinates *are* the source, and rendering them is exact.

**The mask channels are graded, not binary.** `obsm/dapi_mask` and
`obsm/cell_mask` are float64 with hundreds of distinct values, so they are read
with `H5adMasks(...).iter_masks(cells, raw=True)`. Thresholding them would feed
the model channels it never saw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]          # the SubCellSpace repository
sys.path.insert(0, "/path/to/reduce")               # helper modules of the original run, not part of this repository
sys.path.insert(0, str(REPO / "benchmark"))
sys.path.insert(0, str(REPO))                       # the pickles need module 'CVAE'

import paths                                        # noqa: E402  the one place a path is named

paths.add_src_to_path()

METHOD = "subcellspace_cellmask"   # overridden by --method
N_CHANNELS = 3                     # set from the checkpoint in main()

ARM_FIELDS = {"real": ("subcell_row", "subcell_col"),
              "ctrl": ("subcell_row_shuffled", "subcell_col_shuffled")}
_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    log("signal %s -- finishing the current unit, then stopping" % signum)


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


class Manifest:
    """Unit-level record. Resume is "done *and* the output still exists"."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {"units": {}, "runs": []}
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                log("manifest unreadable, starting fresh (outputs still reused)")
        self.data.setdefault("units", {})
        self.data.setdefault("runs", [])

    def done(self, key, output=None):
        rec = self.data["units"].get(key)
        if not rec or not rec.get("done"):
            return False
        if output is not None and not Path(output).exists():
            log("%s marked done but %s is gone -- redoing" % (key, output))
            return False
        return True

    def mark(self, key, info=None, output=None):
        self.data["units"][key] = {
            "done": True,
            "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "output": str(output) if output else None, "info": info or {}}
        self.flush()

    def note_run(self, argv):
        self.data["runs"].append({
            "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "argv": argv})
        self.flush()

    def flush(self):
        tmp = self.path.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True, default=str))
        tmp.replace(self.path)


def save_npz(path, **arrays):
    """Atomic savez. The temp name ends .npz because savez appends it otherwise."""
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def write_arm(out, X, keys, model, mode, dataset, index_kind, extra=None):
    from revbench import cvae_encode as CE
    return CE.write_encoded(out, X, keys, index_kind=index_kind,
                            method=METHOD, dataset=dataset,
                            model=model, model_path=paths.CVAE3_CKPT, mode=mode,
                            extra=extra)


# --- fidelity check, so a silent channel mistake cannot pass ------------------

def stage_check(a, mf, model):
    """The 2-channel model through this same path reproduces its stored latents.

    It checks the render + encode plumbing, not the ablation: if this drifts, the
    ablation's numbers are meaningless whatever they say.
    """
    key = "cvae3:check"
    if mf.done(key):
        log("fidelity check already recorded")
        return
    import h5py
    from anndata.experimental import read_elem
    from revbench import controls as C, cvae_encode as CE

    shipped = CE.load_model(paths.PUBLIC / "SubCellSpace", device=a.device_str)
    # Deliberately in WORK, not ARTIFACTS: this 64-cell control is plumbing for
    # the check and is not an input to any benchmark. Nothing it touches reaches
    # the results store, so it is not a partial-scale run.
    cs = C.extract_from_h5ad(paths.APEX, "cvae3_fidelity_check",
                             cache_dir=paths.WORK, max_cells=64, with_in_dapi=False)
    cnt = cs.points.groupby("obs_key").size()
    keys = list(cnt[(cnt > 10) & (cnt < 101)].index)[:300]
    cells = sorted({k.rsplit("__", 1)[0] for k in keys})
    with h5py.File(paths.APEX, "r") as f:
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
    ri = {str(c): i for i, c in enumerate(obs.index.astype(str))}
    ci = {str(g): i for i, g in enumerate(var["gene_name"].astype(str))}
    by_cell = {}
    for k in keys:
        by_cell.setdefault(k.rsplit("__", 1)[0], []).append(k)
    stored = {}
    with h5py.File(paths.APEX, "r") as f:
        lay = f["layers/random_subcell_embedding"]
        for c_ in sorted(cells, key=lambda x: ri[x]):
            blk = np.asarray(lay[ri[c_]])
            for k in by_cell[c_]:
                stored[k] = blk[ci[k.rsplit("__", 1)[1]]]
    keys = [k for k in keys if np.abs(stored[k]).sum() > 0]
    dapi, _ = CE.masks_by_cell(paths.APEX, cells)
    X, used = CE.encode_control_set(shipped, cs, dapi, mode="z", seed=101,
                                    obs_keys=keys, batch_size=a.batch_size)
    S = np.stack([stored[k] for k in used])
    cos = (X * S).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(S, axis=1))
    med = float(np.median(cos))
    log("fidelity: shipped re-encode vs stored latent, cosine median %.6f (n=%d)"
        % (med, len(used)))
    if med < 0.9999:
        raise SystemExit("render/encode path does not reproduce stored latents")
    mf.mark(key, {"cosine_median": med, "n": len(used)})


# --- stage 1: simulated artifact A -------------------------------------------

def stage_atrain(a, mf, model):
    from revbench import cvae_encode as CE
    from revbench.detection.classifier import load_training_pair

    outs = {arm: paths.CVAE3_DIR / ("atrain_%s.npz" % arm) for arm in ("real", "ctrl")}
    if all(mf.done("cvae3:atrain:%s" % arm, outs[arm]) for arm in outs):
        log("artifact A already built")
        return
    t0 = time.time()
    real, ctrl = load_training_pair()
    stems = [str(s) for s in ctrl.obs["control_key"].astype(str)]
    log("artifact A: %d matched stems (canonical order from load_training_pair)"
        % len(stems))
    trees = {"real": paths.SIM_TRAIN / "coordinate_parsed_with_coords",
             "ctrl": paths.SIM_TRAIN / "coordinate_parsed_with_coords_shuffled"}
    for arm, out in outs.items():
        key = "cvae3:atrain:%s" % arm
        if mf.done(key, out):
            continue
        if _STOP:
            return
        t = time.time()
        X, kept = CE.encode_tree_stems(model, trees[arm], stems, mode=a.mode,
                                       batch_size=a.batch_size, io_workers=a.io_workers,
                                       progress=True)
        if len(kept) != len(stems):
            raise SystemExit("%s: encoded %d of %d stems -- incomplete coverage would "
                             "shrink the shared panel" % (arm, len(kept), len(stems)))
        write_arm(out, X, kept, model, a.mode, "simulated", "image_stem",
                  extra={"stage": "atrain", "arm": arm, "n": len(kept)})
        mf.mark(key, {"n": len(kept), "min": round((time.time() - t) / 60, 1)}, out)
        log("  atrain %s: %d in %.1f min" % (arm, len(kept), (time.time() - t) / 60))
    log("artifact A done in %.1f min" % ((time.time() - t0) / 60))


# --- stage 2: simulated validation split -------------------------------------

def stage_simval(a, mf, model):
    import pandas as pd
    from revbench import cvae_encode as CE

    out = paths.CVAE3_DIR / "simval.npz"
    key = "cvae3:simval"
    if mf.done(key, out):
        log("simval already built")
        return
    # The canonical order is 00_prepare's, never this script's: silhouette draws a
    # seeded subsample, so a different row order would draw different rows.
    obs = pd.read_parquet(paths.ARTIFACTS / "simval_obs.parquet")
    stems = [str(s) for s in obs["obs_key"]]
    log("simval: %d rows in simval_obs.parquet order" % len(stems))
    root = paths.SIM_VAL
    gene = [str(root / "coordinate_parsed_zoomed_again" / "unbalanced"
                 / "cellids_left_out" / "test" / (s + ".tif")) for s in stems]
    dapi = [str(root / "rotated_dapi_zoomed_again" / (s + "_DAPI.tif")) for s in stems]
    cell = [str(root / "coordinate_parsed_with_coords" / "cell_mask_zoomed"
                 / (s + "_cell-mask.tif")) for s in stems]
    missing = [p for trio in zip(gene, dapi, cell) for p in trio if not Path(p).is_file()]
    if missing:
        raise SystemExit("%d simval channel files missing (first %s)"
                         % (len(missing), missing[0]))
    t = time.time()
    X = CE.encode_image_paths(model, gene, dapi, cell, mode=a.mode,
                              batch_size=a.batch_size, io_workers=a.io_workers,
                              progress=True)
    write_arm(out, X, stems, model, a.mode, "simulated", "image_stem",
              extra={"stage": "simval", "n": len(stems),
                     "order": "artifacts/simval_obs.parquet"})
    mf.mark(key, {"n": len(stems), "min": round((time.time() - t) / 60, 1)}, out)
    log("simval done: %d in %.1f min" % (len(stems), (time.time() - t) / 60))


# --- stage 3: ileum, and stage 4: APEX ---------------------------------------

def _pair_stage(a, mf, model, dataset, h5ad, block_cells=None):
    """Both arms of an exploded (cell, gene) dataset, rendered from coordinates."""
    from revbench import controls as C, cvae_encode as CE
    from revbench.detection import explode_matched

    outs = {arm: paths.CVAE3_DIR / ("%s_%s.npz" % (dataset, arm))
            for arm in ("real", "ctrl")}
    if all(mf.done("cvae3:%s:%s" % (dataset, arm), outs[arm]) for arm in outs):
        log("%s already built" % dataset)
        return
    real, ctrl = explode_matched(h5ad)
    akey = (real.obs["original_obs_index"].astype(str) + "__"
            + real.obs["gene_name"].astype(str)).values
    log("%s: %d exploded pairs (canonical order from the explode)" % (dataset, len(akey)))
    cells = sorted({k.rsplit("__", 1)[0] for k in akey})
    dapi, cellm = CE.masks_by_cell(h5ad, cells)
    for arm, out in outs.items():
        key = "cvae3:%s:%s" % (dataset, arm)
        if mf.done(key, out):
            continue
        if _STOP:
            return
        t = time.time()
        pts = C.extract_from_h5ad(h5ad, "reduce_%s_%s" % (dataset, arm),
                                  cache_dir=paths.ARTIFACTS / "controls",
                                  fields=ARM_FIELDS[arm], with_in_dapi=False)
        X, used = CE.encode_control_set(model, pts, dapi,
                                        cellm if N_CHANNELS == 3 else None, mode=a.mode,
                                        obs_keys=list(akey), batch_size=a.batch_size)
        if len(used) != len(akey):
            raise SystemExit("%s %s: encoded %d of %d pairs -- incomplete coverage"
                             % (dataset, arm, len(used), len(akey)))
        write_arm(out, X, used, model, a.mode, dataset, "cell_gene",
                  extra={"stage": dataset, "arm": arm, "n": len(used)})
        mf.mark(key, {"n": len(used), "min": round((time.time() - t) / 60, 1)}, out)
        log("  %s %s: %d in %.1f min" % (dataset, arm, len(used), (time.time() - t) / 60))


def stage_ileum(a, mf, model):
    _pair_stage(a, mf, model, "ileum", paths.ILEUM)


def _apex_panel():
    """The canonical (cell, gene) keys -- the exploded pair's, nothing more.

    `iter_point_blocks` yields every pair with a molecule on it; the benchmark
    scores only the pairs `explode_matched` retains, which is where the published
    pipeline's 10 < n < 101 window has already been applied. Measured on the first
    block, the difference is 4.6x: 1,256,979 pairs streamed against ~270,000 in
    the panel. Encoding the rest is hours of work nothing reads, and it is where
    the degenerate observations live -- a cell whose mask is uniform min-max
    scales to NaN, which is what stopped the first attempt.
    """
    from revbench.detection import explode_matched
    r, _c = explode_matched(paths.APEX)
    keys = set((r.obs["original_obs_index"].astype(str) + "__"
                + r.obs["gene_name"].astype(str)).values)
    del r, _c
    log("canonical APEX panel: %d (cell, gene) pairs" % len(keys))
    return keys


def stage_apex(a, mf, model):
    """APEX streamed in cell blocks, then reassembled -- 8.4M pairs per arm."""
    from revbench import cvae_encode as CE
    from revbench import ella_io as EIO
    from revbench.controls import RENDER_SHAPE

    panel = _apex_panel()
    for arm in ("real", "ctrl"):
        final = paths.CVAE3_DIR / ("apex_hek_%s.npz" % arm)
        if mf.done("cvae3:apex_hek:%s" % arm, final):
            log("apex %s already reassembled" % arm)
            continue
        digests = []
        for bi, cells, points in EIO.iter_point_blocks(
                paths.APEX, fields=ARM_FIELDS[arm], block_cells=a.block_cells,
                with_in_dapi=False):
            if points.empty:
                continue
            digests.append(EIO.block_fingerprint(points))
            out = paths.CVAE3_DIR / "apex" / arm / ("block_%04d.npz" % bi)
            key = "cvae3:apex_hek:%s:%04d" % (arm, bi)
            if mf.done(key, out):
                continue
            if _STOP:
                return
            t = time.time()
            points = points[points["obs_key"].isin(panel)]
            if points.empty:
                mf.mark(key, {"n": 0, "skipped": "no panel pair in this block"}, None)
                continue
            dapi, cellm = CE.masks_by_cell(paths.APEX, list(cells))
            keys, rowcols, dapis, cimgs = [], [], [], []
            n_flat = 0
            for k, g in points.groupby("obs_key", sort=False):
                c_ = str(k).rsplit("__", 1)[0]
                d, cm = dapi.get(c_), cellm.get(c_)
                if d is None or (N_CHANNELS == 3 and cm is None):
                    continue
                # A uniform channel min-max scales to NaN, so it cannot be encoded
                # at all. Skipping is the only honest option; the count is logged
                # and the coverage check at the end is what decides whether it
                # mattered.
                if d.min() == d.max() or (N_CHANNELS == 3 and cm.min() == cm.max()):
                    n_flat += 1
                    continue
                keys.append(str(k))
                rowcols.append((g["row"].values, g["col"].values))
                dapis.append(np.asarray(d, dtype=np.float32).reshape(RENDER_SHAPE))
                if N_CHANNELS == 3:
                    cimgs.append(np.asarray(cm, dtype=np.float32).reshape(RENDER_SHAPE))
            if n_flat:
                log("    %d pairs skipped: a uniform DAPI or cell mask" % n_flat)
            if not keys:
                mf.mark(key, {"n": 0, "skipped": "no encodable pair"}, None)
                continue
            X = CE.encode_images(model, dapi_imgs=dapis,
                                 cell_imgs=(cimgs if N_CHANNELS == 3 else None),
                                 gene_points=rowcols, mode=a.mode, seed=a.seed,
                                 batch_size=a.batch_size)
            save_npz(out, X=X.astype(np.float32),
                     obs_index=np.asarray(keys, dtype=np.str_))
            mf.mark(key, {"n": len(keys), "n_flat_skipped": n_flat,
                          "s": round(time.time() - t, 1)}, out)
            log("  apex %s block %04d: %6d pairs in %.0fs" % (arm, bi, len(keys),
                                                              time.time() - t))
        if _STOP:
            return
        # reassemble
        Xs, keys = [], []
        for p in sorted((paths.CVAE3_DIR / "apex" / arm).glob("block_*.npz")):
            with np.load(p, allow_pickle=False) as z:
                Xs.append(np.asarray(z["X"], dtype=np.float32))
                keys.extend(str(k) for k in z["obs_index"])
        X = np.vstack(Xs)
        write_arm(final, X, keys, model, a.mode, "apex_hek", "cell_gene",
                  extra={"stage": "apex_hek", "arm": arm, "n": len(keys),
                         "block_digest": hashlib.sha1("".join(digests).encode()).hexdigest(),
                         "fingerprint_scheme": "blockwise_sha1_v1"})
        missing = len(panel - set(keys))
        log("apex %s: %d pairs from %d blocks | %d of the panel's %d not covered"
            % (arm, len(keys), len(Xs), missing, len(panel)))
        if missing:
            log("    (those pairs cannot be encoded -- a uniform mask channel. The "
                "ablation's panel is intersected across the compared set, so this "
                "shrinks its own comparison and nothing else.)")
        mf.mark("cvae3:apex_hek:%s" % arm,
                {"n": len(keys), "blocks": len(Xs), "panel": len(panel),
                 "not_covered": missing}, final)


STAGES = {"check": stage_check, "atrain": stage_atrain, "simval": stage_simval,
          "ileum": stage_ileum, "apex": stage_apex}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-stages", default="check,atrain,simval,ileum,apex")
    ap.add_argument("-device", type=int, default=1, help="GPU index; 0 is often busy")
    ap.add_argument("-batch-size", type=int, default=2048)
    ap.add_argument("-block-cells", type=int, default=5000)
    ap.add_argument("-io-workers", type=int, default=24)
    ap.add_argument("-mode", default="mu", choices=("mu", "z"))
    ap.add_argument("-seed", type=int, default=101)
    ap.add_argument("--ckpt", default=None, help="checkpoint; defaults to paths.CVAE3_CKPT")
    ap.add_argument("--out-dir", default=None, help="output dir; defaults to paths.CVAE3_DIR")
    ap.add_argument("--method", default="subcellspace_cellmask",
                    help="method name stamped into the dumps")
    ap.add_argument("--manifest", default=None, help="manifest json; defaults per method")
    a = ap.parse_args(argv)
    a.device_str = "cuda:%d" % a.device

    paths.ensure()
    # Rebind the module-level targets so every downstream use follows, rather than
    # threading new arguments through a dozen call sites.
    if a.ckpt:
        paths.CVAE3_CKPT = Path(a.ckpt)
    if a.out_dir:
        paths.CVAE3_DIR = Path(a.out_dir)
    global METHOD, N_CHANNELS
    METHOD = a.method
    paths.CVAE3_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    from revbench import cvae_encode as CE
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(a.device)
    model = CE.load_model(paths.CVAE3_CKPT, device=a.device_str)
    info = CE.model_summary(model, paths.CVAE3_CKPT)
    if info["n_channels"] not in (2, 3):
        raise SystemExit("expected a 2- or 3-channel checkpoint, got %d" % info["n_channels"])
    N_CHANNELS = info["n_channels"]
    log("=" * 78)
    log("matched encode | method %s" % METHOD)
    log("channels %d | ld %d | mode %s | device %s"
        % (info["n_channels"], info["latent_dim"], a.mode, a.device_str))
    log("checkpoint: %s" % paths.CVAE3_CKPT.name)

    mf = Manifest(Path(a.manifest) if a.manifest
                  else paths.WORK / ("matched_manifest_%s.json" % METHOD))
    mf.note_run(argv)
    for name in [s for s in a.stages.split(",") if s]:
        if _STOP:
            log("stopped; rerun the same command to resume")
            return
        log("--- stage %s ---" % name)
        STAGES[name](a, mf, model)
    log("finished. output: %s" % paths.CVAE3_DIR)


if __name__ == "__main__":
    main()
