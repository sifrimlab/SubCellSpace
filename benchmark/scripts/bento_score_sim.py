"""Score the simulated arms with Bento, using geometry built by build_sim_geometry.py.

Split out from the geometry stage because the two parallelise differently:
geometry is GIL-bound Python/skimage and needs *processes*; scoring is a pool of
subprocesses and needs *threads* to shepherd them.

Two things this gets right that the first draft did not:

* **Grouping.** `pts.groupby("obs_key")` on artifact A builds 805,454 tiny
  DataFrames. A stem -> block map plus one `groupby(block)` gives 162 groups
  instead, off the same 20.1M-row frame.
* **Shared geometry.** Artifact A's real and shuffled trees hold byte-identical
  cell-mask and DAPI TIFs (md5-checked), so both arms read one polygon set and
  the block partition must match the one geometry was built with -- same
  `sorted(set(stems))`, same `--block-stems`.
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
import paths
from revbench import controls as C

ADAPTER = Path(paths.REVISION) / "src" / "revbench" / "adapters" / "bento_rnaforest.py"
CHILD_ENV = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "NUMEXPR_NUM_THREADS": "1"}
GENE = "sim"

#: arm -> (ControlSet cache name, geometry set). Both artifact-A arms share the
#: "atrain" geometry; only their point tables differ.
ARMS = {"atrain_real": ("reduce_atrain_real", "atrain"),
        "atrain_ctrl": ("reduce_atrain_ctrl", "atrain"),
        "simval": ("reduce_simval", "simval")}

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="atrain_real,atrain_ctrl,simval")
ap.add_argument("--block-stems", type=int, default=5000)
ap.add_argument("--concurrency", type=int, default=24)
A = ap.parse_args()

OUT = paths.ARTIFACTS / "bento" / "simulated"
GEOM, SHARD = OUT / "geom", OUT / "shards"
PTS = paths.WORK / "bento_pc" / "simulated"
for d in (OUT, SHARD, PTS):
    d.mkdir(parents=True, exist_ok=True)
MF_PATH = paths.WORK / "bento_manifest_simulated.json"
_lock = threading.Lock()
MF = json.loads(MF_PATH.read_text()) if MF_PATH.is_file() else {"units": {}}
MF.setdefault("units", {})


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


def run_unit(arm, geom_set, bi, sub):
    key = "bento:sim:%s:%04d" % (arm, bi)
    out = SHARD / arm / ("block_%04d.npz" % bi)
    lab = SHARD / arm / ("block_%04d_labels.parquet" % bi)
    out.parent.mkdir(parents=True, exist_ok=True)
    if done(key, out):
        return bi, arm, 0, 0.0
    cp = GEOM / geom_set / ("block_%04d_cell.npz" % bi)
    npz = GEOM / geom_set / ("block_%04d_nuc.npz" % bi)
    for f in (cp, npz):
        if not f.exists():
            raise FileNotFoundError("missing geometry %s -- run build_sim_geometry.py" % f)
    t0 = time.time()
    pq = PTS / arm / ("block_%04d.parquet" % bi)
    pq.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(pq) + ".tmp")
    sub.to_parquet(tmp, index=False)
    tmp.replace(pq)
    try:
        cmd = [paths.BENTO_ENV, str(ADAPTER),
               "--points", str(pq), "--cell-poly", str(cp), "--nucleus-poly", str(npz),
               "--out", str(out), "--labels", str(lab), "--dataset", "simulated",
               "--index-kind", "image_stem", "--gene-for-stems", GENE,
               "--num-workers", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=CHILD_ENV)
        if r.returncode != 0:
            raise RuntimeError("bento %s block %d exit %d\n%s"
                               % (arm, bi, r.returncode, r.stderr[-2000:]))
    finally:
        pq.unlink(missing_ok=True)
    with np.load(out, allow_pickle=False) as z:
        n = int(z["X"].shape[0])
    mark(key, {"n": n, "s": round(time.time() - t0, 1)}, out)
    return bi, arm, n, time.time() - t0


arms = [a for a in A.arms.split(",") if a]
t_all = time.time()
n_units = n_rows = 0
failures = []
gate = threading.Semaphore(A.concurrency + 8)
with ThreadPoolExecutor(max_workers=A.concurrency) as ex:
    futs = []
    for arm in arms:
        cache_name, geom_set = ARMS[arm]
        cs = C.load_control(cache_name, cache_dir=paths.ARTIFACTS / "controls")
        pts = cs.points
        pts["obs_key"] = pts["obs_key"].astype(str)
        if "nucleus" not in pts.columns:
            pts["nucleus"] = pts["in_dapi"]
        stems_all = sorted(set(pts["obs_key"]))
        # the partition geometry was built with: same order, same block size
        blk_of = {s: i // A.block_stems for i, s in enumerate(stems_all)}
        pts = pts.assign(_blk=pts["obs_key"].map(blk_of).astype(int))
        n_blocks = int(pts["_blk"].max()) + 1
        log("%s: %d points / %d stems / %d blocks (geometry set %r)"
            % (arm, len(pts), len(stems_all), n_blocks, geom_set))
        for bi, sub in pts.groupby("_blk", sort=True):
            gate.acquire()

            def job(arm=arm, geom_set=geom_set, bi=int(bi),
                    sub=sub.drop(columns=["_blk"])):
                try:
                    return run_unit(arm, geom_set, bi, sub)
                finally:
                    gate.release()
            futs.append(ex.submit(job))
        del pts, cs
    for f in futs:
        try:
            bi, arm, n, dt = f.result()
        except Exception as exc:
            failures.append(str(exc)[:500])
            continue
        n_units += 1
        n_rows += n
        if n_units % 20 == 0 or n_units == len(futs):
            el = time.time() - t_all
            log("  %d/%d units | %d rows | %.0f rows/s | %.1f min"
                % (n_units, len(futs), n_rows, n_rows / max(el, 1e-9), el / 60))
if failures:
    for m in failures[:3]:
        log("FAILURE: %s" % m)
    raise RuntimeError("%d unit(s) failed -- completed blocks kept, re-run resumes"
                       % len(failures))
log("features done in %.1f min" % ((time.time() - t_all) / 60))

for arm in arms:
    out = OUT / ("%s.npz" % arm)
    key = "bento:sim:concat:%s" % arm
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
        raise ValueError("%d duplicate keys in %s" % (dup, arm))
    labs = pd.concat([pd.read_parquet(p) for p in
                      sorted((SHARD / arm).glob("block_*_labels.parquet"))],
                     ignore_index=True)
    tmp = Path(str(out)[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, X=X, obs_index=np.asarray(keys, dtype=np.str_))
    tmp.replace(out)
    labs.to_parquet(OUT / ("labels_%s.parquet" % arm), index=False)
    nan_rows = int(np.isnan(X).any(axis=1).sum())
    Path(str(out)[:-4] + ".meta.json").write_text(json.dumps({
        "schema": 1, "method": "bento_rnaforest", "dataset": "simulated",
        "index_kind": "image_stem", "n": int(X.shape[0]),
        "latent_dim": int(X.shape[1]), "arm": arm, "n_rows_with_nan": nan_rows,
        "n_all_zero_rows": int((np.nan_to_num(X) == 0).all(axis=1).sum()),
        "block_stems": A.block_stems, "gene_for_stems": GENE,
        "frame": "zoomed_100x100_gridded", "clip_to_cell": True,
        # carried up from the per-block adapter meta so the assembled dump is
        # self-describing: `_lp.py` warns "Do not change order of features!"
        "feature_order": json.loads(
            (list(sorted((SHARD / arm).glob("block_*.npz")))[0]
             .with_suffix("").with_suffix(".meta.json")).read_text()
        ).get("feature_order") if list((SHARD / arm).glob("block_*.meta.json")) else None,
    }, indent=2))
    mark(key, {"n": len(keys), "d": int(X.shape[1])}, out)
    log("assembled %s: %s | NaN rows %d (%.3f%%)"
        % (arm, X.shape, nan_rows, 100.0 * nan_rows / len(keys)))
log("=== BENTO_SIM_SCORED ===")
