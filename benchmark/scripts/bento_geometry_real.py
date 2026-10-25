"""Cell + nucleus polygons for a real dataset, in PROCESSES, keyed by sub-block.

Same reason as `build_sim_geometry.py`: `build_geometry_with_nucleus` is
GIL-bound, so doing it inside the scoring driver's thread pool serialises it
behind one lock (~21 min for APEX) and starves the pool early on. Across 32
processes the whole 150,151-cell set takes a couple of minutes.

Units are keyed `global_cell_index // sub_cells`, which is exactly how the
scoring driver names them, so the two agree without either knowing how the point
file happened to be read.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

# resolved from this file, not hardcoded, so the script moves with the repo.
# Kept as module-level strings because the joblib workers re-insert them after
# the spawn -- a Path object would work too, but the workers want str.
_REVISION = Path(__file__).resolve().parent.parent
SRC = str(_REVISION / "src")
RED = str(_REVISION / "notebooks" / "reduce")
for p in (SRC, RED):
    if p not in sys.path:
        sys.path.insert(0, p)
import paths


def save_polys(polys, path):
    cells = sorted(polys)
    verts = (np.concatenate([polys[c] for c in cells]).astype(np.float32)
             if cells else np.zeros((0, 2), np.float32))
    offsets = np.cumsum([0] + [len(polys[c]) for c in cells]).astype(np.int64)
    tmp = Path(str(path)[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, cells=np.asarray(cells, dtype=np.str_),
                        offsets=offsets, verts=verts)
    tmp.replace(path)


def one_unit(sub, cells, h5, geom_dir):
    for p in (SRC, RED):
        if p not in sys.path:
            sys.path.insert(0, p)
    from revbench import ella_io as EIO
    geom_dir = Path(geom_dir)
    cp_path = geom_dir / ("block_%04d_cell.npz" % sub)
    np_path = geom_dir / ("block_%04d_nuc.npz" % sub)
    rp_path = geom_dir / ("report_%04d.parquet" % sub)
    if cp_path.exists() and np_path.exists() and rp_path.exists():
        return sub, -1, -1
    cp, npl, rep = EIO.build_geometry_with_nucleus(
        EIO.H5adMasks(h5), cells, iou_sample=min(25, len(cells)), seed=101,
        dapi_threshold=0.0, clip_to_cell=True)
    save_polys(cp, cp_path)
    save_polys(npl, np_path)
    rep["block"] = sub
    rep.to_parquet(rp_path, index=False)
    return sub, len(cp), len(npl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ileum", "apex_hek"])
    ap.add_argument("--sub-cells", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=32)
    A = ap.parse_args()

    import h5py
    from anndata.experimental import read_elem
    from revbench.detection import explode_matched

    H5 = {"ileum": paths.ILEUM, "apex_hek": paths.APEX}[A.dataset]
    real, _ = explode_matched(H5)
    usable = set(real.obs["original_obs_index"].astype(str))
    del real
    with h5py.File(H5, "r") as f:
        obs_names = [str(c) for c in read_elem(f["obs"]).index.astype(str)]

    # group the USABLE cells by the same key the scoring driver derives
    units = {}
    for i, c in enumerate(obs_names):
        if c in usable:
            units.setdefault(i // A.sub_cells, []).append(c)
    geom_dir = paths.ARTIFACTS / "bento" / A.dataset / "geom"
    geom_dir.mkdir(parents=True, exist_ok=True)
    print("%s: %d usable cells -> %d units of <=%d, %d processes"
          % (A.dataset, len(usable), len(units), A.sub_cells, A.n_jobs), flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    out = Parallel(n_jobs=A.n_jobs, verbose=5)(
        delayed(one_unit)(sub, cells, str(H5), str(geom_dir))
        for sub, cells in sorted(units.items()))
    built = [o for o in out if o[1] >= 0]
    print("geometry: %d units (%d newly built) in %.1f min"
          % (len(out), len(built), (time.time() - t0) / 60), flush=True)
    R = pd.concat([pd.read_parquet(p) for p in sorted(geom_dir.glob("report_*.parquet"))],
                  ignore_index=True)
    R.to_parquet(paths.ARTIFACTS / "bento" / A.dataset / "geometry_report.parquet",
                 index=False)
    print("cells %d | cell poly %d | nucleus poly %d | clipped %d (%.1f%%) | nuc missing %d"
          % (len(R), int(R.ok.sum()), int(R.nuc_ok.sum()),
             int((R.nuc_clipped_px > 0).sum()), 100.0 * (R.nuc_clipped_px > 0).mean(),
             int((~R.nuc_ok).sum())), flush=True)
    print("cell IoU median %.6f | nucleus IoU median %.6f"
          % (R.iou.median(), R[R.nuc_ok].nuc_iou.median()), flush=True)
    print("=== REAL_GEOM_%s_DONE ===" % A.dataset.upper(), flush=True)


if __name__ == "__main__":
    main()
