"""Simulated cell + nucleus polygons, in PROCESSES not threads.

`build_geometry_with_nucleus` is pure Python / numpy / skimage, so a thread pool
gives it about 1.2 cores no matter how many threads (measured: 122% CPU on 10
threads while building simval). Artifact A is 805,454 stems x 2 contours, which
is ~1 hour that way and ~2 minutes across 32 processes.

Built ONCE per stem set, not once per arm: the cell-mask and DAPI TIFs are
byte-identical between `coordinate_parsed_with_coords` and
`..._shuffled` (verified by md5 on a sample), because shuffling relocates
transcripts inside an unchanged cell.
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


def one_block(bi, stems, tree, geom_dir):
    """Worker: polygons for one block of stems. Runs in its own process."""
    for p in (SRC, RED):
        if p not in sys.path:
            sys.path.insert(0, p)
    from revbench import ella_io as EIO
    geom_dir = Path(geom_dir)
    cp_path = geom_dir / ("block_%04d_cell.npz" % bi)
    np_path = geom_dir / ("block_%04d_nuc.npz" % bi)
    rp_path = geom_dir / ("report_%04d.parquet" % bi)
    if cp_path.exists() and np_path.exists() and rp_path.exists():
        return bi, -1, -1
    masks = EIO.TreeMasks(tree)
    cp, npl, rep = EIO.build_geometry_with_nucleus(
        masks, stems, iou_sample=min(25, len(stems)), seed=101,
        dapi_threshold=0.0, clip_to_cell=True)
    save_polys(cp, cp_path)
    save_polys(npl, np_path)
    rep["block"] = bi
    rep.to_parquet(rp_path, index=False)
    return bi, len(cp), len(npl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["atrain", "simval"])
    ap.add_argument("--block-stems", type=int, default=5000)
    ap.add_argument("--n-jobs", type=int, default=32)
    A = ap.parse_args()

    from revbench import controls as C
    if A.which == "atrain":
        # the real tree; its masks are byte-identical to the shuffled tree's
        tree = paths.SIM_TRAIN / "coordinate_parsed_with_coords"
        cs = C.load_control("reduce_atrain_real",
                            cache_dir=paths.ARTIFACTS / "controls")
        stems_all = sorted(set(cs.points["obs_key"].astype(str)))
    else:
        tree = paths.SIM_VAL / "coordinate_parsed_with_coords"
        stems_all = sorted(set(
            pd.read_parquet(paths.ARTIFACTS / "simval_obs.parquet")["obs_key"]
            .astype(str)))

    geom_dir = paths.ARTIFACTS / "bento" / "simulated" / "geom" / A.which
    geom_dir.mkdir(parents=True, exist_ok=True)
    blocks = [stems_all[i:i + A.block_stems]
              for i in range(0, len(stems_all), A.block_stems)]
    print("%s: %d stems -> %d blocks, %d processes"
          % (A.which, len(stems_all), len(blocks), A.n_jobs), flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    out = Parallel(n_jobs=A.n_jobs, verbose=5)(
        delayed(one_block)(bi, s, str(tree), str(geom_dir))
        for bi, s in enumerate(blocks))
    built = [(b, c, n) for b, c, n in out if c >= 0]
    print("geometry: %d blocks (%d newly built) in %.1f min"
          % (len(out), len(built), (time.time() - t0) / 60), flush=True)
    reps = sorted(geom_dir.glob("report_*.parquet"))
    R = pd.concat([pd.read_parquet(p) for p in reps], ignore_index=True)
    R.to_parquet(geom_dir / "geometry_report.parquet", index=False)
    print("cells %d | cell poly %d | nucleus poly %d | clipped %d (%.1f%%)"
          % (len(R), int(R.ok.sum()), int(R.nuc_ok.sum()),
             int((R.nuc_clipped_px > 0).sum()),
             100.0 * (R.nuc_clipped_px > 0).mean()), flush=True)
    print("cell IoU median %.6f | nucleus IoU median %.6f"
          % (R.iou.median(), R[R.nuc_ok].nuc_iou.median()), flush=True)
    print("=== SIM_GEOM_%s_DONE ===" % A.which.upper(), flush=True)


if __name__ == "__main__":
    main()
