"""Embed the simulated arms with the trained CNN baseline, and record its own calls.

Emits, per arm, exactly what the registry needs:

* ``cnn/<arm>.npz``            the 15-d penultimate embedding (+ ``.meta.json``)
* ``cnn/labels_<arm>.parquet``  the two heads' predictions and probabilities

Inputs are the *same* two-channel renders the CNN was trained on -- the blurred
gene TIF and the graded DAPI TIF, each per-image min-max normalised -- read from
each arm's own tree, so training and inference share one path and no
render-versus-TIF equivalence has to be argued:

    atrain_real  all_proportions_more/coordinate_parsed_with_coords
    atrain_ctrl  all_proportions_more/coordinate_parsed_with_coords_shuffled
    simval       even_less_spots/.../cellids_left_out/test

Both artifact-A trees carry all 805,549 gene TIFs and their own ``dapi_zoomed``,
and the shuffled tree's cell-mask and DAPI TIFs are byte-identical to the real
tree's -- only the transcript positions move. The 805,454 stems scored are the
matched intersection; the 95 left over are `scale.json`'s
``n_stems_artifact_a_unmatched``.

Streamed in blocks so no arm needs 64 GB resident.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
for p in (REVISION / "src", REVISION / "notebooks" / "reduce"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import paths  # noqa: E402

DUMP_SCHEMA = 1

ARMS = {
    "atrain_real": (paths.SIM_TRAIN / "coordinate_parsed_with_coords",
                    paths.SIM_TRAIN / "coordinate_parsed_with_coords" / "dapi_zoomed",
                    "image_stem"),
    "atrain_ctrl": (paths.SIM_TRAIN / "coordinate_parsed_with_coords_shuffled",
                    paths.SIM_TRAIN / "coordinate_parsed_with_coords_shuffled" / "dapi_zoomed",
                    "image_stem"),
    "simval": (paths.SIM_VAL / "coordinate_parsed_zoomed_again" / "unbalanced"
               / "cellids_left_out" / "test",
               paths.SIM_VAL / "rotated_dapi_zoomed_again",
               "image_stem"),
}


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def write_dump(path, X, obs_index, index_kind, method, dataset, extra=None):
    """Mirror of ``revbench.representations.write_dump``; see the adapters."""
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    keys = np.asarray([str(k) for k in obs_index], dtype=np.str_)
    if keys.shape[0] != X.shape[0]:
        raise ValueError("obs_index %d rows, X %d" % (keys.shape[0], X.shape[0]))
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, X=X, obs_index=keys)
    os.replace(tmp, path)
    meta = {"schema": DUMP_SCHEMA, "method": method, "dataset": dataset,
            "index_kind": index_kind, "n": int(X.shape[0]),
            "latent_dim": int(X.shape[1]),
            "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "producer": "scripts/encode_cnn_baseline.py"}
    meta.update(extra or {})
    with open(path[:-4] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    return path


def _load_one(stem, gene_dir, dapi_dir):
    from skimage import io
    g = np.asarray(io.imread(Path(gene_dir) / (stem + ".tif")), dtype=np.float32)
    d = np.asarray(io.imread(Path(dapi_dir) / (stem + "_DAPI.tif")), dtype=np.float32)
    out = np.empty((2, g.shape[0], g.shape[1]), dtype=np.float32)
    for i, a in enumerate((g, d)):
        lo, hi = float(a.min()), float(a.max())
        out[i] = (a - lo) / (hi - lo) if hi > lo else 0.0
    return out


def stems_for(arm):
    if arm == "simval":
        return [str(s) for s in
                pd.read_parquet(paths.ARTIFACTS / "simval_obs.parquet")["obs_key"]]
    from revbench import controls as C
    name = "reduce_atrain_real" if arm == "atrain_real" else "reduce_atrain_ctrl"
    cs = C.load_control(name, cache_dir=paths.ARTIFACTS / "controls")
    return sorted(set(cs.points["obs_key"].astype(str)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="atrain_real,atrain_ctrl,simval")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--block", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-jobs", type=int, default=32)
    A = ap.parse_args()

    import torch
    from joblib import Parallel, delayed
    sys.path.insert(0, str(HERE))
    from train_cnn_baseline import build_model

    ck = torch.load(paths.CNN_CKPT, map_location="cpu")
    classes = list(ck["classes"])
    rnd = ck["random_class"]
    dev = torch.device(A.device if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes), int(ck["embed_dim"]), dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    log("checkpoint epoch %s | classes %s | embed_dim %s | held-out acc bin %.4f multi %.4f"
        % (ck["epoch"], classes, ck["embed_dim"], ck["holdout_acc_binary"],
           ck["holdout_acc_multi"]))

    out_dir = paths.ARTIFACTS / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in [a for a in A.arms.split(",") if a]:
        gene_dir, dapi_dir, index_kind = ARMS[arm]
        stems = stems_for(arm)
        log("%s: %d stems | %s" % (arm, len(stems), Path(gene_dir).name))
        Z, PB, PM = [], [], []
        t0 = time.time()
        for s in range(0, len(stems), A.block):
            chunk = stems[s:s + A.block]
            imgs = Parallel(n_jobs=A.n_jobs, batch_size=256)(
                delayed(_load_one)(st, str(gene_dir), str(dapi_dir)) for st in chunk)
            Xb = np.stack(imgs)
            with torch.no_grad():
                for b in range(0, len(Xb), A.batch_size):
                    xb = torch.from_numpy(Xb[b:b + A.batch_size]).to(dev)
                    z, lb, lm = model(xb)
                    Z.append(z.cpu().numpy().astype(np.float32))
                    PB.append(torch.softmax(lb, 1).cpu().numpy().astype(np.float32))
                    PM.append(torch.softmax(lm, 1).cpu().numpy().astype(np.float32))
            del Xb, imgs
            log("  %d/%d stems | %.1f min" % (min(s + A.block, len(stems)), len(stems),
                                              (time.time() - t0) / 60))
        Z = np.vstack(Z); PB = np.vstack(PB); PM = np.vstack(PM)
        nonfinite = int((~np.isfinite(Z)).any(axis=1).sum())
        allzero = int((Z == 0).all(axis=1).sum())
        if nonfinite or allzero:
            raise ValueError("%d non-finite and %d all-zero embeddings in %s -- "
                             "run_suite's coverage guard would read the all-zero "
                             "rows as 'not measured'" % (nonfinite, allzero, arm))
        write_dump(out_dir / ("%s.npz" % arm), Z, stems, index_kind,
                   "cnn_baseline", "simulated",
                   extra={"gene_dir": str(gene_dir), "dapi_dir": str(dapi_dir),
                          "arm": arm, "classes": classes, "random_class": rnd,
                          "checkpoint": str(paths.CNN_CKPT),
                          "checkpoint_epoch": int(ck["epoch"]),
                          "input": "2ch gene+dapi TIF, per-image min-max",
                          "embed_dim": int(ck["embed_dim"])})
        lab = pd.DataFrame({"obs_key": stems})
        lab["pred_binary"] = np.where(PB.argmax(1) == 1, "pattern", rnd)
        lab["p_pattern"] = PB[:, 1]
        lab["pred_multi"] = [classes[i] for i in PM.argmax(1)]
        for i, c in enumerate(classes):
            lab["p_" + c] = PM[:, i]
        lab.to_parquet(out_dir / ("labels_%s.parquet" % arm), index=False)
        log("%s: wrote %s and labels (%.1f min) | pred_binary pattern-rate %.4f"
            % (arm, Z.shape, (time.time() - t0) / 60,
               float((lab.pred_binary == "pattern").mean())))
    log("=== CNN_ENCODE_DONE ===")


if __name__ == "__main__":
    main()
