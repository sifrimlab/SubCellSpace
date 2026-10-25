"""Embed the real datasets' (cell, gene) pairs with the trained CNN baseline.

Extends `cnn_baseline` from simulated-only to ileum and APEX. The CNN takes the
same two channels it was trained on, rebuilt from real data:

* **channel 0**, the gene image: the pair's transcripts rendered with
  `cvae_encode.render_batch` -- scatter-then-blur at sigma 1.5, pixel *set* to 1
  so duplicates collapse. That is `controls.render_image`'s convention, and it
  reproduces the simulated training TIFs exactly (verified: correlation 1.0000
  between a rendered ControlSet and the tree's own `_blurred1.5.tif`).
* **channel 1**, the nucleus image: the cell's `obsm/dapi_mask` with its **stored
  graded values**, not thresholded. `H5adMasks.iter_masks`' docstring is explicit
  that this graded image is what the CVAE's second channel is, so thresholding
  would feed the model something it never saw. Read chunk-aligned.

Each channel is per-image min-max normalised, exactly as `train_cnn_baseline`'s
loader does and as `cvae_encode.prepare_batch` proves the CVAE's own
preprocessing collapses to.

All the pairs of a cell share one DAPI image, so it is read once per cell and
broadcast. One pass of `iter_point_blocks` per arm -- that reader costs far more
per cell at small block sizes, so `--read-block` stays at the proven 5,000.

    /mnt/data/miniconda3/envs/cvae_david3/bin/python encode_cnn_real.py apex_hek \
        --device cuda:0
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
_REVISION = HERE.parent
for _p in (_REVISION / "src", _REVISION / "notebooks" / "reduce"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import paths  # noqa: E402

ARM_FIELDS = {"real": ("subcell_row", "subcell_col"),
              "ctrl": ("subcell_row_shuffled", "subcell_col_shuffled")}
DUMP_SCHEMA = 1


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
            "producer": "scripts/encode_cnn_real.py"}
    meta.update(extra or {})
    with open(path[:-4] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ileum", "apex_hek"])
    ap.add_argument("--arms", default="real,ctrl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--read-block", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=4096)
    A = ap.parse_args()

    import torch
    from revbench import ella_io as EIO
    from revbench.cvae_encode import render_batch
    from revbench.detection import explode_matched
    from revbench.geometry import RENDER_SHAPE
    sys.path.insert(0, str(HERE))
    from train_cnn_baseline import build_model

    H5 = {"ileum": paths.ILEUM, "apex_hek": paths.APEX}[A.dataset]

    # the pair set is the pipeline's, never the file's
    real, _ = explode_matched(H5)
    pair_key = (real.obs["original_obs_index"].astype(str) + "__"
                + real.obs["gene_name"].astype(str)).to_numpy()
    WANT = set(str(k) for k in pair_key)
    N_PAIRS = len(pair_key)
    log("%s: %d exploded pairs to cover" % (A.dataset, N_PAIRS))
    del real

    ck = torch.load(paths.CNN_CKPT, map_location="cpu")
    dev = torch.device(A.device if torch.cuda.is_available() else "cpu")
    model = build_model(len(ck["classes"]), int(ck["embed_dim"]), dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    classes = list(ck["classes"])
    log("checkpoint epoch %s | embed_dim %s | classes %s"
        % (ck["epoch"], ck["embed_dim"], classes))

    masks = EIO.H5adMasks(H5)
    out_dir = paths.ARTIFACTS / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = RENDER_SHAPE

    for arm in [a for a in A.arms.split(",") if a]:
        out = out_dir / ("%s_%s.npz" % (A.dataset, arm))
        if out.exists():
            log("skip %s %s (already built)" % (A.dataset, arm))
            continue
        t0 = time.time()
        Z, PB, PM, KEYS = [], [], [], []
        n_blocks = 0
        for bi, blk_cells, points in EIO.iter_point_blocks(
                H5, fields=ARM_FIELDS[arm], block_cells=A.read_block,
                with_in_dapi=False):
            if points.empty:
                continue
            keep = pd.Series(points["obs_key"].to_numpy()).isin(WANT).to_numpy()
            if not keep.any():
                continue
            p = points.loc[keep]
            n_blocks += 1

            # one graded DAPI per cell, read chunk-aligned, reused by its pairs
            cells_arr, _ = EIO.split_keys(p["obs_key"].to_numpy(), "cell_gene")
            cells_here = sorted(set(str(c) for c in cells_arr))
            dapi = {}
            for cell, _cm, dm in masks.iter_masks(cells_here, raw=True):
                dapi[cell] = (np.zeros(RENDER_SHAPE, dtype=np.float32) if dm is None
                              else np.asarray(dm, dtype=np.float32))

            # group the block's points by pair, order fixed and recorded
            g = p.groupby("obs_key", sort=True)
            keys = list(g.groups.keys())
            rowcols = [(p["row"].to_numpy()[idx], p["col"].to_numpy()[idx])
                       for idx in (g.indices[k] for k in keys)]
            owner = [k.rsplit("__", 1)[0] for k in keys]

            for s in range(0, len(keys), A.batch_size):
                rc = rowcols[s:s + A.batch_size]
                ow = owner[s:s + A.batch_size]
                gene = render_batch(rc, device=dev, dtype=torch.float32)  # (n,1,H,W)
                dp = torch.from_numpy(
                    np.stack([dapi.get(c, np.zeros(RENDER_SHAPE, np.float32))
                              for c in ow])).to(dev).unsqueeze(1)
                x = torch.cat([gene, dp], dim=1)                          # (n,2,H,W)
                # per-image, per-channel min-max -- identical to the trainer
                flat = x.reshape(x.shape[0], 2, -1)
                lo = flat.min(dim=2).values[:, :, None, None]
                hi = flat.max(dim=2).values[:, :, None, None]
                x = torch.where(hi > lo, (x - lo) / (hi - lo), torch.zeros_like(x))
                with torch.no_grad():
                    z, lb, lm = model(x)
                Z.append(z.cpu().numpy().astype(np.float32))
                PB.append(torch.softmax(lb, 1).cpu().numpy().astype(np.float32))
                PM.append(torch.softmax(lm, 1).cpu().numpy().astype(np.float32))
                KEYS.extend(keys[s:s + A.batch_size])
            del dapi
            if n_blocks % 5 == 0:
                log("  %s %s: %d blocks | %d pairs | %.1f min"
                    % (A.dataset, arm, n_blocks, len(KEYS), (time.time() - t0) / 60))

        Z = np.vstack(Z); PB = np.vstack(PB); PM = np.vstack(PM)
        dup = len(KEYS) - len(set(KEYS))
        if dup:
            raise ValueError("%d duplicate pair keys in %s %s" % (dup, A.dataset, arm))
        # assert what the PIPELINE retains: short coverage would shrink
        # shared_pair_keys for EVERY method and move the published numbers
        if len(KEYS) != N_PAIRS:
            raise ValueError("%s %s covers %d of %d exploded pairs"
                             % (A.dataset, arm, len(KEYS), N_PAIRS))
        nonfinite = int((~np.isfinite(Z)).any(axis=1).sum())
        allzero = int((Z == 0).all(axis=1).sum())
        if nonfinite or allzero:
            raise ValueError("%d non-finite and %d all-zero embeddings -- an "
                             "all-zero row reads as 'not measured' to "
                             "from_adata and would trip run_suite's coverage "
                             "guard for every method" % (nonfinite, allzero))
        write_dump(out, Z, KEYS, "cell_gene", "cnn_baseline", A.dataset, extra={
            "arm": arm, "classes": classes, "random_class": ck["random_class"],
            "checkpoint": str(paths.CNN_CKPT), "checkpoint_epoch": int(ck["epoch"]),
            "embed_dim": int(ck["embed_dim"]), "read_block": A.read_block,
            "input": ("2ch: render_batch(sigma=1.5) of the pair's transcripts + "
                      "the cell's GRADED obsm/dapi_mask; per-image min-max each"),
            "n_pairs_expected": N_PAIRS, "minutes": round((time.time() - t0) / 60, 1),
        })
        lab = pd.DataFrame({"obs_key": KEYS})
        lab["pred_binary"] = np.where(PB.argmax(1) == 1, "pattern", ck["random_class"])
        lab["p_pattern"] = PB[:, 1]
        lab["pred_multi"] = [classes[i] for i in PM.argmax(1)]
        for i, c in enumerate(classes):
            lab["p_" + c] = PM[:, i]
        lab.to_parquet(out_dir / ("labels_%s_%s.parquet" % (A.dataset, arm)),
                       index=False)
        log("%s %s: %s in %.1f min | coverage %d/%d OK | pattern-rate %.4f"
            % (A.dataset, arm, Z.shape, (time.time() - t0) / 60, len(KEYS),
               N_PAIRS, float((lab.pred_binary == "pattern").mean())))
    log("=== CNN_REAL_%s_DONE ===" % A.dataset.upper())


if __name__ == "__main__":
    main()
