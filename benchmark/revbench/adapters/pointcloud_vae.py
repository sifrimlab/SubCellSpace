"""Embed a shared point set with the PointNet-VAE competitor.

Runs in the competitor's environment (``/mnt/data/miniconda3/envs/point_cloud``:
python 3.12, torch 2.3), NOT in ``subcellspace_env``. It therefore imports
nothing from ``revbench`` and carries its own copy of the dump writer.

    /mnt/data/miniconda3/envs/point_cloud/bin/python pointcloud_vae.py \
        --points   /mnt/data/david/benchmarks/cache/control_ileum_pc.parquet \
        --checkpoint /mnt/data/.../rotated_unbalanced_out_dir_64_kld_w/epoch_399.pt \
        --latent-dim 64 \
        --out      /mnt/data/david/benchmarks/cache/pointnet__ileum_control.npz \
        --method   pointnet_vae_64 --dataset ileum

Why a parquet rather than the CSV-per-observation layout the repo reads today:
the APEX control has ~8.4M (cell, gene) observations, and 8.4M small files on a
98%-full filesystem is not a reasonable ask.

Two gotchas in the upstream reader that this bypasses:

* ``point_clouds/data.py`` looks for a nucleus column named ``nucleus`` while
  every CSV on disk names it ``in_dapi``; with ``img_dir=None`` it silently
  falls back to all-False, dropping the nucleus channel entirely. We pass the
  flag explicitly.
* ``parse_coords_csv`` extracts the cell id with ``re.findall(r'icell(\\d+)')``,
  which raises IndexError on the ``cellID123`` naming used by the shuffled
  trees. We parse metadata ourselves and accept both spellings.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DUMP_SCHEMA = 1

_CELL_RE = re.compile(r"(?:icell|cellID)(\d+)")
_ROT_RE = re.compile(r"rotated(\d+)")
_PROP_RE = re.compile(r"prop(\d\.\d+)")
_IT_RE = re.compile(r"(\d+)it")


def write_dump(path, X, obs_index, index_kind, method, dataset, extra=None):
    """Mirror of ``revbench.representations.write_dump``.

    Duplicated deliberately: this module runs where revbench is not installed.
    Keep the arrays and the ``schema`` field in step with that function.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    keys = np.asarray([str(k) for k in obs_index], dtype=np.str_)
    if keys.shape[0] != X.shape[0]:
        raise ValueError("obs_index has {} entries, X has {} rows"
                         .format(keys.shape[0], X.shape[0]))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, X=X, obs_index=keys)
    meta = {"schema": DUMP_SCHEMA, "method": method, "dataset": dataset,
            "index_kind": index_kind, "n": int(X.shape[0]),
            "latent_dim": int(X.shape[1]),
            "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "producer": "revbench.adapters.pointcloud_vae"}
    meta.update(extra or {})
    with open(os.path.splitext(path)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    return path


def _meta_from_key(key):
    """Best-effort metadata from an observation key.

    Only used for bookkeeping -- we bypass ``build_training_dataset``, so none
    of it drives stratification. Missing pieces default rather than raise,
    which is the difference between this and ``parse_coords_csv``.
    """
    cell = _CELL_RE.search(key)
    rot = _ROT_RE.search(key)
    prop = _PROP_RE.search(key)
    it = _IT_RE.search(key)
    return {
        "cell_type": key.split("_")[0],
        "cell_id": int(cell.group(1)) if cell else -1,
        "rotation": int(rot.group(1)) if rot else 0,
        "proportion": float(prop.group(1)) if prop else float("nan"),
        "iteration": int(it.group(1)) if it else 0,
    }


def build_clouds(points_df, nucleus_col=None):
    """One PointCloud per ``obs_key``, in a stable order."""
    from point_clouds.data import Point, PointCloud

    if nucleus_col is None:
        for cand in ("nucleus", "in_dapi"):
            if cand in points_df.columns:
                nucleus_col = cand
                break
    if nucleus_col is None:
        raise KeyError("no nucleus column found; expected 'nucleus' or 'in_dapi'. "
                       "Without it the nucleus channel would be silently empty.")

    clouds, keys = [], []
    for key, grp in points_df.groupby("obs_key", sort=True):
        m = _meta_from_key(str(key))
        pts = [Point(float(r), float(c))
               for r, c in zip(grp["row"].values, grp["col"].values)]
        clouds.append(PointCloud(
            cell_type=m["cell_type"], iteration=m["iteration"],
            n_points=len(pts), cell_id=m["cell_id"],
            proportion=m["proportion"], rotation=m["rotation"],
            points=pts, nucleus=[bool(v) for v in grp[nucleus_col].values],
            original_coords_file=str(key)))
        keys.append(str(key))
    return clouds, keys


def embed_clouds(clouds, checkpoint, latent_dim=64, device="cpu", batch_size=128):
    """Encode with the competitor's model, mirroring ``evaluation.embed_data``."""
    import torch
    from torch.utils.data import DataLoader
    from point_clouds import evaluation, utils
    from point_clouds.data import PointCloudDataset

    model = evaluation.load_checkpoint(checkpoint, device=torch.device(device),
                                       n_latent_dims=latent_dim)
    loader = DataLoader(PointCloudDataset(clouds), batch_size=batch_size,
                        shuffle=False, collate_fn=utils.data_collate_fn)
    out = []
    with torch.no_grad():
        for inputs, _, _ in loader:
            mu, _ = model.encode(inputs)
            out += mu.unbind()
    # encode returns a nested tensor; each entry is (1, d)
    return np.stack([e.numpy(force=True).copy(order="C")[0, :] for e in out], axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", required=True, help="control/points parquet")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--method", default="pointnet_vae")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--repo", default="/mnt/data/david/sifrim-point-cloud-representation-learning")
    ap.add_argument("--limit", type=int, default=None, help="first N observations only")
    ap.add_argument("--index-kind", default="canonical_key")
    args = ap.parse_args(argv)

    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)

    df = pd.read_parquet(args.points)
    if args.limit is not None:
        keep = df["obs_key"].drop_duplicates().head(args.limit)
        df = df[df["obs_key"].isin(set(keep))]
    clouds, keys = build_clouds(df)
    print("built {} point clouds ({} points)".format(len(clouds), len(df)), flush=True)

    X = embed_clouds(clouds, args.checkpoint, latent_dim=args.latent_dim,
                     device=args.device, batch_size=args.batch_size)
    print("embedded -> {}".format(X.shape), flush=True)

    fp = None
    side = os.path.splitext(args.points)[0] + ".json"
    if os.path.isfile(side):
        with open(side) as fh:
            fp = json.load(fh).get("fingerprint")

    write_dump(args.out, X, keys, args.index_kind, args.method, args.dataset,
               extra={"checkpoint": args.checkpoint, "latent_dim": args.latent_dim,
                      "control_fingerprint": fp, "points_file": args.points,
                      "nucleus_channel": "explicit"})
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
