"""A simple supervised CNN on the simulated renders: feature extraction + classifier.

A simple CNN, consisting of feature extraction and a classifier, that performs binary
and multi-class prediction on simulated images: (1) patterned or not and (2) which
pattern subtype. It is a supervised counterpart to the CVAE.

    /mnt/data/miniconda3/envs/cvae_david3/bin/python train_cnn_baseline.py \
        --epochs 15 --device cuda:2

Why this split, and why it is clean
-----------------------------------
The simulated tree carries a **native `random` class**, so no negative arm has to
be constructed and artifact A is not needed for training:

    unbalanced/cellids_left_out/train   175,981 imgs  253 cell ids  (50,309 random)
    unbalanced/cellids_left_out/test     44,706 imgs   64 cell ids  (12,724 random)

The two cell-id sets are disjoint (253 + 64 = 317), the 44,706 `test` images *are*
the validation split every other method is scored on, and **artifact A is built
from those same 64 held-out cells**. So a model trained on `train` has artifact A
and the validation split both fully held out -- the same guarantee the shipped
CVAE has, with no bespoke split and nothing to caveat.

Note the label space here is six classes: five patterns plus `random`. It is
*not* `config.PATTERN_NAMES` (six patterns, no random, includes `protrusion`) and
not `train_subtype_classifier`'s seven (those six plus a `random` contributed by
the shuffled control arm). Three label spaces live in this project; they are not
interchangeable.

Design choices that keep the comparison about architecture
----------------------------------------------------------
* **Same input as the CVAE**: two channels, the blurred gene render and the graded
  DAPI, 100x100, each per-image min-max normalised -- which is what
  `cvae_encode.prepare_batch` proves the CVAE's own `prepare_new_input` collapses
  to.
* **A 15-dimensional penultimate layer**, matching the CVAE's latent. That is not
  cosmetic: `detection.dataset_threshold` is calibrated for a 15-dim latent, so a
  15-dim embedding is the only one whose fixed-threshold detection numbers are
  comparable without a caveat.
* Two linear heads off that layer, so the "feature extraction" and the
  "classifier" are separable exactly as the comment describes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
for p in (REVISION / "src", REVISION / "notebooks" / "reduce"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paths  # noqa: E402  (notebooks/reduce/paths.py -- the one place a path is named)

TREE = paths.SIM_VAL / "coordinate_parsed_zoomed_again" / "unbalanced" / "cellids_left_out"
DAPI_DIR = paths.SIM_VAL / "rotated_dapi_zoomed_again"
RANDOM_CLASS = "random"
_PAT_RE = re.compile(r"^([a-zA-Z-]+)_\d+it_")
_CELL_RE = re.compile(r"(?:icell|cellID)(\d+)")


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def scan(split: str):
    """(stems, patterns, cell_ids) for one split, parsed from the filenames."""
    d = TREE / split
    stems, pats, cells = [], [], []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".tif"):
            continue
        stem = f[:-4]
        m, c = _PAT_RE.match(stem), _CELL_RE.search(stem)
        if not m or not c:
            raise ValueError("cannot parse %r" % stem)
        stems.append(stem)
        pats.append(m.group(1))
        cells.append(c.group(1))
    return stems, np.asarray(pats), np.asarray(cells)


def _load_one(stem, split):
    """(2, 100, 100) float32: blurred gene render + graded DAPI, each min-maxed."""
    from skimage import io
    g = np.asarray(io.imread(TREE / split / (stem + ".tif")), dtype=np.float32)
    d = np.asarray(io.imread(DAPI_DIR / (stem + "_DAPI.tif")), dtype=np.float32)
    out = np.empty((2, g.shape[0], g.shape[1]), dtype=np.float32)
    for i, a in enumerate((g, d)):
        lo, hi = float(a.min()), float(a.max())
        out[i] = (a - lo) / (hi - lo) if hi > lo else 0.0
    return out


def load_split(stems, split, n_jobs=32):
    from joblib import Parallel, delayed
    t0 = time.time()
    arrs = Parallel(n_jobs=n_jobs, batch_size=256, verbose=0)(
        delayed(_load_one)(s, split) for s in stems)
    X = np.stack(arrs)
    log("loaded %s: %s (%.1f GB) in %.1f min"
        % (split, X.shape, X.nbytes / 2**30, (time.time() - t0) / 60))
    return X


def build_model(n_classes, embed_dim, device):
    import torch.nn as nn

    class CNNBaseline(nn.Module):
        """Feature extractor -> 15-d embedding -> two linear heads."""

        def __init__(self):
            super().__init__()
            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                     nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                     nn.Conv2d(o, o, 3, padding=1),
                                     nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2))
            self.features = nn.Sequential(blk(2, 32), blk(32, 64), blk(64, 128),
                                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.embed = nn.Linear(128, embed_dim)
            self.head_binary = nn.Linear(embed_dim, 2)
            self.head_multi = nn.Linear(embed_dim, n_classes)

        def forward(self, x):
            z = self.embed(self.features(x))
            return z, self.head_binary(z), self.head_multi(z)

    return CNNBaseline().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=15)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--holdout-cells", type=int, default=25,
                    help="train cell ids reserved for early stopping (NOT the test set)")
    ap.add_argument("--seed", type=int, default=101)
    A = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(A.seed)
    np.random.seed(A.seed)

    stems, pats, cells = scan("train")
    classes = sorted(set(pats))
    log("train: %d images | %d cell ids | classes %s"
        % (len(stems), len(set(cells)), classes))
    assert RANDOM_CLASS in classes, "no native %r class to use as the negative" % RANDOM_CLASS

    # inner hold-out by CELL ID, so early stopping never sees a training cell's
    # other rotations. Disjoint from the 64 test cell ids by construction.
    rng = np.random.RandomState(A.seed)
    uniq = np.array(sorted(set(cells)))
    rng.shuffle(uniq)
    val_cells = set(uniq[:A.holdout_cells])
    is_val = np.array([c in val_cells for c in cells])
    log("inner split: %d train / %d early-stop images (%d/%d cell ids)"
        % ((~is_val).sum(), is_val.sum(), len(uniq) - len(val_cells), len(val_cells)))

    X = load_split(stems, "train", n_jobs=A.n_jobs)
    y_multi = np.array([classes.index(p) for p in pats], dtype=np.int64)
    y_bin = (pats != RANDOM_CLASS).astype(np.int64)      # 1 = patterned, 0 = random

    dev = torch.device(A.device if torch.cuda.is_available() else "cpu")
    log("device %s" % dev)
    model = build_model(len(classes), A.embed_dim, dev)
    n_par = sum(p.numel() for p in model.parameters())
    log("model: %d parameters, embed_dim %d, %d classes" % (n_par, A.embed_dim, len(classes)))

    opt = torch.optim.Adam(model.parameters(), lr=A.lr)
    ce = nn.CrossEntropyLoss()
    paths.CNN_RUNS.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(paths.CNN_RUNS / time.strftime("cnn_%Y%m%d_%H%M%S")))

    tr_idx = np.flatnonzero(~is_val)
    va_idx = np.flatnonzero(is_val)
    Xt = torch.from_numpy(X)
    yb = torch.from_numpy(y_bin)
    ym = torch.from_numpy(y_multi)

    def evaluate(idx):
        model.eval()
        nb = nm = n = 0
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                j = idx[s:s + 1024]
                xb = Xt[j].to(dev, non_blocking=True)
                _, lb, lm = model(xb)
                nb += int((lb.argmax(1).cpu() == yb[j]).sum())
                nm += int((lm.argmax(1).cpu() == ym[j]).sum())
                n += len(j)
        return nb / n, nm / n

    best, best_ep, hist = -1.0, -1, []
    for ep in range(A.epochs):
        model.train()
        rng.shuffle(tr_idx)
        tot = 0.0
        t0 = time.time()
        for s in range(0, len(tr_idx), A.batch_size):
            j = tr_idx[s:s + A.batch_size]
            xb = Xt[j].to(dev, non_blocking=True)
            _, lb, lm = model(xb)
            loss = ce(lb, yb[j].to(dev)) + ce(lm, ym[j].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        acc_b, acc_m = evaluate(va_idx)
        tr_loss = tot / len(tr_idx)
        hist.append({"epoch": ep, "train_loss": tr_loss,
                     "holdout_acc_binary": acc_b, "holdout_acc_multi": acc_m})
        writer.add_scalar("loss/train", tr_loss, ep)
        writer.add_scalar("acc/holdout_binary", acc_b, ep)
        writer.add_scalar("acc/holdout_multi", acc_m, ep)
        log("epoch %2d | loss %.4f | held-out acc binary %.4f multi %.4f | %.1f min"
            % (ep, tr_loss, acc_b, acc_m, (time.time() - t0) / 60))
        score = acc_b + acc_m
        if score > best:
            best, best_ep = score, ep
            paths.CNN_CKPT.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(paths.CNN_CKPT) + ".tmp")
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "random_class": RANDOM_CLASS, "embed_dim": A.embed_dim,
                        "epoch": ep, "holdout_acc_binary": acc_b,
                        "holdout_acc_multi": acc_m, "seed": A.seed,
                        "n_parameters": n_par, "lr": A.lr,
                        "batch_size": A.batch_size,
                        "train_split": str(TREE / "train"),
                        "n_train_images": int(len(tr_idx)),
                        "holdout_cell_ids": sorted(val_cells),
                        "history": hist}, tmp)
            tmp.replace(paths.CNN_CKPT)
    writer.close()
    log("best epoch %d (binary %.4f, multi %.4f) -> %s"
        % (best_ep, hist[best_ep]["holdout_acc_binary"],
           hist[best_ep]["holdout_acc_multi"], paths.CNN_CKPT))
    (paths.CNN_CKPT.parent / "cnn_baseline_history.json").write_text(
        json.dumps({"classes": classes, "best_epoch": best_ep, "history": hist,
                    "n_parameters": n_par}, indent=2))
    log("=== CNN_TRAIN_DONE ===")


if __name__ == "__main__":
    main()
