#!/usr/bin/env python
# Provenance only: this script produced the 2-channel and 3-channel checkpoints (100 and 300 epochs) compared in cell_mask_channel.ipynb. Needs the original simulation tree and a GPU.
# Paths are those of the original run; it is not meant to run unchanged.
"""Train the 2-channel and 3-channel CVAEs for 300 epochs, checkpointing every 10.

    cd /path/to/CVAE          # the CVAE training code; ./runs/ is relative
    nohup python ~/SubCellSpace/additional_analyses/scripts/cell_mask/train_long_2ch_3ch.py \\
        --channels 2 -device 1 -epochs 300 > /path/to/work/epochs/train_2ch.log 2>&1 &
    nohup python ~/SubCellSpace/additional_analyses/scripts/cell_mask/train_long_2ch_3ch.py \\
        --channels 3 -device 2 -epochs 300 > /path/to/work/epochs/train_3ch.log 2>&1 &

Why this exists
---------------
Both shipped comparisons stopped at 100 epochs. At that point the 2-channel
model's held-out loss was still falling and its best epoch was its last, which is
exactly what an undertrained model looks like -- so "the 3-channel model overfits"
could instead be "we stopped both too early, and they were on different parts of
their curves". This runs both to 300 epochs on the same split and keeps a
checkpoint every 10, so the claim can be checked against the whole curve rather
than one arbitrary cut, and so downstream metrics can be evaluated at any epoch
rather than only at the end.

It is `train_3channel_heldout.py` generalised over the channel count. Both arms
use the same seed, the same pre-split `cellids_left_out/{train,test}` directories,
the same DAPI images, and the same hyperparameters; the *only* difference between
them is 2 vs 3 input channels and the matching dataset class. The model and
training code are imported from `~/CVAE`, unmodified, so the runs are built the
same way the existing checkpoints were.

Checkpointing without disturbing training
-----------------------------------------
`train_model_with_validation_with_2_classifiers` creates all four Adam optimizers
and both classifier heads *before* its epoch loop, so calling it repeatedly in
10-epoch chunks would reset optimizer moments and reinitialise the heads -- a
different training run, not the same one observed more often. Instead the epoch
loop's `trange` is swapped for a generator that saves after each epoch it yields.
The training code is untouched and the optimizer state is continuous; the only
added work is a `torch.save` every 10 epochs.
"""
import argparse
import inspect
import os
import random
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

CVAE_REPO = "/path/to/CVAE"   # the CVAE training code repository
sys.path.insert(0, CVAE_REPO)
import CVAE as CVAE_mod                                     # noqa: E402
from CVAE import CVAE_hard_coded                            # noqa: E402
from utils import (ImageDatasetWithDapisAndLabels,          # noqa: E402
                   ImageDatasetWithDapisAndCellsAndLabels)

SEED = 101
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path("/mnt/data/david/simulated_cells/with_dapi/even_less_spots")
SPLIT = ROOT / "coordinate_parsed_zoomed_again/unbalanced/cellids_left_out"
TRAIN_GLOB_FULL = str(SPLIT / "train" / "*.tif")
VAL_GLOB_FULL = str(SPLIT / "test" / "*.tif")
DAPI_GLOB = str(ROOT / "rotated_dapi_zoomed_again" / "*.tif")
CELL_MASK_DIR = ROOT / "coordinate_parsed_with_coords" / "cell_mask_zoomed"
OUT_ROOT = Path("/path/to/work/epochs")


#: The four Adam optimizers `train_model_with_validation_with_2_classifiers` builds
#: as locals before its epoch loop. Captured so a later extension can be a real
#: resume rather than a fresh run: without their moments, restarting reinitialises
#: Adam *and* -- because the training call rebuilds both heads at CVAE.py:382-386 --
#: throws away the trained classifiers, which is a different trajectory, not a
#: continuation. Only written on `opt_every` checkpoints, since the state roughly
#: triples a checkpoint's size and only the last one is ever resumed from.
OPTIMIZER_LOCALS = ("optimizer", "optimizer_enc",
                    "binary_classify_optimizer", "classify_optimizer")


def checkpointing_trange(total_epochs, every, save_fn, opt_every=100):
    """A `trange` stand-in that calls `save_fn(epoch, losses, opt)` after each epoch.

    The generator yields the epoch index; control returns here once the loop body
    for that epoch has finished, which is where the save happens. `losses` is read
    out of the calling frame so that every checkpoint carries its own history and
    a run that dies at epoch 250 still leaves 25 usable ones behind. The optimizer
    states are read the same way, on a coarser cadence.
    """
    from tqdm import trange as _trange

    def gen(n, *a, **kw):
        if n != total_epochs:                    # some other trange in the module
            for i in _trange(n, *a, **kw):
                yield i
            return
        bar = _trange(n, *a, **kw)
        for i in bar:
            yield i
            if (i + 1) % every == 0 or (i + 1) == n:
                losses, opt = None, None
                try:
                    caller = inspect.currentframe().f_back.f_locals
                    losses = caller.get("losses")
                    if (i + 1) % opt_every == 0 or (i + 1) == n:
                        opt = {k: caller[k].state_dict()
                               for k in OPTIMIZER_LOCALS if k in caller}
                except Exception:
                    pass
                save_fn(i + 1, losses, opt)
    return gen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", type=int, choices=(2, 3), required=True)
    ap.add_argument("-epochs", type=int, nargs="?", default=300)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--optimizer-every", type=int, default=100,
                    help="save Adam state this often (and at the final epoch)")
    ap.add_argument("--out-dir", default=None,
                    help="checkpoint directory; defaults to <work>/epochs/<arm>")
    ap.add_argument("-lr", type=float, nargs="?", default=1e-5)
    ap.add_argument("-class_lr", type=float, nargs="?", default=1e-3)
    ap.add_argument("-n_classes", type=int, nargs="?", default=6)
    ap.add_argument("-beta", type=float, nargs="?", default=0.0001)
    ap.add_argument("-batch_size", type=int, nargs="?", default=512)
    ap.add_argument("-n_latent_dims", type=int, nargs="?", default=15)
    ap.add_argument("-device", type=int, nargs="?", default=1)
    ap.add_argument("-reduction", type=str, nargs="?", default="sum")
    ap.add_argument("--train-glob", default=TRAIN_GLOB_FULL)
    ap.add_argument("--val-glob", default=VAL_GLOB_FULL)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    arm = "%dch" % args.channels
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [str(Path(f).stem).split("_")[0] for f in glob(TRAIN_GLOB_FULL)]
    unique_patterns = sorted(set(patterns))
    print("arm: %s | patterns: %s" % (arm, unique_patterns), flush=True)
    assert args.n_classes == len(unique_patterns)

    # Pre-flight: the dataset class raises on a missing file, which would waste hours.
    dapi_dir = Path(DAPI_GLOB).parent
    missing_d, missing_c, n = [], [], 0
    for g in (args.train_glob, args.val_glob):
        for f in glob(g):
            stem = Path(f).stem
            n += 1
            if not (dapi_dir / (stem + "_DAPI.tif")).is_file():
                missing_d.append(stem)
            if args.channels == 3 and not (CELL_MASK_DIR / (stem + "_cell-mask.tif")).is_file():
                missing_c.append(stem)
    print("pre-flight: %d gene images | missing DAPI %d | missing cell mask %d"
          % (n, len(missing_d), len(missing_c)), flush=True)
    assert not missing_d and not missing_c, "incomplete channels; not starting a long run"
    if args.check_only:
        return

    common = dict(dapi_prefixes=["icell", "rotated"], n_classes=args.n_classes,
                  unique_patterns=unique_patterns, also_binary=True)
    if args.channels == 3:
        def build(g):
            return ImageDatasetWithDapisAndCellsAndLabels(
                g, DAPI_GLOB, CELL_MASK_DIR,
                replace_dapi_to_cell={"DAPI": "cell-mask"}, **common)
    else:
        def build(g):
            return ImageDatasetWithDapisAndLabels(g, DAPI_GLOB, **common)

    print("loading train set (eager)", flush=True)
    train_data = build(args.train_glob)
    print("loading val set", flush=True)
    val_data = build(args.val_glob)
    print("train %s | val %s" % (tuple(train_data.scaled_data.shape),
                                 tuple(val_data.scaled_data.shape)), flush=True)
    # `scaled_data` holds the gene channel only; the channels are concatenated in
    # __getitem__, so check an actual sample rather than the backing array.
    sample_ch = train_data[0][0].shape[0]
    print("sample tensor: %s -> %d channels" % (tuple(train_data[0][0].shape), sample_ch),
          flush=True)
    assert sample_ch == args.channels, \
        "dataset yields %d channels, expected %d" % (sample_ch, args.channels)

    cvae = CVAE_hard_coded(n_channels=args.channels,
                           img_shape=tuple(train_data.scaled_data.shape)[2:],
                           latent_dim=args.n_latent_dims, kernel_size=3)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                             shuffle=False)

    model_name = ("%s_cvae_%s_cellidsleftout_lr%.0e_classlr%.0e_nclasses%d_beta%.0e_"
                  "%dld_%depochs" % (datetime.now().strftime("%Y-%m-%d"), arm, args.lr,
                                     args.class_lr, args.n_classes, args.beta,
                                     args.n_latent_dims, args.epochs))
    print("model name: %s" % model_name, flush=True)

    writer = SummaryWriter("./runs/%s/" % model_name)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    cvae.to(args.device)
    args_dict = vars(args)

    def payload(epochs_done, losses):
        d = {"cvae": cvae, "cvae_state_dict": cvae.state_dict(),
             "epochs": epochs_done, "epochs_planned": args.epochs,
             "losses": losses, "lr": args.lr, "ld": args.n_latent_dims,
             "n_channels": args.channels, "arm": arm,
             "dataset": args.train_glob, "test_dataset": args.val_glob,
             "dapi_dataset": DAPI_GLOB, "beta": args.beta, "device": args.device,
             "n_classes": args.n_classes, "class_lr": args.class_lr,
             "unique_patterns": unique_patterns,
             "dapi_prefixes": ["icell", "rotated"], "reduction": args.reduction,
             "seed": SEED}
        if args.channels == 3:
            d["cell_mask_dir"] = CELL_MASK_DIR
        return d

    def save_checkpoint(epochs_done, losses, opt=None):
        p = out_dir / ("%s__epoch%03d" % (model_name, epochs_done))
        d = payload(epochs_done, losses)
        if opt:
            d["optimizer_state"] = opt
        torch.save(d, p)
        print("[checkpoint] epoch %d -> %s%s"
              % (epochs_done, p.name, "  (+optimizer)" if opt else ""), flush=True)

    # Swap the epoch loop's trange for one that checkpoints. Training code untouched.
    original_trange = CVAE_mod.trange
    CVAE_mod.trange = checkpointing_trange(args.epochs, args.checkpoint_every,
                                           save_checkpoint,
                                           opt_every=args.optimizer_every)
    try:
        losses = cvae.train_model_with_validation_with_2_classifiers(
            train_loader, val_loader, args_dict, writer=writer, with_dapi=True)
    finally:
        CVAE_mod.trange = original_trange

    # The last in-loop checkpoint (epoch == args.epochs) already carries the
    # optimizer state; this copy is written after the training call has returned,
    # where the optimizers are out of scope, so it deliberately has none.
    final = out_dir / model_name
    torch.save(payload(args.epochs, losses), final)
    print("saved final: %s  (no optimizer state -- resume from __epoch%03d)"
          % (final, args.epochs), flush=True)


if __name__ == "__main__":
    main()
