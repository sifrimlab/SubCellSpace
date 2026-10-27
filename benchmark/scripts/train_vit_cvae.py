#!/usr/bin/env python
"""Train a ViT-encoder CVAE on the same held-out split the shipped model used.

    /mnt/data/miniconda3/envs/vit_env/bin/python scripts/train_vit_cvae.py \
        --variant small -device 0

Mirrors `scripts/train_3channel_heldout.py`, which is the precedent for a
held-out retrain of this architecture: same dataset class, same training call,
same saved fields. Two differences, both deliberate:

* **two channels, not three.** These variants ablate the *encoder*, so they take
  the shipped model's own inputs -- the blurred gene render and the graded DAPI.
* **no pickled model object in the checkpoint.** `train_3channel_heldout.py`
  saves `{"cvae": cvae, ...}`, which needs the model's class importable wherever
  the file is read. `05_ablations` runs in `subcellspace_env`, where
  `transformers` does not exist, and it opens every registered checkpoint just to
  read the loss curve. Weights and scalars load anywhere; a pickled
  `transformers.ViTModel` does not. `vit_cvae.load_vit_cvae` rebuilds from the
  state dict, which costs nothing.

What is held fixed, so only the encoder differs
-----------------------------------------------
ld 15, beta 1e-4, lr 1e-5, class_lr 1e-3, n_classes 6, batch 512, reduction sum,
seed 101, the `cellids_left_out/train` split (175,981 images over 253 cell ids,
disjoint from the 64 cell ids of `test`), the two `sample_classifier` heads, and
the decoder -- which is `CVAE_hard_coded`'s untouched, so the reconstruction
target stays 100x100 and `reconstruction_loss` / `val_loss` remain on the
reference model's scale.

**No augmentation, geometric or intensity.** `01_simulated` scores
`rotation_interval_45` and `n_spots_interval_fixed` as negative controls, and
rotation augmentation would force rotation invariance -- making the variant look
good on a control that is meant to measure the encoder, not the augmentation.

Measured cost per 100 epochs, batch 512 on one A40 (5 encoder forward+backward
passes per batch, which is what the trainer does):

    reference conv 3.73M   2.9 min/epoch    4.8 h
    vit small      3.83M   5.1 min/epoch    8.5 h
    vit b16        86.2M   OOM at 512; 30.6 min/epoch at batch 128 with bf16,
                           i.e. 51 h -- see --amp and --batch_size
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
for _p in (REVISION / "src", REVISION / "notebooks" / "reduce"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import paths  # noqa: E402

sys.path.insert(0, str(paths.CVAE_REPO))          # CVAE_hard_coded + the datasets
os.environ.setdefault("HF_HOME", str(paths.HF_HOME))
os.environ.setdefault("MPLBACKEND", "Agg")

SEED = 101
ROOT = paths.SIM_VAL
SPLIT = ROOT / "coordinate_parsed_zoomed_again" / "unbalanced" / "cellids_left_out"
TRAIN_GLOB = str(SPLIT / "train" / "*.tif")
VAL_GLOB = str(SPLIT / "test" / "*.tif")
DAPI_GLOB = str(ROOT / "rotated_dapi_zoomed_again" / "*.tif")
#: only used to locate one DAPI file's naming; the per-image lookup is by stem
DAPI_PREFIXES = ["icell", "rotated"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="small", choices=("small", "b16"))
    ap.add_argument("-epochs", type=int, default=100)
    ap.add_argument("-lr", type=float, default=1e-5)
    ap.add_argument("-class_lr", type=float, default=1e-3)
    ap.add_argument("-n_classes", type=int, default=6)
    ap.add_argument("-beta", type=float, default=1e-4)
    ap.add_argument("-batch_size", type=int, default=512)
    ap.add_argument("-n_latent_dims", type=int, default=15)
    ap.add_argument("-device", type=int, default=0)
    ap.add_argument("-reduction", type=str, default="sum")
    ap.add_argument("--train-glob", default=TRAIN_GLOB,
                    help="narrow for a smoke run; labels still come from the full split")
    ap.add_argument("--val-glob", default=VAL_GLOB)
    ap.add_argument("--tag", default="cellidsleftout")
    ap.add_argument("--out", default=None, help="checkpoint path; default from paths.py")
    ap.add_argument("--val-batch-size", type=int, default=0,
                    help="validation batch size; 0 means use -batch_size. "
                         "The reference trainer's validation loop has NO "
                         "torch.no_grad(): every val batch builds a full autograd "
                         "graph that nothing backprops. Harmless for the conv "
                         "encoder at 512x2x100x100, fatal for ViT-B/16 at "
                         "512x3x224x224 -- it OOMs at 44 GB entering validation. "
                         "Reducing it is numerically free: the loss is "
                         "reduction='sum' accumulated across batches, so the "
                         "total does not depend on how the batches are cut.")
    ap.add_argument("--amp", action="store_true",
                    help="run the training call under bfloat16 autocast. Measured on "
                         "b16: 103 min/epoch without, 31 min/epoch with. bf16 needs no "
                         "GradScaler, parameters and gradients stay fp32, and "
                         "F.mse_loss is on autocast's fp32 list so the sum-reduced "
                         "reconstruction loss is NOT accumulated in bf16.")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute ViT activations in the backward pass instead of "
                         "storing them. b16 OOMs at the reference batch of 512 "
                         "without it; with it, peak is 37.9 GB of a 46 GB A40 and "
                         "the cost over batch 128 is only 10%% (33.8 vs 30.7 "
                         "min/epoch), which buys back the reference batch size.")
    ap.add_argument("--check-only", action="store_true",
                    help="verify every stem has a DAPI, build the model, then exit")
    A = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from utils import ImageDatasetWithDapisAndLabels          # noqa: E402
    from revbench.models import vit_cvae as V                 # noqa: E402

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    out = Path(A.out) if A.out else (paths.VIT_CKPT if A.variant == "small"
                                     else paths.VITB16_CKPT)

    # Labels always come from the FULL training split, so a narrowed --train-glob
    # cannot silently change the class set out from under the classifier heads.
    patterns = [str(Path(f).stem).split("_")[0] for f in glob(TRAIN_GLOB)]
    unique_patterns = sorted(set(patterns))
    log("patterns: %s" % unique_patterns)
    assert A.n_classes == len(unique_patterns), \
        "n_classes=%d but %d patterns" % (A.n_classes, len(unique_patterns))

    # Pre-flight: the dataset class imreads every pair and raises on a missing
    # file, which would waste hours of eager loading before failing.
    dapi_dir = Path(DAPI_GLOB).parent
    missing, n = [], 0
    for g in (A.train_glob, A.val_glob):
        for f in glob(g):
            n += 1
            if not (dapi_dir / (Path(f).stem + "_DAPI.tif")).is_file():
                missing.append(Path(f).stem)
    log("pre-flight: %d gene images | missing DAPI %d" % (n, len(missing)))
    assert not missing, "incomplete DAPI channel; not starting a long run: %s" % missing[:3]

    model = V.build_vit_cvae(A.variant, latent_dim=A.n_latent_dims)
    spec = V.describe(model)
    log("variant %s | encoder %s | %s params (%.2fx the 3,727,770 conv encoder)"
        % (A.variant, spec["encoder_kind"], format(spec["encoder_parameters"], ","),
           spec["encoder_parameters"] / 3727770.0))
    log("input: %s" % spec["input_spec"])
    log("augmentation: %s" % spec["augmentation"])
    if A.check_only:
        return

    common = dict(dapi_prefixes=DAPI_PREFIXES, n_classes=A.n_classes,
                  unique_patterns=unique_patterns, also_binary=True)
    log("loading train set (eager; ~14 GB for the full split)")
    train_data = ImageDatasetWithDapisAndLabels(A.train_glob, DAPI_GLOB, **common)
    log("loading val set (the 64 held-out cell ids)")
    val_data = ImageDatasetWithDapisAndLabels(A.val_glob, DAPI_GLOB, **common)
    log("train %s | val %s  (gene channel; DAPI is concatenated by __getitem__)"
        % (tuple(train_data.scaled_data.shape), tuple(val_data.scaled_data.shape)))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=A.batch_size,
                                               shuffle=True)
    val_bs = A.val_batch_size or A.batch_size
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_bs,
                                             shuffle=False)
    if val_bs != A.batch_size:
        log("validation batch %d (training batch %d) -- see --val-batch-size; "
            "val_loss is reduction='sum' so the value is unaffected" % (val_bs, A.batch_size))

    # Assert the CONTRACT, not the storage: `scaled_data` is the gene channel
    # alone and `scaled_dapi` sits beside it, so the two-channel tensor only
    # exists once __getitem__ concatenates them. Checking the attribute would
    # have asserted the wrong thing; check what the model is actually handed.
    _b, _lab, _bin = next(iter(train_loader))
    assert tuple(_b.shape)[1:] == (2, 100, 100), \
        "the loader yields %s; the encoder expects (B, 2, 100, 100)" % (tuple(_b.shape),)
    assert _lab.shape[1] == A.n_classes and _bin.shape[1] == 2, \
        "labels %s / binary %s do not match n_classes=%d" % (_lab.shape, _bin.shape, A.n_classes)
    log("batch contract OK: %s | labels %s | binary %s"
        % (tuple(_b.shape), tuple(_lab.shape), tuple(_bin.shape)))
    del _b, _lab, _bin

    model_name = ("%s_cvae_even_less_spots_vit-%s_2classifiers_coordinateparsed_"
                  "channel_dapi_and_labels_correctrotation_zoomed_fixedlocation_%s_"
                  "lr%.0e_classlr%.0e_nclasses%d_beta%.0e_%dld_%depochs"
                  % (datetime.now().strftime("%Y-%m-%d"), A.variant, A.tag, A.lr,
                     A.class_lr, A.n_classes, A.beta, A.n_latent_dims, A.epochs))
    log("model name: %s" % model_name)

    paths.VIT_RUNS.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(paths.VIT_RUNS / model_name))
    if torch.cuda.is_available():
        torch.cuda.set_device(A.device)
    model.to(A.device)

    args_dict = vars(A)
    args_dict["device"] = A.device
    t0 = time.time()
    # The reference trainer is used verbatim -- its five encoder passes per batch
    # and the gradient it accumulates across two of them are part of the recipe,
    # not an inefficiency to optimise away. Autocast wraps the call rather than
    # editing it, so the only difference is arithmetic precision.
    model.amp = bool(A.amp)
    if A.grad_checkpointing:
        if A.variant != "b16":
            raise SystemExit("--grad-checkpointing only applies to the b16 trunk")
        model.encoder_vit.vit.gradient_checkpointing_enable()
        log("gradient checkpointing: on (activations recomputed in backward)")
    log("autocast: %s" % ("bfloat16, confined to the ViT trunk (losses, decoder "
                          "and classifier heads stay fp32)" if A.amp
                          else "off (fp32 throughout, as the reference)"))
    losses = model.train_model_with_validation_with_2_classifiers(
        train_loader, val_loader, args_dict, writer=writer, with_dapi=True)
    log("training finished in %.1f h" % ((time.time() - t0) / 3600))

    out.parent.mkdir(parents=True, exist_ok=True)
    ck = {"cvae_state_dict": model.state_dict(),
          "binary_classifier_state_dict": model.binary_classifier.state_dict(),
          "classifier_state_dict": model.classifier.state_dict(),
          "epochs": A.epochs, "losses": losses, "lr": A.lr,
          "ld": A.n_latent_dims, "dataset": A.train_glob,
          "test_dataset": A.val_glob, "dapi_dataset": DAPI_GLOB,
          "beta": A.beta, "device": A.device, "n_classes": A.n_classes,
          "class_lr": A.class_lr, "unique_patterns": unique_patterns,
          "dapi_prefixes": DAPI_PREFIXES, "reduction": A.reduction,
          "batch_size": A.batch_size, "val_batch_size": val_bs,
          "seed": SEED, "model_name": model_name,
          "amp": "bfloat16 (ViT trunk only)" if A.amp else "off",
          "grad_checkpointing": bool(A.grad_checkpointing),
          "trunk_kw": {}, "hours": round((time.time() - t0) / 3600, 2)}
    ck.update(V.describe(model))          # variant, encoder_kind, input_spec, counts
    tmp = Path(str(out) + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(out)
    log("saved %s" % out)
    log("val_loss  first %.4g  last %.4g  best epoch %d"
        % (losses["val_loss"][0], losses["val_loss"][-1],
           int(np.argmin(losses["val_loss"])) + 1))
    log("=== VIT_TRAIN_%s_DONE ===" % A.variant.upper())


if __name__ == "__main__":
    main()
