#!/usr/bin/env python
# Provenance only: this script produced cell_mask/matched_detection_raw.csv (genes called at the published thresholds on Xenium HEK293T and MERFISH small intestine). Needs the embeddings from encode_matched_reduce.py.
# Paths are those of the original run; it is not meant to run unchanged.
"""Detection sweep for the matched 300-epoch 2ch / 3ch pair, published protocol.

The published cell-mask comparison scored the *shipped* 2-channel checkpoint against
a 3-channel one trained by a different script on a different date. These two are
matched -- same script, same day, same seed, differing only in the channel count --
and were encoded through an identical path, so any surviving gap is attributable to
the channel rather than to the draw.

`subcellspace` (the published reference) is scored alongside as a GATE: it must
reproduce APEX ROC-AUC 0.7014 and ileum 0.8371, or the scoring path is wrong and the
other two arms mean nothing.

Nothing in the benchmark track is edited. The two arms are registered in-process and
unregistered afterwards, and `paths.RESULTS` is rebound so results land in a
separate store rather than the benchmark one.
"""
import sys
import time
from pathlib import Path

import pandas as pd

REDUCE = Path("/path/to/reduce")   # helper modules of the original run, not part of this repository
sys.path.insert(0, str(REDUCE))
import paths                                                    # noqa: E402
paths.add_src_to_path()
paths.ensure()
sys.path.insert(0, str(paths.PUBLIC))

import ablations, methods                                       # noqa: E402
import revbench as rb                                           # noqa: E402
from revbench.detection import load_apex, load_ileum            # noqa: E402

EP = Path("/path/to/work/epochs")
OUT = EP / "detection"; OUT.mkdir(parents=True, exist_ok=True)
paths.RESULTS = OUT                      # keep results out of the benchmark store
RUN_LABEL = "matched_epoch300"

# (checkpoint dir, dump dir, label) -- the checkpoints and the encoded dumps live
# in different directories, which is exactly the mistake this tuple prevents.
ARMS = {
    "subcellspace_2ch_e300": ("2ch", "2ch_e300", "2-channel, 300 epochs"),
    "subcellspace_3ch_e300": ("3ch", "3ch_e300", "+ cell mask (3ch), 300 epochs"),
}


def dumps_for(dump_dir):
    d = EP / "encoded" / dump_dir
    return {k: d / f"{k}.npz" for k in
            ("atrain_real", "atrain_ctrl", "simval",
             "ileum_real", "ileum_ctrl", "apex_hek_real", "apex_hek_ctrl")}


def make_method(name, ckpt_dir, dump_dir, label):
    D = dumps_for(dump_dir)
    missing = [k for k, p in D.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"{name}: missing dumps {missing} under {EP/'encoded'/dump_dir}")
    ckpt = sorted((EP / ckpt_dir).glob("*300epochs"))[0]

    def train_pair():
        return (methods._cvae3_sim_adata(D["atrain_real"], "pattern"),
                methods._cvae3_sim_adata(D["atrain_ctrl"], "random"), None)

    def pair_matrix(dataset, keys, real, ctrl):
        return (methods._rows_for(D[f"{dataset}_real"], keys),
                methods._rows_for(D[f"{dataset}_ctrl"], keys))

    return methods.Method(
        name=name, kind="embedding", role="ablation", label=label,
        supports={"simulated": True, "ileum": True, "apex_hek": True},
        checkpoint=ckpt,
        train_pair=train_pair,
        val_keys=lambda: methods._dump(D["simval"])[1],
        val_matrix=lambda keys: methods._rows_for(D["simval"], keys),
        pair_keys=lambda dataset: methods._dump(D[f"{dataset}_real"])[1],
        pair_matrix=pair_matrix,
        notes=("Matched 300-epoch retrain; see scripts/train_long_2ch_3ch.py. "
               "Classifier refit from this latent's own artifact A, like every "
               "other non-shipping method."))


def main():
    registered = []
    try:
        for name, (ckpt_dir, dump_dir, label) in ARMS.items():
            methods.register(make_method(name, ckpt_dir, dump_dir, label))
            registered.append(name)
        compared = [methods.REGISTRY["subcellspace"]] + \
                   [methods.REGISTRY[n] for n in ARMS]
        print("comparing: " + ", ".join(m.label for m in compared))
        print("results  : %s  (run_label=%r)\n" % (paths.RESULTS, RUN_LABEL))

        SCALE = __import__("json").loads((paths.ARTIFACTS / "scale.json").read_text())
        out = {}
        for ds, h5ad, loader, ctrl in (
                ("ileum", paths.ILEUM, lambda t: load_ileum(tested_genes=t), "ileum"),
                ("apex_hek", paths.APEX, lambda t: load_apex(gene_panel=t), "apex_demo")):
            t0 = time.time()
            S = SCALE[ds]
            res = ablations.run_detection(
                ds, compared, h5ad, loader,
                methods.scale_label(S["n_usable"], S["n_usable"]), ctrl,
                run_label=RUN_LABEL)
            n_err = int((res.metric == "error").sum())
            print("%s: %d rows, %d errors | %.1f min\n"
                  % (ds, len(res), n_err, (time.time() - t0) / 60))
            assert n_err == 0, f"{ds} produced errors"
            out[ds] = res
    finally:
        for n in registered:
            methods.unregister(n)

    df = pd.concat(out.values(), ignore_index=True)
    df.to_csv(OUT / "matched_detection_raw.csv", index=False)
    key = df[(df.benchmark == "detection_sweep") & (df.metric == "roc_auc")]
    piv = key.groupby(["dataset", "method"])["value"].mean().unstack("method")
    piv.to_csv(OUT / "matched_detection_roc_auc.csv")
    print("=== detection ROC-AUC ===")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n=== GATE: the published reference ===")
    for ds, want in (("apex_hek", 0.7014), ("ileum", 0.8371)):
        got = piv.loc[ds, "subcellspace"]
        print(f"  {ds:<9} {got:.4f}   published {want:.4f}   "
              f"{'OK' if abs(got - want) < 0.002 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
