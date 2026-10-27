#!/usr/bin/env python
# Provenance for additional_analyses/cell_boundary_vs_cytoplasmic.ipynb.
# Encodes the two simulated patterns with the published model (mode="mu", needs a GPU),
# stacks them with the cached latents of the 44,706 held-out simulated observations and
# writes combined.npz, eightclass_raw.csv and confusion_row_normalised.csv
# (copied to <data>/cell_boundary/). It also writes a binary probe and centroid distances,
# which the notebook does not use. Expects the original simulation tree; paths are the
# original ones.
"""Step 3: put the two new 2D arms into the shipped model's simulated latent space
and measure whether they separate.

The question is whether SubCellSpace can distinguish RNAs uniformly along the cell
boundary from RNAs uniform in the cytoplasm but not the nucleus. Both arms are
encoded with the *shipped* 2-channel model, mode="mu" -- the same model and mode
that produced the cached 44,706 held-out latents -- so all three sets live in one
space and nothing is confounded by the encoder.

Measured, not assumed:
  * celledge vs cytoplasm, a grouped binary probe against its permuted floor
  * an 8-class probe (6 existing + 2 new) with per-class F1 and a confusion matrix,
    so where the new classes get absorbed is a result rather than a guess
  * which existing pattern each new arm's centroid actually sits nearest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]   # the SubCellSpace repository
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
from revbench import cvae_encode as ce, controls
from revbench.metrics.probe import rf_probe

SIM = Path("/mnt/data/david/simulated_cells/with_dapi/even_less_spots")
CACHE = Path("/path/to/work/rotation/arm_a_latents.npz")   # shipped-model mu
NEW = {"celledge": Path("/path/to/work/celledge/sim"),
       "cytoplasm": Path("/path/to/work/cytoplasm/sim")}
OUT = Path("/path/to/work/celledge/step3"); OUT.mkdir(parents=True, exist_ok=True)
MODEL = REPO / "SubCellSpace"
DEVICE = "cuda:3"
SEED = 101


def encode_new():
    cache = OUT / "new_arm_latents.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print("loaded cached new-arm latents")
        return {k: (z[k + "_X"], list(z[k + "_stems"])) for k in NEW}
    model = ce.load_model(str(MODEL), device=DEVICE)
    assert ce.model_summary(model)["n_channels"] == 2
    out = {}
    for name, root in NEW.items():
        stems = sorted(p.name[:-4] for p in (root / "rotated").glob("*.tif"))
        gp = [root / "rotated" / f"{s}.tif" for s in stems]
        dp = [root / "rotated_dapi" / f"{s}_DAPI.tif" for s in stems]
        missing = [p for p in dp if not p.exists()]
        assert not missing, f"{name}: {len(missing)} DAPI missing"
        print(f"encoding {name}: {len(stems):,} observations", flush=True)
        out[name] = (ce.encode_image_paths(model, gp, dp, mode="mu", progress=True), stems)
    np.savez_compressed(cache, **{f"{k}_X": v[0] for k, v in out.items()},
                        **{f"{k}_stems": np.array(v[1]) for k, v in out.items()})
    return out


def main():
    z = np.load(CACHE, allow_pickle=True)
    X_old, stems_old = z["mu"], list(z["stems"])
    md_old = controls.simulated_metadata(stems_old)
    print(f"existing latent space: {X_old.shape[0]:,} observations, "
          f"{md_old['pattern'].nunique()} patterns")

    new = encode_new()
    Xs, pats, cells = [X_old], [md_old["pattern"].astype(str).values], \
                      [md_old["cell_id"].astype(str).values]
    for name, (X, stems) in new.items():
        md = controls.simulated_metadata(stems)
        Xs.append(X); pats.append(md["pattern"].astype(str).values)
        cells.append(md["cell_id"].astype(str).values)
        print(f"  + {name}: {X.shape[0]:,}")
    X = np.vstack(Xs).astype(np.float64)
    pattern = np.concatenate(pats); cell = np.concatenate(cells)
    print(f"combined: {X.shape}")
    np.savez_compressed(OUT / "combined.npz", X=X.astype(np.float32),
                        pattern=pattern, cell_id=cell)

    # --- 1. the direct question: celledge vs cytoplasm -----------------------------------
    m = np.isin(pattern, ["celledge", "cytoplasm"])
    rows = []
    for shuffle in (False, True):
        r = rf_probe(X[m], pattern[m], groups=cell[m], cv_scheme="stratified_group",
                     n_repeats=3, label_shuffle=shuffle, target="celledge_vs_cytoplasm")
        d = pd.DataFrame([x.as_dict() for x in r])
        for met in ("kappa", "accuracy"):
            v = d.loc[d["metric"] == met, "value"]
            rows.append({"probe": "celledge vs cytoplasm", "metric": met,
                         "label_shuffle": shuffle, "mean": v.mean(), "sd": v.std(),
                         "n": int(m.sum())})
    binary = pd.DataFrame(rows)
    binary.to_csv(OUT / "binary_probe.csv", index=False)
    print("\n=== celledge vs cytoplasm ===")
    print(binary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- 2. eight-class probe, per-class F1 and confusion ---------------------
    r = rf_probe(X, pattern, groups=cell, cv_scheme="stratified_group", n_repeats=3,
                 target="pattern8", confusion=True)
    d = pd.DataFrame([x.as_dict() for x in r])
    d.to_csv(OUT / "eightclass_raw.csv", index=False)
    k = d.loc[(d["metric"] == "kappa"), "value"]
    print(f"\n=== 8-class probe: kappa {k.mean():.4f} ± {k.std():.4f} ===")
    f1 = (d[(d["metric"] == "f1") & d["class_label"].ne("")]
          .groupby("class_label")["value"].mean().sort_values())
    print(f1.to_string(float_format=lambda v: f"{v:.4f}"))
    conf = d[d["metric"] == "confusion_count"]
    if len(conf):
        cm = conf.copy()
        cm[["true", "pred"]] = cm["class_label"].str.split("|", expand=True)
        piv = cm.pivot_table(index="true", columns="pred", values="value",
                             aggfunc="sum").fillna(0)
        piv = piv.div(piv.sum(1), axis=0)
        piv.to_csv(OUT / "confusion_row_normalised.csv")
        print("\n=== confusion (row-normalised) ===")
        print(piv.round(3).to_string())

    # --- 3. where do the new arms actually sit? ------------------------------
    cent = pd.DataFrame({p: X[pattern == p].mean(0) for p in np.unique(pattern)}).T
    D = pd.DataFrame(
        np.linalg.norm(cent.values[:, None, :] - cent.values[None, :, :], axis=2),
        index=cent.index, columns=cent.index)
    D.to_csv(OUT / "centroid_distances.csv")
    print("\n=== nearest existing pattern by centroid ===")
    existing = [p for p in cent.index if p not in ("celledge", "cytoplasm")]
    for p in ("celledge", "cytoplasm"):
        s = D.loc[p, existing].sort_values()
        print(f"  {p:<10} -> {s.index[0]:<13} {s.iloc[0]:.3f}   "
              f"(next: {s.index[1]} {s.iloc[1]:.3f})")
    print(f"  celledge <-> cytoplasm centroid distance: {D.loc['celledge','cytoplasm']:.3f}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
