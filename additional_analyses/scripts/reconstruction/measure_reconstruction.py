#!/usr/bin/env python
# Provenance: produces the cached inputs of reconstruction_error.ipynb --
# recon_simulated.csv (stage recon), pca_ladder.csv + pca_provenance.json (stage pca),
# epoch_sweep_e500.csv (stage sweep, over the checkpoints of the 500-epoch run made by
# train_long_2ch_3ch.py) and pair_separability.csv (stage pairs).
# Expects the original simulation tree and a GPU; paths are those of the original run.
"""How well does SubCellSpace reconstruct, and does that measure anything useful?

A quantitative reconstruction error for the CVAE. In the reconstructions shown in the
paper, `extranuclear` and `cell-edge` look alike, as do `pericellular` and `perinuclear`.

A bare MSE answers nothing, so everything here is reported against references that give it
a scale: the trivial prediction, the best *linear* code of the same width (PCA), and the
ceiling imposed by the decoder itself. Two properties of the architecture set that scale:

  * the decoder's native output is 26x26 = 676 units, nearest-upsampled to 100x100 by
    `nn.Upsample(size=(100,100))` (CVAE.py:86; the intermediate upsamples at lines 81 and 84
    are commented out). It cannot render individual transcripts.
  * the training loss is gene-channel only (CVAE.py:180) while the validation loss covers
    all channels (CVAE.py:290). A single error quoted across channels is dominated by a
    channel the model never optimises, so everything below is per channel.

Convention check, reproduced before anything is reported: the stored `losses["val_loss"]` is
an epoch total under `reduction="sum"`. Dividing by images x channels x pixels gives 0.276
per pixel at epoch 100, and gene 0.00815 + DAPI 0.54204 averages to exactly that.

Writes to /path/to/work/reconstruction/work/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from skimage.transform import resize

REPO = Path(__file__).resolve().parents[3]   # the SubCellSpace repository
import sys
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "benchmark"))
from revbench import cvae_encode as ce, controls                       # noqa: E402

SIM = Path("/mnt/data/david/simulated_cells/with_dapi/even_less_spots")
TREE = SIM / "coordinate_parsed_zoomed_again"
DAPI = SIM / "rotated_dapi_zoomed_again"
TEST = TREE / "unbalanced" / "cellids_left_out" / "test"
MODEL = REPO / "SubCellSpace"
OUT = Path("/path/to/work/reconstruction/work")
DECODER_NATIVE = 26
SEED = 101


def minmax(a):
    """Per-image min-max to [0,1] -- exactly what the model does to its own input."""
    a = a.astype(np.float32)
    r = a.max() - a.min()
    return (a - a.min()) / r if r > 0 else np.zeros_like(a)


def load_pair(stems):
    g = np.stack([minmax(imread(str(TREE / f"{s}.tif"))) for s in stems])
    d = np.stack([minmax(imread(str(DAPI / f"{s}_DAPI.tif"))) for s in stems])
    return g, d


def ceiling_26(x):
    """Best any 26x26 bottleneck could do: area-average down, nearest back up.

    Nearest on the way up because that is literally what the decoder's final layer is.
    """
    return resize(resize(x, (DECODER_NATIVE, DECODER_NATIVE), order=1, anti_aliasing=True),
                  x.shape, order=0)


def stage_recon(batch=512, device="cuda:0", limit=None):
    stems = sorted(p.name[:-4] for p in TEST.glob("*.tif"))
    if limit:
        stems = list(np.random.RandomState(SEED).choice(stems, limit, replace=False))
    m = ce.load_model(str(MODEL), device=device)
    s = ce.model_summary(m)
    assert s["latent_dim"] == 15 and s["n_channels"] == 2, s
    cvae = m["cvae"]
    rows = []
    for i in range(0, len(stems), batch):
        b = stems[i:i + batch]
        G, D = load_pair(b)
        X = torch.from_numpy(np.stack([G, D], 1)).to(device)
        with torch.no_grad():
            _, mu, _ = cvae.embed(X)
            R = cvae.decode(mu).cpu().numpy()
        for j, st in enumerate(b):
            g, d, rg, rd = G[j], D[j], R[j, 0], R[j, 1]
            rows.append(dict(obs_key=st,
                             mse_gene=float(((g - rg) ** 2).mean()),
                             mse_dapi=float(((d - rd) ** 2).mean()),
                             var_gene=float(g.var()), var_dapi=float(d.var()),
                             mse_ceiling=float(((g - ceiling_26(g)) ** 2).mean())))
        if (i // batch) % 10 == 0:
            print(f"  {i + len(b):,}/{len(stems):,}", flush=True)
    d = pd.DataFrame(rows)
    md = controls.simulated_metadata(d.obs_key.tolist())
    d["pattern"] = md["pattern"].values
    d["cell_id"] = md["cell_id"].astype(str).values
    d["n_spots"] = md["n_spots"].values
    d["r2_gene"] = 1 - d.mse_gene / d.var_gene
    d["r2_ceiling"] = 1 - d.mse_ceiling / d.var_gene
    d["r2_dapi"] = 1 - d.mse_dapi / d.var_dapi
    d.to_csv(OUT / "recon_simulated.csv", index=False)
    print(f"\n{len(d):,} held-out images")
    print(f"  gene  MSE mean {d.mse_gene.mean():.6f}  R2 median {d.r2_gene.median():+.4f}")
    print(f"  DAPI  MSE mean {d.mse_dapi.mean():.6f}  R2 median {d.r2_dapi.median():+.4f}")
    print(f"  26x26 ceiling   R2 median {d.r2_ceiling.median():+.4f}")
    print(f"\n  both-channel mean MSE {(d.mse_gene.mean() + d.mse_dapi.mean()) / 2:.6f}"
          f"   <- compare with the stored val_loss, 0.276/pixel")
    print("\nper pattern:")
    print(d.groupby("pattern")[["mse_gene", "r2_gene", "r2_ceiling"]]
          .median().to_string(float_format=lambda v: f"{v:.5f}"))
    return d


def stage_pca(n_fit=8000, n_eval=8000, widths=(5, 15, 30, 50, 100)):
    """The best LINEAR code of a given width, as the reference the CVAE is judged against.

    Fitted and evaluated on disjoint held-out images, neither of which the CVAE trained on.
    """
    from sklearn.decomposition import PCA
    stems = sorted(p.name[:-4] for p in TEST.glob("*.tif"))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(stems))
    fit = [stems[i] for i in perm[:n_fit]]
    ev = [stems[i] for i in perm[n_fit:n_fit + n_eval]]
    FIT = load_pair(fit)[0].reshape(len(fit), -1)
    EV = load_pair(ev)[0].reshape(len(ev), -1)
    var = EV.reshape(len(ev), 100, 100).var(axis=(1, 2))
    rows = []
    for n in widths:
        p = PCA(n_components=n, random_state=SEED).fit(FIT)
        rec = p.inverse_transform(p.transform(EV))
        mse = ((EV - rec) ** 2).mean(1)
        rows.append(dict(code="PCA-%d" % n, n_dims=n, mse_mean=float(mse.mean()),
                         r2_median=float(np.median(1 - mse / var))))
        print(f"  PCA-{n:<4d} MSE mean {mse.mean():.6f}  R2 median {np.median(1 - mse / var):+.4f}",
              flush=True)
    gm = FIT.mean(0)
    mse = ((EV - gm) ** 2).mean(1)
    rows.append(dict(code="global mean image", n_dims=0, mse_mean=float(mse.mean()),
                     r2_median=float(np.median(1 - mse / var))))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "pca_ladder.csv", index=False)
    json.dump(dict(n_fit=len(fit), n_eval=len(ev), seed=SEED),
              open(OUT / "pca_provenance.json", "w"), indent=2)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    return d


E500 = Path("/path/to/work/epochs/2ch_e500")


def sweep_subsample(n=8000):
    """One fixed, pattern-stratified index list, shared by every checkpoint.

    Fixed because the point of the sweep is how the metrics move with training; resampling
    per checkpoint would put sampling noise on the same axis.
    """
    stems = sorted(p.name[:-4] for p in TEST.glob("*.tif"))
    md = controls.simulated_metadata(stems)
    pat = md["pattern"].astype(str).values
    rng = np.random.RandomState(SEED)
    take = []
    per = n // len(np.unique(pat))
    for u in np.unique(pat):
        idx = np.flatnonzero(pat == u)
        take.append(rng.choice(idx, min(per, len(idx)), replace=False))
    take = np.sort(np.concatenate(take))
    return [stems[i] for i in take], md.iloc[take]


def stage_sweep(device="cuda:0", batch=512, every=10):
    """Reconstruction loss and representation quality against training epoch.

    The 500-epoch run is 2-channel only -- `3ch_e500/` was never populated -- so this axis
    is the 2-channel arm. Loss comes free out of each checkpoint's own `losses` history;
    the representation metrics need an encode per checkpoint.
    """
    from revbench.metrics.silhouette import silhouette_rows
    from revbench.metrics.probe import rf_probe
    stems, md = sweep_subsample()
    pattern = md["pattern"].astype(str).values
    cell = md["cell_id"].astype(str).values
    print(f"fixed evaluation subsample: {len(stems):,} images, "
          f"{len(np.unique(pattern))} patterns, {len(np.unique(cell))} cells", flush=True)
    gp = [str(TREE / f"{s}.tif") for s in stems]
    dp = [str(DAPI / f"{s}_DAPI.tif") for s in stems]

    cks = sorted(p for p in E500.glob("*__epoch*") if p.is_file())
    cks = [p for p in cks if int(str(p)[-3:]) % every == 0]
    print(f"{len(cks)} checkpoints", flush=True)
    rows = []
    for p in cks:
        ep = int(str(p)[-3:])
        m = ce.load_model(str(p), device=device)
        X = ce.encode_image_paths(m, gp, dp, mode="mu", batch_size=batch).astype(np.float64)
        L = m["losses"]
        sil = [r for r in silhouette_rows(X, pattern, one_vs_rest=False,
                                          sample_size=len(X), n_repeats=1, seed=SEED)
               if r.metric == "silhouette"]
        pr = [r for r in rf_probe(X, pattern, groups=cell, n_splits=3, n_repeats=1,
                                  seed=SEED, max_depth=32)
              if r.metric == "kappa"]
        rows.append(dict(epoch=ep,
                         silhouette=float(np.mean([r.value for r in sil])),
                         kappa=float(np.mean([r.value for r in pr])),
                         train_recon_per_px=L["reconstruction_loss"][ep - 1] / (175981 * 1 * 10000),
                         heldout_val_per_px=L["val_loss"][ep - 1] / (44706 * 2 * 10000)))
        print(f"  epoch {ep:3d}  sil {rows[-1]['silhouette']:+.4f}  kappa {rows[-1]['kappa']:.4f}"
              f"  recon {rows[-1]['train_recon_per_px']:.6f}"
              f"  val {rows[-1]['heldout_val_per_px']:.6f}", flush=True)
        del m
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "epoch_sweep_e500.csv", index=False)
    print("\n" + d.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return d


def stage_pairs(device="cuda:0", batch=512):
    """Where does the information that separates two patterns actually live?

    Similar-looking reconstructions could be read as a failure. The question that
    settles it is whether a pair is separable in the space the model *uses* -- the latent --
    even where it is not separable in the picture the decoder draws. Same classifier, same
    folds grouped by cell, three representations:

      input_26    the input at the decoder's own working resolution
      recon_26    the reconstruction at its native resolution, before the nearest blow-up
      latent      the 15 numbers the rest of the method consumes
    """
    from revbench.metrics.probe import rf_probe
    import itertools
    stems, md = sweep_subsample()
    pattern = md["pattern"].astype(str).values
    cell = md["cell_id"].astype(str).values
    m = ce.load_model(str(MODEL), device=device)
    cvae = m["cvae"]
    I26, R26, Z = [], [], []
    for i in range(0, len(stems), batch):
        b = stems[i:i + batch]
        G, D = load_pair(b)
        X = torch.from_numpy(np.stack([G, D], 1)).to(device)
        with torch.no_grad():
            _, mu, _ = cvae.embed(X)
            R = cvae.decode(mu).cpu().numpy()
        for j in range(len(b)):
            I26.append(resize(G[j], (DECODER_NATIVE, DECODER_NATIVE), order=1,
                              anti_aliasing=True).ravel())
            # the reconstruction is a 26x26 image held in a 100x100 array; take it back
            R26.append(resize(R[j, 0], (DECODER_NATIVE, DECODER_NATIVE), order=0).ravel())
        Z.append(mu.cpu().numpy())
        if (i // batch) % 5 == 0:
            print(f"  {i + len(b):,}/{len(stems):,}", flush=True)
    SP = {"input_26": np.asarray(I26, float), "recon_26": np.asarray(R26, float),
          "latent": np.vstack(Z).astype(float)}
    print({k: v.shape for k, v in SP.items()})
    rows = []
    for a, b in itertools.combinations(sorted(np.unique(pattern)), 2):
        sel = np.isin(pattern, [a, b])
        for name, M in SP.items():
            pr = [r.value for r in rf_probe(M[sel], pattern[sel], groups=cell[sel],
                                            n_splits=3, n_repeats=1, seed=SEED, max_depth=32)
                  if r.metric == "kappa"]
            rows.append(dict(pair=f"{a} vs {b}", a=a, b=b, space=name,
                             n=int(sel.sum()), kappa=float(np.mean(pr))))
        print(f"  {a} vs {b}: " + "  ".join(
            f"{r['space']} {r['kappa']:.3f}" for r in rows[-3:]), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "pair_separability.csv", index=False)
    w = d.pivot(index="pair", columns="space", values="kappa")
    print("\n" + w.to_string(float_format=lambda v: f"{v:.4f}"))
    return d


STAGES = {"recon": stage_recon, "pca": stage_pca, "sweep": stage_sweep, "pairs": stage_pairs}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["recon", "pca"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for st in a.stages:
        print(f"=== {st} ===", flush=True)
        kw = {"recon": {"device": a.device, "limit": a.limit},
              "sweep": {"device": a.device, "every": a.every},
              "pairs": {"device": a.device}}.get(st, {})
        STAGES[st](**kw)
