#!/usr/bin/env python
# Provenance only: this script produced cell_mask/epoch_sweep_raw.csv (pattern-probe kappa per checkpoint on the held-out simulations). Needs the checkpoints above, the original simulation tree and a GPU.
# Paths are those of the original run; it is not meant to run unchanged.
"""Encode the 100- and 300-epoch checkpoints and rescore the simulated benchmarks.

Five arms, so both contrasts are clean:
  shipped_2ch   the published SubCellSpace checkpoint -- a GATE: if re-encoding it
                does not reproduce silhouette 0.1219 / pattern kappa 0.7756, the
                setup is wrong and nothing else here means anything.
  run2ch_e100 / run2ch_e300     same run, 100 vs 300 epochs
  run3ch_e100 / run3ch_e300     same run, 100 vs 300 epochs

Comparing e100 with e300 *within a run* isolates the epoch effect from the
run-to-run variation seen between the original 100-epoch runs and these.
Protocol copied from `reduce/ablations.py::run_simulated` so the numbers land on
the same footing as the published ablation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

REPO = Path(__file__).resolve().parents[3]   # the SubCellSpace repository
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
import revbench as rb
from revbench import cvae_encode as ce, controls
from revbench.config import derive_metadata_columns, SEED

SIM = Path("/mnt/data/david/simulated_cells/with_dapi/even_less_spots")
TREE = SIM / "coordinate_parsed_zoomed_again"
DAPI = SIM / "rotated_dapi_zoomed_again"
CELL = SIM / "coordinate_parsed_with_coords" / "cell_mask_zoomed"
TEST = TREE / "unbalanced" / "cellids_left_out" / "test"
EP = Path("/path/to/work/epochs")
OUT = EP / "metrics"
DEVICE = "cuda:1"


def ckpt(arm, epoch):
    d = EP / arm
    pat = f"*300epochs__epoch{epoch:03d}" if epoch != 300 else "*300epochs"
    hits = sorted(p for p in d.glob(pat) if p.is_file())
    if not hits:
        raise FileNotFoundError(f"no checkpoint for {arm} epoch {epoch}")
    return hits[0]


ARMS = {
    "shipped_2ch": (REPO / "SubCellSpace", 2),
    "run2ch_e100": (ckpt("2ch", 100), 2),
    "run2ch_e300": (ckpt("2ch", 300), 2),
    "run3ch_e100": (ckpt("3ch", 100), 3),
    "run3ch_e300": (ckpt("3ch", 300), 3),
}


def encode_all(stems):
    cache = OUT / "latents.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["stems"]) == stems and set(z.files) >= set(ARMS) | {"stems"}:
            print("loaded cached latents")
            return {a: z[a] for a in ARMS}
    gp = [TREE / f"{s}.tif" for s in stems]
    dp = [DAPI / f"{s}_DAPI.tif" for s in stems]
    cp = [CELL / f"{s}_cell-mask.tif" for s in stems]
    lat = {}
    for name, (path, nch) in ARMS.items():
        m = ce.load_model(str(path), device=DEVICE)
        s = ce.model_summary(m)
        assert s["n_channels"] == nch, f"{name}: checkpoint has {s['n_channels']} channels"
        print(f"encoding {name}  ({nch}ch, ld {s['latent_dim']}) ...", flush=True)
        lat[name] = ce.encode_image_paths(
            m, gp, dp, cell_paths=(cp if nch == 3 else None), mode="mu", progress=True)
        del m
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, stems=np.array(stems), **lat)
    return lat


def variance_share(X, labels):
    """Multivariate eta^2 -- copied from reduce/ablations.py so it is the same number."""
    X = np.asarray(X, float); X = X - X.mean(0)
    total = float((X ** 2).sum())
    if total == 0.0:
        return float("nan")
    labels = np.asarray(labels)
    within = sum(float(((X[labels == g] - X[labels == g].mean(0)) ** 2).sum())
                 for g in np.unique(labels))
    return 1.0 - within / total


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name[:-4] for p in TEST.glob("*.tif"))
    print(f"{len(stems):,} held-out simulated observations")
    lat = encode_all(stems)

    md = controls.simulated_metadata(stems)
    md = derive_metadata_columns(md.set_index("obs_key"))
    sim = ad.AnnData(X=np.zeros((len(stems), 1), dtype=np.float32), obs=md)
    sim.obs_names = stems
    for name, X in lat.items():
        sim.obsm[name] = np.asarray(X, dtype=np.float64)
    reps = [rb.from_adata(sim, "obsm:" + n, name=n) for n in ARMS]

    df = rb.run_suite(
        sim, reps, dataset="simulated", subset="all",
        benchmarks=["silhouette", "silhouette_grouped", "pattern_probe",
                    "pattern_probe_floor", "negative_controls"],
        n_repeats=3, seed=SEED, results_dir=OUT,
        run_label="epoch_sweep", scale_label="full_%d_rows" % len(stems),
        n_cells=len(stems),
        extra_params={"stage": "epoch_sweep", "track": "sanity"})
    df.to_csv(OUT / "epoch_sweep_raw.csv", index=False)

    rows = []
    for name, X in lat.items():
        rows.append({"arm": name,
                     "variance_from_pattern": round(variance_share(X, md["pattern"].values), 4),
                     "variance_from_cell_id": round(variance_share(X, md["cell_id"].values), 4)})
    var = pd.DataFrame(rows).set_index("arm")
    var.to_csv(OUT / "epoch_sweep_variance.csv")
    print("\n=== variance share (eta^2) ===")
    print(var.to_string())
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
