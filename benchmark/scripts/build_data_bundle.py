#!/usr/bin/env python
# Collects everything the benchmark reads into one directory with a flat, stable
# layout, so config.py needs a single DATA_ROOT and the bundle can be uploaded
# as-is. Symlinks by default (instant, for local use); --copy for the real thing.
#
#   python scripts/build_data_bundle.py --out /path/to/benchmark_data [--copy]

import argparse
import os
import shutil
import sys
from pathlib import Path

OUT = Path("/mnt/data/david/benchmarks/reduced_benchmarks")
ARTIFACTS = OUT / "artifacts"
CACHE = Path("/mnt/data/david/benchmarks/cache")
ELLA = Path("/mnt/data/david/benchmarks/ELLA")
PUBLIC = Path(__file__).resolve().parents[3]
SUBCELL_DATA = Path("/mnt/data/david/subcellspace_data")
SIM = Path("/mnt/data/david/simulated_cells/with_dapi")

# destination -> source
FILES = {
    "datasets/apex_hek.h5ad":
        SUBCELL_DATA / "apex_validation_data/adata_subcell_including_randomized_embeddings.h5ad",
    "datasets/ileum.h5ad":
        SUBCELL_DATA / "small_intestine_data/adata_subcell_including_randomized_embeddings.h5ad",
    # the _with_umap copy is the one already deposited, and is the same object
    # plus X_pca/X_umap -- verified identical obs_names, train_or_val and latents
    "datasets/simulated_validation.h5ad":
        SIM / "even_less_spots/adata_unbalanced_traintestsplit_with_umap.h5ad",
    "datasets/simulated_patterned.h5ad":
        SUBCELL_DATA / "simulated_data/adata_all_proportions_only_test.h5ad",
    "datasets/simulated_shuffled.h5ad":
        SUBCELL_DATA / "simulated_data/adata_all_proportions_only_test_randomized.h5ad",
    "datasets/validation_order.parquet": ARTIFACTS / "simval_obs.parquet",
    "datasets/scale.json": ARTIFACTS / "scale.json",

    "gene_tables/ella/ileum_real.csv": ELLA / "dumps/ella__ileum_real_genes.csv",
    "gene_tables/ella/ileum_control.csv": ELLA / "dumps/ella__ileum_control_genes.csv",
    "gene_tables/ella/apex_hek_real.csv": ELLA / "full/genes/ella__apex_full_real_genes.csv",
    "gene_tables/ella/apex_hek_control.csv": ELLA / "full/genes/ella__apex_full_control_genes.csv",

    "models/SubCellSpace": PUBLIC / "SubCellSpace",
    "models/rf_pattern_nonpattern.pkl": PUBLIC / "classifiers/rf_pattern_nonpattern.pkl",
    "models/multiclass_rf.pkl": PUBLIC / "classifiers/multiclass_rf.pkl",
}

# embeddings/<method>/<arm>.npz  ->  wherever that arm currently lives
ARMS = ["patterned", "shuffled", "validation",
        "ileum_real", "ileum_control", "apex_hek_real", "apex_hek_control"]

EMBEDDINGS = {
    "pointnet_vae_64": [
        ARTIFACTS / "pointnet/atrain_real.npz", ARTIFACTS / "pointnet/atrain_ctrl.npz",
        CACHE / "pn__simval_n44706.npz",
        CACHE / "pn__ileum_pairs_real.npz", CACHE / "pn__ileum_pairs_ctrl.npz",
        ARTIFACTS / "pointnet/apex_real.npz", ARTIFACTS / "pointnet/apex_ctrl.npz"],
    "cnn_baseline": [ARTIFACTS / "cnn" / f"{n}.npz" for n in
        ["atrain_real", "atrain_ctrl", "simval",
         "ileum_real", "ileum_ctrl", "apex_hek_real", "apex_hek_ctrl"]],
    "bento_features": [
        ARTIFACTS / "bento/simulated/atrain_real.npz", ARTIFACTS / "bento/simulated/atrain_ctrl.npz",
        ARTIFACTS / "bento/simulated/simval.npz",
        ARTIFACTS / "bento/ileum/ileum_real.npz", ARTIFACTS / "bento/ileum/ileum_ctrl.npz",
        ARTIFACTS / "bento/apex_hek/apex_hek_real.npz", ARTIFACTS / "bento/apex_hek/apex_hek_ctrl.npz"],
    "image_transformer_small": [ARTIFACTS / "vit/small" / f"{n}.npz" for n in
        ["atrain_real", "atrain_ctrl", "simval",
         "ileum_real", "ileum_ctrl", "apex_hek_real", "apex_hek_ctrl"]],
    "image_transformer_large": [ARTIFACTS / "vit/b16" / f"{n}.npz" for n in
        ["atrain_real", "atrain_ctrl", "simval",
         "ileum_real", "ileum_ctrl", "apex_hek_real", "apex_hek_ctrl"]],
}

# The shuffled molecule sets each dataset is scored against. Named
# control_<dataset> because that is how the library looks them up, and the name
# recorded inside the sidecar is rewritten to match.
CONTROLS = {
    "ileum": CACHE / "control_ileum.parquet",
    "apex_hek": CACHE / "control_apex_demo.parquet",
    "patterned": ARTIFACTS / "controls/control_reduce_atrain_real.parquet",
    "shuffled": ARTIFACTS / "controls/control_reduce_atrain_ctrl.parquet",
}

# each method's distance cut, derived on simulated data only
THRESHOLD_RENAME = {"subcellspace_vit": "image_transformer_small",
                    "subcellspace_vit_b16": "image_transformer_large"}


def place(src, dest, copy):
    src, dest = Path(src), Path(dest)
    if not src.exists():
        return f"MISSING {src}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(src, dest)
    else:
        dest.symlink_to(src.resolve())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    ap.add_argument("--upload-only", action="store_true",
                    help="leave out the files already in the SubCellSpace data "
                         "record, so what is written is exactly what still needs "
                         "uploading. The result is NOT runnable on its own.")
    args = ap.parse_args()
    root = Path(args.out)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    skip = set(config.ALREADY_DEPOSITED) if args.upload_only else set()

    problems = []
    for dest, src in FILES.items():
        if dest in skip:
            continue
        problems.append(place(src, root / dest, args.copy))

    for method, sources in EMBEDDINGS.items():
        for arm, src in zip(ARMS, sources):
            problems.append(place(src, root / f"embeddings/{method}/{arm}.npz", args.copy))
            meta = Path(str(src).replace(".npz", ".meta.json"))
            if meta.exists():
                place(meta, root / f"embeddings/{method}/{arm}.meta.json", args.copy)

    import json
    for name, src in CONTROLS.items():
        problems.append(place(src, root / f"controls/control_{name}.parquet", args.copy))
        sidecar = src.with_suffix(".json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            meta["name"] = name
            dest = root / f"controls/control_{name}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                dest.unlink()
            dest.write_text(json.dumps(meta, indent=1))

    # thresholds are always rewritten rather than linked: the method name is
    # recorded inside the file and has to match the public one
    import json
    for src in sorted((ARTIFACTS / "thresholds").glob("*.json")):
        name = THRESHOLD_RENAME.get(src.stem, src.stem)
        derivation = json.loads(src.read_text())
        derivation["method"] = name
        dest = root / f"thresholds/{name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            dest.unlink()
        dest.write_text(json.dumps(derivation, indent=1))

    problems = [p for p in problems if p]
    n = sum(1 for _ in root.rglob("*") if _.is_file() or _.is_symlink())
    kind = "upload set" if args.upload_only else "bundle"
    print(f"{kind} at {root}: {n} entries")
    if skip:
        print(f"left out ({len(skip)} already deposited):")
        for dest in sorted(skip):
            print(f"  {dest}  ->  {config.ALREADY_DEPOSITED[dest]}")
    if problems:
        print(f"\n{len(problems)} problems:")
        for p in problems:
            print(" ", p)
    else:
        print("every source found")


if __name__ == "__main__":
    main()
