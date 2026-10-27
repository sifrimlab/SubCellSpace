# Getting the data and each method's embedding off disk. Nothing here decides
# anything -- it just loads what config.py points at.

import json

import anndata as ad
import numpy as np
import pandas as pd

import config
import revbench as rb
from revbench.detection import explode_matched

# SubCellSpace's latents are stored in the datasets themselves, so it has no
# embedding files: on the simulated pair they sit in obsm["latent"], and on the
# real datasets they are the matrix the gene-cell explode produces.
STORED_LATENT = "subcellspace"

_cache = {}


def dataset_sizes():
    return json.loads((config.DATASETS / "scale.json").read_text())


def simulated_validation():
    # the held-out split, in the order every method is scored in
    order = pd.read_parquet(config.DATASETS / "validation_order.parquet")
    obs = rb.derive_metadata_columns(rb.simulated_metadata([str(k) for k in order["obs_key"]]))
    obs.index = [str(k) for k in order["obs_key"]]
    validation = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    validation.obs_names = list(obs.index)
    return validation


def simulated_pair():
    # patterned and shuffled halves of the simulated training set, used to fit
    # the classifier each method is scored with
    from revbench.detection.classifier import load_training_pair
    return load_training_pair()


def real_pair(dataset):
    # every gene-cell observation of a real dataset, and the same observations
    # with their transcripts randomly repositioned
    real, control = explode_matched(config.REAL_DATASETS[dataset])
    keys = (real.obs["original_obs_index"].astype(str) + "__"
            + real.obs["gene_name"].astype(str)).values
    real.obs_names, control.obs_names = keys, keys

    tested = [g for g, n in real.obs["gene_name"].value_counts().items()
              if n >= config.min_obs_per_gene]
    keep = real.obs["gene_name"].astype(str).isin(set(map(str, tested))).values
    return real[keep].copy(), control[keep].copy()


def shuffled_control(dataset):
    return rb.load_control(dataset, cache_dir=config.CONTROLS)


def ground_truth(dataset, tested_genes):
    from revbench.detection import load_apex, load_ileum
    if dataset == "apex_hek":
        return load_apex(gene_panel=tested_genes)
    return load_ileum(tested_genes=tested_genes)


def read_embedding(method, arm):
    path = config.embedding_file(method, arm)
    if str(path) not in _cache:
        matrix, keys, _ = rb.read_dump(path)
        _cache[str(path)] = (np.asarray(matrix, dtype=np.float32), [str(k) for k in keys])
    return _cache[str(path)]


def embedding_for(method, arm, keys):
    # rows of `arm`'s embedding, in the order of `keys`
    matrix, stored = read_embedding(method, arm)
    position = {k: i for i, k in enumerate(stored)}
    missing = [k for k in keys if k not in position]
    if missing:
        raise KeyError(f"{method}/{arm} is missing {len(missing)} of {len(keys)} observations")
    out = np.asarray(matrix[[position[k] for k in keys]], dtype=np.float64)
    return fill_gaps(out, method, arm)


def fill_gaps(matrix, method, arm):
    # Bento leaves a feature undefined for a few thin protrusion shapes. Filling
    # with the column median of its own patterned arm keeps those observations in
    # the comparison; an all-zero row would be read as "not measured" instead.
    if not np.isfinite(matrix).all():
        reference, _ = read_embedding(method, "patterned")
        with np.errstate(all="ignore"):
            median = np.nan_to_num(np.nanmedian(np.asarray(reference, np.float64), axis=0))
        matrix = matrix.copy()
        gaps = ~np.isfinite(matrix)
        matrix[gaps] = np.take(median, np.where(gaps)[1])
    return matrix


def attach_embeddings(dataset, real, control, methods=None):
    # put every method's embedding on the two AnnDatas, so the benchmarks can
    # ask for them by name
    keys = list(real.obs_names)
    for method in (methods or config.method_order):
        if method == "ella":
            continue
        if method == STORED_LATENT:
            real.obsm[method] = np.asarray(real.X, dtype=np.float64)
            control.obsm[method] = np.asarray(control.X, dtype=np.float64)
            continue
        real.obsm[method] = embedding_for(method, f"{dataset}_real", keys)
        control.obsm[method] = embedding_for(method, f"{dataset}_control", keys)
    return real, control


def simulated_embedding(method, arm, keys):
    # arm is "patterned", "shuffled" or "validation"
    if method == STORED_LATENT:
        patterned, shuffled = simulated_pair()
        source = {"patterned": patterned, "shuffled": shuffled}.get(arm)
        if source is not None:
            return np.asarray(source[list(keys)].obsm["latent"], dtype=np.float64)
        validation = ad.read_h5ad(config.SIMULATED_VALIDATION)
        validation = validation[validation.obs["train_or_val"].astype(str) == "val"].copy()
        from pathlib import Path
        validation.obs_names = [Path(str(p)).stem for p in validation.obs["original_image_paths"]]
        return np.asarray(validation[list(keys)].obsm["latent"], dtype=np.float64)
    return embedding_for(method, arm, keys)


def shared_observations(dataset, canonical, methods=None):
    # the observations every method can serve, in the canonical order
    arm = "validation" if dataset == "simulated" else f"{dataset}_real"
    keep = list(canonical)
    for method in (methods or config.method_order):
        if method in ("ella", STORED_LATENT):
            continue
        _, stored = read_embedding(method, arm)
        have = set(stored)
        keep = [k for k in keep if k in have]
    return keep


def derived_thresholds(dataset):
    # each method's distance cut-off, derived on simulated data alone. Different
    # classifiers put the distance on different scales, so the cut is recomputed
    # per method rather than shared.
    from revbench.detection import calibrate
    derivations = calibrate.load_all(config.THRESHOLDS)
    from revbench.config import dataset_threshold
    published = {"subcellspace": dataset_threshold(dataset)}
    return calibrate.resolve(derivations, dataset, published=published)


def ella_gene_tables(dataset):
    from revbench import ella_io
    return (ella_io.read_gene_table(config.gene_table("ella", dataset, "real")),
            ella_io.read_gene_table(config.gene_table("ella", dataset, "control")))
