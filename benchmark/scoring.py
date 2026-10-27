# Running the benchmarks. Stage 1 scores the simulated validation split, stage 2
# the two real datasets. Both write into one results table.

import pandas as pd

import config
import loading
import revbench as rb

SCRATCH = None          # set on first use; run_suite wants somewhere to write


def _scratch():
    global SCRATCH
    if SCRATCH is None:
        SCRATCH = config.DATA_ROOT / "cache" / "suite"
        SCRATCH.mkdir(parents=True, exist_ok=True)
    return SCRATCH


def classifiers_for(methods):
    # a classifier per method, fitted on the simulated pair. A forest fitted on
    # one embedding cannot read another, so every method gets its own.
    from revbench.detection.classifier import train_pattern_classifier, train_subtype_classifier
    import pickle

    patterned, shuffled = loading.simulated_pair()
    binary, multiclass = {}, {}
    for method in methods:
        if method == "ella":
            continue
        if method == loading.STORED_LATENT:
            # SubCellSpace ships the classifiers used in the paper
            binary[method] = pickle.load(open(config.PATTERN_CLASSIFIER, "rb"))
            multiclass[method] = pickle.load(open(config.SUBTYPE_CLASSIFIER, "rb"))
            continue
        real_arm, control_arm = _as_adata(method, patterned, shuffled)
        binary[method] = train_pattern_classifier(
            real_arm, control_arm, rep=None, seed=config.seed, qc=False)[0]
        multiclass[method] = train_subtype_classifier(
            real_arm, control_arm, rep=None, seed=config.seed, qc=False)[0]
    return binary, multiclass


def _as_adata(method, patterned, shuffled):
    # the patterned and shuffled halves as a pair of AnnDatas carrying this
    # method's embedding. Each method's file brings its own observation keys, so
    # the frame is built from the file rather than the other way round.
    import anndata as ad
    import numpy as np

    if method == loading.STORED_LATENT:
        return patterned, shuffled

    out = []
    for origin, arm in (("pattern", "patterned"), ("random", "shuffled")):
        matrix, keys = loading.read_embedding(method, arm)
        matrix = loading.fill_gaps(np.asarray(matrix, dtype=np.float64), method, arm)
        obs = rb.simulated_metadata(keys)
        obs["origin"] = origin
        obs.index = obs["obs_key"].astype(str)
        out.append(ad.AnnData(X=matrix, obs=obs))
    return out


def stage1(validation, methods=None):
    methods = [m for m in (methods or config.method_order) if m != "ella"]
    keys = loading.shared_observations("simulated", list(validation.obs_names), methods)
    validation = validation[keys].copy()
    for method in methods:
        validation.obsm[method] = loading.simulated_embedding(method, "validation", keys)

    representations = [rb.from_adata(validation, "obsm:" + m, name=m) for m in methods]
    return rb.run_suite(
        validation, representations, dataset="simulated", subset="all",
        benchmarks=["silhouette", "silhouette_grouped", "pattern_probe",
                    "pattern_probe_floor", "stage1_pattern_binary",
                    "stage1_pattern_binary_floor", "negative_controls"],
        n_repeats=3, seed=config.seed, results_dir=_scratch(),
        scale_label=f"full_{validation.n_obs}_rows", n_cells=validation.n_obs)


def stage2(dataset, methods=None):
    methods = methods or config.method_order
    real, control = loading.real_pair(dataset)
    shared = loading.shared_observations(dataset, list(real.obs_names),
                                         [m for m in methods if m != "ella"])
    if len(shared) < real.n_obs:
        real, control = real[shared].copy(), control[shared].copy()
    loading.attach_embeddings(dataset, real, control, methods)

    tested = [str(g) for g in real.obs["gene_name"].unique()]
    truth = loading.ground_truth(dataset, tested)
    binary, multiclass = classifiers_for(methods)
    thresholds = loading.derived_thresholds(dataset)
    served = [m for m in methods if m in real.obsm]

    benchmarks = ["gene_detection", "detection_at_threshold"]
    if dataset == "apex_hek":
        benchmarks.append("subtype_agreement")

    rows = rb.run_suite(
        real, [rb.from_adata(real, "obsm:" + m, name=m) for m in served],
        dataset=dataset, subset="all", benchmarks=benchmarks,
        n_repeats=1, seed=config.seed, results_dir=_scratch(),
        scale_label=f"full_{real.obs['original_obs_index'].nunique()}_cells",
        n_cells=int(real.obs["original_obs_index"].nunique()),
        extra={"real": real, "control": control, "truth": truth,
               "classifier": binary, "multi_classifier": multiclass,
               "control_set": loading.shuffled_control(dataset),
               "distance_threshold": thresholds},
        extra_params={"min_obs": config.min_obs_per_gene})

    if "ella" in methods:
        rows = pd.concat([rows, _ella(dataset, truth, real, tested)], ignore_index=True)
    return rows


def _restrict_fdr(table, genes):
    # ELLA corrects for multiple testing across every gene it was given. The
    # other methods are scored on the genes clearing min_obs, so its correction
    # is recomputed over that same panel -- otherwise its q-values answer a
    # different question than everyone else's.
    import numpy as np

    restricted = table[table.gene_name.isin(set(genes))].copy()
    out = []
    for _, group in (restricted.groupby("repeat") if "repeat" in restricted.columns
                     else [(-1, restricted)]):
        group = group.sort_values("pv_cauchy").reset_index(drop=True)
        n = len(group)
        harmonic = np.sum(1.0 / np.arange(1, n + 1))
        q = group.pv_cauchy.values * n * harmonic / np.arange(1, n + 1)
        group["pv_fdr"] = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
        out.append(group)
    restricted = pd.concat(out, ignore_index=True)
    restricted["ella_neglog10p"] = -np.log10(restricted.pv_fdr.clip(1e-300))
    restricted["passes"] = restricted.pv_fdr <= 0.05
    return restricted


def _ella(dataset, truth, real, tested):
    # ELLA reports a p-value per gene instead of an embedding, so it goes
    # through the gene-level test rather than the classifier
    import numpy as np
    from revbench import gene_level

    real_table, control_table = loading.ella_gene_tables(dataset)

    # recomputing over ELLA's own unrestricted gene set must return ELLA's own
    # q-values, or the restricted ones cannot be trusted either
    check = _restrict_fdr(real_table, real_table.gene_name.unique())
    merged = real_table.merge(check[["gene_name", "repeat", "pv_fdr"]],
                              on=["gene_name", "repeat"], suffixes=("_ella", "_mine"))
    assert float(np.abs(merged.pv_fdr_ella - merged.pv_fdr_mine).max()) < 1e-9

    return gene_level.run_gene_table_suite(
        _restrict_fdr(real_table, tested), truth, dataset=dataset, method="ella",
        score_col="ella_neglog10p", subtype_col="ella_pattern",
        control_table=_restrict_fdr(control_table, tested),
        results_dir=_scratch(), seed=config.seed, subset="all",
        n_cells=int(real.obs["original_obs_index"].nunique()),
        extra_params={"min_obs": config.min_obs_per_gene})


def save(frames, name):
    # each stage writes its own rows; the summary reads the two together
    results = pd.concat(frames, ignore_index=True)
    drop = ["run_id", "run_label", "code_version", "timestamp", "subset",
            "scale_label", "seed", "parent_method"]
    results = results.drop(columns=[c for c in drop if c in results.columns])
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.RESULTS / f"{name}.csv.gz", index=False, compression="gzip")
    return results


def combine():
    stages = [pd.read_csv(config.RESULTS / f"{n}.csv.gz")
              for n in ("stage1", "stage2") if (config.RESULTS / f"{n}.csv.gz").exists()]
    results = pd.concat(stages, ignore_index=True)
    results = results.sort_values(
        ["dataset", "benchmark", "method", "metric", "target", "class_label", "repeat"],
        na_position="last").reset_index(drop=True)
    results.to_csv(config.RESULTS_FILE, index=False, compression="gzip")
    return results
