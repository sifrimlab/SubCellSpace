"""The benchmarks shipped with v1. Importing this module registers them."""
from __future__ import annotations

from typing import List

import numpy as np

from .metrics import negative_control_rows, rf_probe, silhouette_rows
from .registry import register
from .results import MetricRow


def _has_labels(ctx) -> bool:
    """Skip rather than error when the dataset carries no pattern labels."""
    return ctx.label_key in ctx.obs.columns


@register("silhouette")
def _silhouette(ctx) -> List[MetricRow]:
    """Silhouette on the latent, global and one-vs-rest, row-subsampled."""
    if not _has_labels(ctx):
        return []
    return silhouette_rows(
        ctx.X, ctx.obs[ctx.label_key].values,
        groups=ctx.obs[ctx.group_key].values if ctx.group_key in ctx.obs else None,
        sample_by="rows", sample_size=ctx.sample_size,
        n_repeats=ctx.n_repeats, seed=ctx.seed, target=ctx.label_key,
        benchmark="silhouette")


@register("silhouette_grouped")
def _silhouette_grouped(ctx) -> List[MetricRow]:
    """Silhouette sampling whole cells, so rotated copies cannot inflate it."""
    if not _has_labels(ctx):
        return []
    if ctx.group_key not in ctx.obs:
        return []
    return silhouette_rows(
        ctx.X, ctx.obs[ctx.label_key].values,
        groups=ctx.obs[ctx.group_key].values,
        sample_by="groups", sample_size=ctx.sample_size,
        n_repeats=ctx.n_repeats, seed=ctx.seed, target=ctx.label_key,
        benchmark="silhouette_grouped")


@register("pattern_probe")
def _pattern_probe(ctx) -> List[MetricRow]:
    """RF probe for `pattern`, StratifiedGroupKFold grouped on cell_id."""
    if not _has_labels(ctx):
        return []
    groups = ctx.obs[ctx.group_key].values if ctx.group_key in ctx.obs else None
    scheme = "stratified_group" if groups is not None else "stratified"
    return rf_probe(ctx.X, np.asarray(ctx.obs[ctx.label_key].astype(str)),
                    groups=groups, cv_scheme=scheme, n_splits=ctx.n_splits,
                    n_repeats=ctx.n_repeats, seed=ctx.seed,
                    benchmark="pattern_probe", target=ctx.label_key)


@register("pattern_probe_floor")
def _pattern_probe_floor(ctx) -> List[MetricRow]:
    """Chance floor for `pattern`, same estimator and CV, labels permuted."""
    if not _has_labels(ctx):
        return []
    groups = ctx.obs[ctx.group_key].values if ctx.group_key in ctx.obs else None
    scheme = "stratified_group" if groups is not None else "stratified"
    return rf_probe(ctx.X, np.asarray(ctx.obs[ctx.label_key].astype(str)),
                    groups=groups, cv_scheme=scheme, n_splits=ctx.n_splits,
                    n_repeats=ctx.n_repeats, seed=ctx.seed, label_shuffle=True,
                    benchmark="pattern_probe_floor", target=ctx.label_key)


#: The simulated tree's own no-pattern class. This is Stage 1's negative and it
#: is NOT the shuffled control Stage 2 evaluates against: `random` is a cell the
#: simulator generated as unstructured, while the shuffled control is a genuinely
#: patterned cell whose transcripts were relocated inside the same mask, keeping
#: its exact count. A decision boundary fitted against one does not transfer to
#: the other -- measured at 0.30 ROC-AUC on the CNN baseline's own head -- so the
#: two readings must never be quoted for each other.
SIM_RANDOM_CLASS = "random"


def _binary_pattern_labels(ctx) -> np.ndarray:
    """``pattern`` collapsed to patterned-vs-random, as readable class labels.

    Strings rather than a boolean array so the emitted per-class rows and
    confusion cells name the two classes; the partition is exactly
    ``obs[label_key] != "random"``.
    """
    lab = ctx.obs[ctx.label_key].astype(str).to_numpy()
    return np.where(lab == SIM_RANDOM_CLASS, SIM_RANDOM_CLASS, "patterned")


@register("stage1_pattern_binary")
def _stage1_pattern_binary(ctx) -> List[MetricRow]:
    """STAGE 1: does this embedding separate patterned from no-pattern?

    The same estimator, CV scheme, splits, repeats and seed as `pattern_probe`,
    with its six-class label collapsed to two. This is the direct answer to
    "does the model embed features that separate pattern from no-pattern", on
    held-out cells, against the simulation's own negative.

    It is deliberately a separate benchmark from `pattern_classifier_qc`, which
    asks the Stage-2 question -- is this observation *shuffled*? -- on artifact A
    against a different negative. See `SIM_RANDOM_CLASS`.
    """
    if not _has_labels(ctx):
        return []
    y = _binary_pattern_labels(ctx)
    if np.unique(y).size < 2:
        return []          # no `random` rows here, so there is nothing to separate
    groups = ctx.obs[ctx.group_key].values if ctx.group_key in ctx.obs else None
    scheme = "stratified_group" if groups is not None else "stratified"
    return rf_probe(ctx.X, y, groups=groups, cv_scheme=scheme,
                    n_splits=ctx.n_splits, n_repeats=ctx.n_repeats, seed=ctx.seed,
                    benchmark="stage1_pattern_binary", target=ctx.label_key,
                    extra_params={"stage": 1, "negative_class": SIM_RANDOM_CLASS,
                                  "negative_kind": "simulated_random_class"})


@register("stage1_pattern_binary_floor")
def _stage1_pattern_binary_floor(ctx) -> List[MetricRow]:
    """Chance floor for the Stage-1 binary reading, labels permuted."""
    if not _has_labels(ctx):
        return []
    y = _binary_pattern_labels(ctx)
    if np.unique(y).size < 2:
        return []
    groups = ctx.obs[ctx.group_key].values if ctx.group_key in ctx.obs else None
    scheme = "stratified_group" if groups is not None else "stratified"
    return rf_probe(ctx.X, y, groups=groups, cv_scheme=scheme,
                    n_splits=ctx.n_splits, n_repeats=ctx.n_repeats, seed=ctx.seed,
                    label_shuffle=True,
                    benchmark="stage1_pattern_binary_floor", target=ctx.label_key,
                    extra_params={"stage": 1, "negative_class": SIM_RANDOM_CLASS,
                                  "negative_kind": "simulated_random_class"})


@register("negative_controls")
def _negative_controls(ctx) -> List[MetricRow]:
    """Can the latent predict cell_id / n_spots / rotation? It should not."""
    if not any(c in ctx.obs.columns for c in ("rotation_interval_45", "n_spots_interval_fixed")):
        return []
    depths = (5,) if ctx.smoke else (5, None)
    return negative_control_rows(
        ctx.X, ctx.obs, group_key=ctx.group_key, n_splits=ctx.n_splits,
        n_repeats=ctx.n_repeats, seed=ctx.seed, max_depths=depths)


# --- detection ---------------------------------------------------------------
# These read the exploded (real, control) pair from ctx.extra rather than using
# ctx.row_mask. Detection is a per-gene statistic, so row subsampling would
# corrupt each gene's n_obs; smoke mode caps the number of genes instead.

_CLF_CACHE = {}


def _per_method(value, method):
    """Resolve a supplied estimator that may be per-method.

    Accepts a single estimator (used for every representation) or a dict keyed
    by method name. The dict form is what a mixed-dimensionality comparison
    needs: a 15-feature classifier cannot score a 64-dim latent, so each method
    must bring its own.
    """
    if isinstance(value, dict):
        return value.get(method)
    return value


def _classifier_for(ctx):
    """The classifier to score with, plus its QC rows and provenance tag."""
    supplied = _per_method(ctx.extra.get("classifier"), ctx.rep.name)
    if supplied is not None:
        return supplied, [], "shipped"
    key = (ctx.rep.name, "pattern_nonpattern", ctx.seed)
    if key not in _CLF_CACHE:
        from .detection.classifier import load_training_pair, train_pattern_classifier
        train_rep = ctx.extra.get("train_rep", "latent")
        real_t, rand_t = ctx.extra.get("training_pair") or load_training_pair()
        _CLF_CACHE[key] = train_pattern_classifier(
            real_t, rand_t, rep=train_rep, seed=ctx.seed,
            qc=True, qc_repeats=1 if ctx.smoke else ctx.n_repeats)
    clf, qc = _CLF_CACHE[key]
    return clf, qc, "retrained"


def _control_fingerprint(ctx):
    """Identity of the shuffled points these scores were computed against.

    Reads ``extra["control_set"]`` -- a ControlSet -- which is distinct from
    ``extra["control"]``, the exploded control AnnData.
    """
    cs = ctx.extra.get("control_set")
    fp = getattr(cs, "fingerprint", None)
    if cs is not None and fp is None:
        raise TypeError("extra['control_set'] must be a ControlSet, got {}"
                        .format(type(cs).__name__))
    return fp or ""


def _detect(ctx):
    """Run (and memoise per context) the per-gene detection table."""
    cache = ctx.extra.setdefault("_detect_cache", {})
    clf, qc, source = _classifier_for(ctx)
    # The earth-mover cut this method is scored at. `None` -- the default --
    # keeps `detect_genes`' per-dataset value, so a caller that supplies nothing
    # behaves exactly as it did before. `reduce/` supplies a dict keyed by method
    # name, derived on simulated data by `detection.calibrate`, because the cut
    # is a point on a probability-distribution distance and each method's
    # classifier has its own scale. It is part of the cache key: two contexts
    # differing only in threshold are not the same result.
    thr = _per_method(ctx.extra.get("distance_threshold"), ctx.rep.name)
    key = (ctx.rep.name, source, thr)
    if key not in cache:
        from .detection.genes import detect_genes
        real, control = ctx.extra["real"], ctx.extra["control"]
        genes = None
        if ctx.smoke:
            counts = real.obs["gene_name"].value_counts()
            genes = [str(g) for g in counts.index[:ctx.extra.get("max_genes", 8)]]
        # store_adata=True keeps the per-gene slices that subtype assignment
        # reads back; ~60 MB on APEX, freed with the run.
        # Score THIS representation. When several methods share one pair of
        # exploded AnnDatas, each method's embedding lives in obsm under its own
        # name; a single-method run keeps its embedding in .X.
        rep = ctx.rep.name if ctx.rep.name in getattr(real, "obsm", {}) else None
        result = detect_genes(real, control, clf, rep=rep, dataset=ctx.dataset,
                              distance_threshold=thr,
                              seed=ctx.seed, gene_panel=genes, store_adata=True,
                              max_obs=200 if ctx.smoke else 1000)
        cache[key] = (result, source, qc)
    return cache[key]


@register("pattern_classifier_qc", datasets=("apex_hek", "ileum"),
          requires=("real", "control", "truth"))
def _classifier_qc(ctx):
    """Held-out quality of the detector itself, so a weak classifier is visible."""
    _, _, qc = _detect(ctx)
    fp = _control_fingerprint(ctx)
    for r in qc:
        r.params["control_fingerprint"] = fp
    return list(qc)


@register("gene_detection", datasets=("apex_hek", "ileum"),
          requires=("real", "control", "truth"))
def _gene_detection(ctx):
    """Per-gene effect size and significance against the shuffled control."""
    from .results import MetricRow
    result, source, _ = _detect(ctx)
    p = {"classifier_source": source,
         "control_fingerprint": _control_fingerprint(ctx),
         "distance_threshold": result.attrs.get("distance_threshold"),
         "max_obs": result.attrs.get("max_obs")}
    n, ng = int(result["n_obs"].sum()), len(result)
    out = [
        MetricRow(benchmark="gene_detection", metric="n_genes_tested",
                  value=float(ng), n=n, n_groups=ng, target=ctx.dataset,
                  seed=ctx.seed, params=p),
        MetricRow(benchmark="gene_detection", metric="median_earth_dist",
                  value=float(result["earth_dist"].median()), n=n, n_groups=ng,
                  target=ctx.dataset, seed=ctx.seed, params=p),
        MetricRow(benchmark="gene_detection", metric="frac_passing",
                  value=float((result["passes"] & result["passes_earth_thresh"]).mean()),
                  n=n, n_groups=ng, target=ctx.dataset, seed=ctx.seed, params=p),
    ]
    return out


@register("detection_sweep", datasets=("apex_hek", "ileum"),
          requires=("real", "control", "truth"))
def _detection_sweep(ctx):
    """Threshold-free ROC/AP on earth_dist, plus the Youden-J optimum."""
    from .detection.sweep import threshold_sweep
    result, source, _ = _detect(ctx)
    _, rows = threshold_sweep(result, ctx.extra["truth"], seed=ctx.seed)
    for r in rows:
        r.params["classifier_source"] = source
        r.params["control_fingerprint"] = _control_fingerprint(ctx)
        r.params["control_fingerprint"] = _control_fingerprint(ctx)
    return rows


@register("detection_at_threshold", datasets=("apex_hek", "ileum"),
          requires=("real", "control", "truth"))
def _detection_at_threshold(ctx):
    """Published call: Bonferroni-significant KS and effect size over the cut."""
    from .detection.sweep import detection_metrics_at
    from .config import dataset_threshold
    result, source, _ = _detect(ctx)
    thr = result.attrs.get("distance_threshold") or dataset_threshold(ctx.dataset)
    _, rows = detection_metrics_at(result, ctx.extra["truth"], thr, seed=ctx.seed)
    # Where that cut came from, stamped so a reader never has to guess whether a
    # fixed-threshold number was scored at the published value or at one derived
    # for this method. Defaults to "published", so an unconfigured caller is
    # unchanged; `reduce/` passes "derived_fig4d" for every method but the
    # reference. `threshold` itself is already in params, from
    # `detection_metrics_at`.
    src = _per_method(ctx.extra.get("threshold_source"), ctx.rep.name) or "published"
    for r in rows:
        r.params["classifier_source"] = source
        r.params["threshold_source"] = src
    return rows


_SUBTYPE_CACHE = {}


def _subtype_classifier_for(ctx):
    """Multiclass subtype classifier, supplied or retrained for this latent."""
    supplied = _per_method(ctx.extra.get("multi_classifier"), ctx.rep.name)
    if supplied is not None:
        return supplied, "shipped"
    key = (ctx.rep.name, "subtype_multiclass", ctx.seed)
    if key not in _SUBTYPE_CACHE:
        from .detection.classifier import load_training_pair, train_subtype_classifier
        real_t, ctrl_t = ctx.extra.get("training_pair") or load_training_pair()
        # the control is required: it supplies the `random` class the published
        # subtype assignment drops, and without it that call raises KeyError
        _SUBTYPE_CACHE[key] = train_subtype_classifier(
            real_t, ctrl_t, rep=ctx.extra.get("train_rep", "latent"),
            seed=ctx.seed, qc=False)
    return _SUBTYPE_CACHE[key][0], "retrained"


# APEX only: the ileum's subtype crosstab is underpowered (13 genes, both
# polarities landing mostly on `pericellular`, chi-square p=0.42) and is not
# part of the benchmark. Every other ileum benchmark stays.
@register("subtype_agreement", datasets=("apex_hek",),
          requires=("real", "control", "truth"))
def _subtype_agreement(ctx):
    """Association between the ground-truth compartment and the called subtype.

    The two label sets are different vocabularies (biological compartment vs
    morphological subtype), so this reports association -- Cramer's V, adjusted
    mutual information, adjusted Rand -- not accuracy. The crosstab itself is
    written to the caller's ``results_dir``/tables/ for the figure.
    """
    from .detection.subtypes import assign_subtypes, subtype_agreement_rows

    result, source, _ = _detect(ctx)
    multi, multi_source = _subtype_classifier_for(ctx)
    if "final_class" not in result.columns:
        assign_subtypes(result, multi)

    counts, rows = subtype_agreement_rows(
        result, ctx.extra["truth"], seed=ctx.seed,
        extra_params={"classifier_source": source,
                      "subtype_classifier_source": multi_source,
                      "control_fingerprint": _control_fingerprint(ctx)})

    out = ctx.results_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / "{}__{}__subtype_crosstab.csv".format(ctx.dataset, ctx.rep.name))
    return rows
