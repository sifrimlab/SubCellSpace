"""Per-gene detection calls.

Delegates to the published
``make_paper_figures/src/classifiers.py::classify_all_genes`` **unmodified**.
That function is the statistic the paper reports, and reusing rather than
reimplementing it is what makes an exact reproduction meaningful. All this
module adds is a representation-aware view, the preconditions that make its
internal assertion legible, and a per-dataset threshold.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ..config import (MIN_OBS_PER_GENE, PUBLISHED_SRC, SEED, dataset_threshold)


def _published_src() -> str:
    src = str(PUBLISHED_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def rep_matrix(adata, rep: Optional[str] = None) -> np.ndarray:
    """The embedding to score: ``obsm[rep]`` when named, else ``X``."""
    X = adata.X if rep is None else adata.obsm[rep]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X)


def _rep_view(adata, rep: Optional[str] = None,
              obs_cols: Sequence[str] = ("gene_name", "n_obs")) -> ad.AnnData:
    """A bare AnnData with ``X`` set to the chosen representation.

    Deliberately *not* ``adata.copy()`` or a view:

    * ``classify_all_genes`` writes ``adata.obs["earth_dist"]`` and
      ``adata.obs["gene_pattern"]`` -- it mutates, so a view would scribble on
      the caller's object;
    * ``obsm["detected_transcripts"]`` is a ragged awkward array that is slow
      and lossy to slice once per gene;
    * it marks ``.X`` as a legacy *calling* convention, not a storage one.
    """
    keep = [c for c in obs_cols if c in adata.obs.columns]
    obs = adata.obs[keep].copy()
    obs.index = adata.obs_names
    return ad.AnnData(X=rep_matrix(adata, rep).astype(np.float64), obs=obs)


def _check(real, control, classifier, rep) -> Dict[str, Any]:
    """Fail readably, rather than through classify_all_genes' bare assert."""
    if real.n_obs != control.n_obs:
        raise ValueError("real has {} rows, control {}".format(real.n_obs, control.n_obs))
    if not real.obs_names.equals(control.obs_names):
        n_diff = int((real.obs_names != control.obs_names).sum())
        raise ValueError(
            "real and control indexes differ in {} of {} positions; they must "
            "match in the same order (use detection.explode_matched)"
            .format(n_diff, real.n_obs))
    if "gene_name" not in real.obs.columns:
        raise KeyError("real.obs needs a 'gene_name' column")
    dim = rep_matrix(real, rep).shape[1]
    n_in = getattr(classifier, "n_features_in_", None)
    if n_in is not None and int(n_in) != dim:
        raise ValueError(
            "classifier expects {} features but the representation has {}. A "
            "classifier fitted on one latent space cannot score another -- "
            "retrain it with detection.train_pattern_classifier."
            .format(n_in, dim))
    classes = list(getattr(classifier, "classes_", []))
    if "pattern" not in classes:
        raise ValueError("classifier.classes_ must contain 'pattern'; got {}".format(classes))
    return {"latent_dim": int(dim), "n_rows": int(real.n_obs)}


def select_genes(adata, min_obs: int = MIN_OBS_PER_GENE,
                 gene_panel: Optional[Sequence[str]] = None) -> List[str]:
    """Genes with enough observations to test.

    This also sets the Bonferroni denominator, since ``classify_all_genes``
    multiplies by ``len(genes_to_test)`` -- testing under-powered genes would
    inflate the correction for every other gene.
    """
    counts = adata.obs["gene_name"].value_counts()
    genes = [str(g) for g, n in counts.items() if n >= min_obs]
    if gene_panel is not None:
        panel = set(map(str, gene_panel))
        genes = [g for g in genes if g in panel]
    return sorted(genes)


def detect_genes(real, control, classifier, rep: Optional[str] = None,
                 dataset: Optional[str] = None,
                 distance_threshold: Optional[float] = None,
                 min_obs: int = MIN_OBS_PER_GENE,
                 max_obs: Optional[int] = 1000,
                 gene_panel: Optional[Sequence[str]] = None,
                 seed: int = SEED, store_adata: bool = False,
                 progress: bool = False) -> pd.DataFrame:
    """Per-gene Wasserstein distance and KS p-value against the shuffled control.

    ``distance_threshold`` defaults per dataset (0.06 for ``apex_hek``, 0.03 for
    ``ileum``) -- the published analyses do not share one value.
    """
    _published_src()
    from classifiers import classify_all_genes

    info = _check(real, control, classifier, rep)
    if distance_threshold is None:
        distance_threshold = dataset_threshold(dataset)

    genes = select_genes(real, min_obs=min_obs, gene_panel=gene_panel)
    if not genes:
        raise ValueError("no gene reaches min_obs={}".format(min_obs))

    keep = real.obs["gene_name"].astype(str).isin(genes).values
    rv = _rep_view(real, rep)[keep]
    cv = _rep_view(control, rep)[keep]

    # classify_all_genes drives a tqdm bar on stderr; useful live, but it fills a
    # saved notebook with thousands of progress frames.
    import contextlib, io
    ctx = contextlib.nullcontext() if progress else contextlib.redirect_stderr(io.StringIO())
    with ctx:
        result = classify_all_genes(
            rv, classifier, cv, earth_dist_threshold=distance_threshold,
            max_number_of_observations_to_consider=max_obs,
            random_seed=seed, store_adata=store_adata)

    result["gene_name"] = result["gene_name"].astype(str)
    result.attrs.update(info)
    result.attrs["distance_threshold"] = float(distance_threshold)
    result.attrs["dataset"] = dataset or ""
    result.attrs["n_genes_tested"] = len(genes)
    result.attrs["max_obs"] = max_obs
    result.attrs["seed"] = seed
    return result
