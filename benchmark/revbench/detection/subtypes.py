"""Subtype agreement: does the model's pattern subtype track the ground-truth
compartment?

The two label sets are **different vocabularies**. Ground truth is a biological
compartment (``Nucleus``, ``Nucleolus``, ``Lamina``, ``Nuclear_Pore``,
``Cytosol``, ``ERM``, ``OMM`` for APEX-seq; ``apical``/``basal`` for the ileum);
the model emits a morphological subtype (``intranuclear``, ``perinuclear``,
``extranuclear``, ``cell-edge``, ``pericellular``, ``protrusion``). There is no
canonical one-to-one mapping between them, so there is no "accuracy" to report
here -- the question is whether the two are *associated*, i.e. whether knowing
the model's subtype tells you anything about the compartment.

That is why the metrics are association measures (Cramer's V, adjusted mutual
information, adjusted Rand, a chi-square test) rather than a diagonal hit rate.
Reading a diagonal off this table would be a category error.

Reproduces the crosstab in ``fig4`` cell 89 / the HEK notebook cell 194.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import PUBLISHED_SRC
from ..results import MetricRow
from .truth import GroundTruth

RANDOM_LABEL = "random"


def assign_subtypes(result_df: pd.DataFrame, multi_classifier) -> pd.DataFrame:
    """Add ``subtype_pattern`` and ``final_class`` to a detection table.

    Delegates to the published
    ``make_paper_figures/src/classifiers.py::classify_all_genes_for_pattern_subtypes``.
    Requires ``detect_genes(..., store_adata=True)``, since that function reads
    the per-gene AnnData slices back off the frame.

    ``subtype_pattern`` is the median-argmax subtype for every gene;
    ``final_class`` is that subtype for genes that pass both detection gates and
    ``"random"`` otherwise.
    """
    src = str(PUBLISHED_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from classifiers import classify_all_genes_for_pattern_subtypes

    for col in ("pattern", "control"):
        if col not in result_df.columns:
            raise KeyError(
                "result_df has no {!r} column -- run detect_genes(..., "
                "store_adata=True), which keeps the per-gene slices the subtype "
                "assignment needs.".format(col))
    n_in = getattr(multi_classifier, "n_features_in_", None)
    dim = result_df["pattern"].iloc[0].X.shape[1]
    if n_in is not None and int(n_in) != int(dim):
        raise ValueError(
            "subtype classifier expects {} features but the representation has "
            "{}; retrain it with detection.train_subtype_classifier."
            .format(n_in, dim))
    return classify_all_genes_for_pattern_subtypes(result_df, multi_classifier)


def subtype_crosstab(result_df: pd.DataFrame, truth: GroundTruth,
                     call_col: str = "final_class",
                     drop_random: bool = True
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Counts and row-normalised fractions of compartment x subtype.

    Only genes that carry a ground-truth compartment *and* received a
    non-random call enter the table -- matching the published figure, and for
    the same reason: a gene the model declined to call says nothing about the
    association between the two vocabularies.
    """
    if call_col not in result_df.columns:
        raise KeyError("{!r} not in result_df; run assign_subtypes first"
                       .format(call_col))
    df = result_df[["gene_name", call_col]].copy()
    df["gene_name"] = df["gene_name"].astype(str)
    df["truth_label"] = df["gene_name"].map(truth.labels)
    df = df.dropna(subset=["truth_label"])
    if drop_random:
        df = df[df[call_col].astype(str) != RANDOM_LABEL]
        df = df[df["truth_label"].astype(str) != RANDOM_LABEL]
    counts = pd.crosstab(df["truth_label"], df[call_col])
    fracs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    counts.attrs["n_genes"] = int(len(df))
    return counts, fracs


def _cramers_v(counts: pd.DataFrame) -> Tuple[float, float, float]:
    """Cramer's V with its chi-square statistic and p-value."""
    from scipy.stats import chi2_contingency
    arr = counts.to_numpy(dtype=float)
    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return float("nan"), float("nan"), float("nan")
    chi2, p, _, _ = chi2_contingency(arr)
    n = arr.sum()
    k = min(arr.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n and k else float("nan")
    return v, float(chi2), float(p)


def subtype_agreement_rows(result_df: pd.DataFrame, truth: GroundTruth,
                           call_col: str = "final_class",
                           benchmark: str = "subtype_agreement",
                           repeat: int = -1, seed: int = -1,
                           extra_params: Optional[Dict[str, Any]] = None
                           ) -> Tuple[pd.DataFrame, List[MetricRow]]:
    """Association between the ground-truth compartment and the model's subtype."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    counts, fracs = subtype_crosstab(result_df, truth, call_col=call_col)
    n = int(counts.attrs.get("n_genes", counts.to_numpy().sum()))

    # The crosstab itself, cell by cell, into the results store. It was already
    # written to results/tables/, but that filename carries no scale or run, so a
    # later run at a different scale silently overwrote it -- and the matrix is
    # the thing a reader actually wants to see behind Cramer's V.
    crosstab_rows = []
    for truth_label in counts.index:
        for call in counts.columns:
            crosstab_rows.append(MetricRow(
                benchmark=benchmark, metric="crosstab_count",
                value=float(counts.loc[truth_label, call]),
                n=n, n_groups=int(counts.shape[0]), target=truth.name,
                class_label="%s|%s" % (truth_label, call),   # "<truth>|<called>"
                repeat=repeat, seed=seed,
                params=dict(extra_params or {}, matrix="subtype_crosstab",
                            call_col=call_col)))

    df = result_df[["gene_name", call_col]].copy()
    df["gene_name"] = df["gene_name"].astype(str)
    df["truth_label"] = df["gene_name"].map(truth.labels)
    df = df.dropna(subset=["truth_label"])
    df = df[df[call_col].astype(str) != RANDOM_LABEL]
    a = df["truth_label"].astype(str).values
    b = df[call_col].astype(str).values

    v, chi2, p = _cramers_v(counts)
    params = {"call_col": call_col, "truth": truth.name,
              "n_truth_classes": int(counts.shape[0]),
              "n_call_classes": int(counts.shape[1]),
              "note": "different vocabularies; association, not accuracy"}
    params.update(extra_params or {})

    def row(metric, value, class_label=""):
        return MetricRow(benchmark=benchmark, metric=metric, value=float(value),
                         n=n, n_groups=int(counts.shape[0]), target=truth.name,
                         class_label=class_label, repeat=repeat, seed=seed,
                         params=params)

    rows = [
        row("cramers_v", v),
        row("chi2", chi2),
        row("chi2_pvalue", p),
        row("adjusted_mutual_info", adjusted_mutual_info_score(a, b) if n else np.nan),
        row("adjusted_rand", adjusted_rand_score(a, b) if n else np.nan),
        row("n_genes", n),
    ]
    # Per-compartment: how concentrated is it on its most common subtype, and
    # which one. Concentration is the interpretable per-row summary when the
    # vocabularies do not line up.
    for compartment in counts.index:
        r = fracs.loc[compartment]
        if r.isna().all():
            continue
        rows.append(row("top_subtype_fraction", float(r.max()), str(compartment)))
        rows.append(row("n_genes_in_compartment",
                        float(counts.loc[compartment].sum()), str(compartment)))
    return counts, rows + crosstab_rows
