"""Threshold sweeps and fixed-threshold detection metrics.

Vendors ``compute_roc_stats_with_optimal_threshold`` (defined inline in the HEK
interpretation notebook and in ``fig4``) with the plotting stripped out and the
results returned as tidy rows.

Two views, deliberately separated:

* **Threshold-free** -- ROC/AP on ``earth_dist`` alone. This is the headline
  number for comparing methods, because the 0.06 cut was calibrated for a
  15-dim latent at ~1000 observations and does not transfer to a different
  dimensionality. For a cut that *does* transfer, derive one per method on
  simulated data with :mod:`revbench.detection.calibrate`.
* **At a fixed threshold** -- reproduces the published rule
  ``passes and passes_earth_thresh``, i.e. a Bonferroni-significant KS test
  *and* an effect size above the cut.

The Bonferroni gate is excluded from the sweep on purpose: sweeping a threshold
on a score that is separately gated by a hypothesis test is not a sweep of that
score. It is reported alongside as ``n_pval_pass``.

The Youden-J optimum is not emitted
-----------------------------------
``threshold_sweep`` used to write ``youden_j_max``, ``threshold_at_youden`` and
five ``<metric>_at_youden`` rows into the results store. **It no longer does**,
and the rows have been stripped from the store.

The optimum is chosen by maximising Youden's J over the *evaluation* dataset's
own ground-truth panel. It is fitted on the labels it is then scored against, so
it cannot be computed for a dataset that has no panel -- which is every dataset
this is actually applied to. The question these benchmarks ask is how a
workflow performs on real data it has never seen, at a threshold derived
beforehand; a label-fitted cut is not a weaker answer to that question but an
answer to a different and inadmissible one -- "how well could this rank if
someone had already told you the answers". Keeping it in the store, even
unreported, is an invitation to quote it.

Removed in both directions: on the datasets here it happened to favour the
reference method on one dataset and its competitors on another, and that is
beside the point.

The full sweep ``curve`` is still returned to the caller, ``youden_j`` column
included, so anything that genuinely needs to inspect it can. Note that
``detection_metrics_at`` does still emit a ``youden_j`` row: that one is
Youden's J evaluated **at the blind threshold**, i.e. a summary of the applied
2x2 exactly like ``mcc``, not a fitted threshold. Do not confuse the two.

Two notebooks outside ``reduce/`` -- ``02_benchmark_ella.ipynb`` and
``03_benchmark_a_subcellspace_variant.ipynb`` -- still read ``f1_at_youden``,
including as a published reference value. They will show it as missing if
re-run. That is deliberate and left visible rather than papered over.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from ..results import MetricRow
from .truth import GroundTruth, truth_frame


def _counts(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum(pred & y_true))
    fp = int(np.sum(pred & ~y_true))
    fn = int(np.sum(~pred & y_true))
    tn = int(np.sum(~pred & ~y_true))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _rates(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    spec = 1.0 - fpr
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"tpr": tpr, "fpr": fpr, "precision": prec, "recall": tpr,
            "specificity": spec, "f1": f1, "mcc": mcc, "accuracy": acc,
            "youden_j": tpr + spec - 1.0}


def threshold_sweep(result_df: pd.DataFrame, truth: GroundTruth,
                    score_col: str = "earth_dist",
                    thresholds: Optional[Sequence[float]] = None,
                    require_pval: bool = False,
                    benchmark: str = "detection_sweep",
                    repeat: int = -1, seed: int = -1
                    ) -> Tuple[pd.DataFrame, List[MetricRow]]:
    """ROC/AP on the score, plus the metrics at the Youden-J optimum."""
    tf = truth_frame(truth, result_df, score_col=score_col)
    if tf.empty:
        raise ValueError("no gene in result_df carries a ground-truth label")
    y = tf["y_true"].values.astype(bool)
    score = tf["score"].values.astype(float)
    if y.all() or (~y).all():
        raise ValueError("ground truth has only one class among tested genes "
                         "({} positive / {} negative)".format(int(y.sum()), int((~y).sum())))

    gate = np.ones_like(y)
    if require_pval and "passes" in result_df.columns:
        gate = (result_df.set_index(result_df["gene_name"].astype(str))
                .loc[tf["gene_name"], "passes"].values.astype(bool))

    if thresholds is None:
        thresholds = np.unique(np.round(np.concatenate(
            [[0.0], np.linspace(score.min(), score.max(), 1000)]), 6))

    rows_curve = []
    for thr in thresholds:
        pred = (score > thr) & gate
        c = _counts(y, pred)
        rows_curve.append(dict(threshold=float(thr), **c, **_rates(c)))
    curve = pd.DataFrame(rows_curve)

    n = int(len(y))
    params = {"score_col": score_col, "require_pval": require_pval,
              "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
              "truth": truth.name, "n_thresholds": int(len(thresholds)),
              "n_dropped_unlabelled": int(tf.attrs.get("n_dropped_unlabelled", 0))}

    rows: List[MetricRow] = []

    def add(metric, value, extra=None):
        rows.append(MetricRow(benchmark=benchmark, metric=metric,
                              value=float(value), n=n,
                              n_groups=params["n_pos"], target=truth.name,
                              repeat=repeat, seed=seed,
                              params=dict(params, **(extra or {}))))

    add("roc_auc", skm.roc_auc_score(y, score))
    add("average_precision", skm.average_precision_score(y, score))
    # The Youden-J optimum used to be emitted here as `youden_j_max`,
    # `threshold_at_youden` and five `<metric>_at_youden` rows. It no longer is.
    # See "The Youden-J optimum is not emitted" in the module docstring: it is
    # fitted on the labels it is then scored against, so it does not belong in a
    # results store at all. The full curve is still RETURNED, so a caller that
    # genuinely wants to inspect it can -- it just does not get written down as
    # if it were a result.
    if "passes" in result_df.columns:
        add("n_pval_pass", int(result_df["passes"].sum()))
        add("frac_pval_pass", float(result_df["passes"].mean()))
    return curve, rows


def detection_metrics_at(result_df: pd.DataFrame, truth: GroundTruth,
                         threshold: float, score_col: str = "earth_dist",
                         require_pval: bool = True,
                         benchmark: str = "detection_at_threshold",
                         repeat: int = -1, seed: int = -1
                         ) -> Tuple[pd.DataFrame, List[MetricRow]]:
    """Confusion matrix and metrics at one threshold.

    ``require_pval=True`` reproduces the published call: a gene is detected when
    its Bonferroni-corrected KS test passes *and* its effect size clears the cut.
    """
    tf = truth_frame(truth, result_df, score_col=score_col)
    y = tf["y_true"].values.astype(bool)
    score = tf["score"].values.astype(float)
    gate = np.ones_like(y)
    if require_pval and "passes" in result_df.columns:
        gate = (result_df.set_index(result_df["gene_name"].astype(str))
                .loc[tf["gene_name"], "passes"].values.astype(bool))
    pred = (score > threshold) & gate

    c = _counts(y, pred)
    r = _rates(c)
    confusion = pd.DataFrame(
        [[c["tp"], c["fn"]], [c["fp"], c["tn"]]],
        index=["truth_positive", "truth_negative"],
        columns=["called_positive", "called_negative"])

    params = {"score_col": score_col, "threshold": float(threshold),
              "require_pval": require_pval, "truth": truth.name,
              "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
              "n_dropped_unlabelled": int(tf.attrs.get("n_dropped_unlabelled", 0))}
    rows = [MetricRow(benchmark=benchmark, metric=k, value=float(v),
                      n=int(len(y)), n_groups=params["n_pos"],
                      target=truth.name, repeat=repeat, seed=seed, params=params)
            for k, v in list(r.items()) + [(k, v) for k, v in c.items()]]
    return confusion, rows
