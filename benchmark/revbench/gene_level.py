"""Scoring a method that reports per *gene* rather than per observation.

``run_suite`` scores ``Representation``s: it needs an embedding per observation,
intersects the valid rows across methods, and records a ``latent_dim``. ELLA has
none of that -- it fits a model per (cell type, gene) and reports a p-value, a
peak-position score and a pattern label. Handing it a placeholder
``Representation`` just to get through ``common_valid`` would put a meaningless
``latent_dim`` and a fabricated row count into the results table, so gene-level
methods get their own entry point here.

What it does *not* do is reimplement any scoring. ``threshold_sweep``,
``detection_metrics_at`` and ``subtype_agreement_rows`` already take a per-gene
frame, a ground truth and a score column; the only thing missing was a caller
that stamps their output into the same shard store ``run_suite`` writes to, so
one ``rb.summarize(rb.load_all())`` sees every method regardless of its shape.

ELLA's p-value is stochastic (see ``adapters/ella.py::estimate``), so a gene
table may carry several ``repeat``s at different seeds. Every benchmark is run
once per repeat and the spread is left in the results rather than averaged away
here -- ``summarize`` is where averaging belongs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, SEED, code_version, dataset_threshold
from .detection.sweep import detection_metrics_at, threshold_sweep
from .results import (MetricRow, infer_scale_label, new_run_id, stamp, utcnow,
                      write_manifest, write_shard)

#: ELLA's own decision rule. `sig_cutoff` is 0.05 on the BY-adjusted p-value,
#: so the fixed-threshold arm reproduces ELLA's call rather than one we chose.
SIG_CUTOFF = 0.05


def calibration_rows(gene_table: pd.DataFrame, score_col: str = "pv_fdr",
                     cutoff: float = SIG_CUTOFF, benchmark: str = "control_calibration",
                     params: Optional[Dict[str, Any]] = None) -> List[MetricRow]:
    """What the method calls on the shared *shuffled* points.

    ELLA tests each gene against internal uniformity rather than against a
    control, so the control arm is not part of its statistic -- which is exactly
    why running it is worth something: every gene called significant here is a
    false positive by construction, on the same shuffled points every other
    method was scored against.
    """
    rows: List[MetricRow] = []
    for repeat, sub in gene_table.groupby(gene_table.get("repeat", 0)):
        p = sub[score_col].to_numpy(dtype=float)
        raw = sub["pv_cauchy"].to_numpy(dtype=float) if "pv_cauchy" in sub else p
        base = dict(params or {}, score_col=score_col, cutoff=cutoff)
        n = int(len(p))
        for metric, value in (
                ("n_genes_tested", float(n)),
                ("n_false_positives", float((p <= cutoff).sum())),
                ("frac_false_positives", float((p <= cutoff).mean())),
                ("median_raw_pvalue", float(np.median(raw))),
                ("frac_raw_p_below_05", float((raw <= 0.05).mean())),
                ("ks_uniform_pvalue", _ks_uniform(raw))):
            rows.append(MetricRow(benchmark=benchmark, metric=metric, value=value,
                                  n=n, n_groups=n, target="shuffled_control",
                                  repeat=int(repeat), seed=int(sub["seed"].iloc[0])
                                  if "seed" in sub else -1, params=base))
    return rows


def _ks_uniform(p: np.ndarray) -> float:
    """One-sample KS of the raw p-values against Uniform(0, 1).

    A calibrated test has uniform p under the null; this is the number behind
    the QQ plot, so the claim is testable and not just a picture.
    """
    from scipy import stats
    x = np.asarray(p, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    return float(stats.kstest(x, "uniform").pvalue)


def gene_table_rows(gene_table: pd.DataFrame, truth, dataset: str,
                    score_col: str, pass_col: str = "passes",
                    subtype_col: Optional[str] = None,
                    seed: int = SEED,
                    params: Optional[Dict[str, Any]] = None,
                    results_dir: Path = RESULTS_DIR) -> List[MetricRow]:
    """Sweep + fixed-threshold + (optional) subtype rows for one gene table."""
    from .detection.subtypes import subtype_agreement_rows

    base = dict(params or {})
    rows: List[MetricRow] = []
    grouped = (gene_table.groupby("repeat") if "repeat" in gene_table.columns
               else [(-1, gene_table)])

    for repeat, sub in grouped:
        sub = sub.reset_index(drop=True)
        rep_seed = int(sub["seed"].iloc[0]) if "seed" in sub.columns else seed
        p = dict(base, repeat=int(repeat))

        n_pos_call = int(sub[pass_col].sum()) if pass_col in sub.columns else -1
        rows.append(MetricRow(benchmark="gene_detection_native",
                              metric="n_genes_tested", value=float(len(sub)),
                              n=len(sub), n_groups=len(sub), target=dataset,
                              repeat=int(repeat), seed=rep_seed, params=p))
        rows.append(MetricRow(benchmark="gene_detection_native",
                              metric="n_called_significant", value=float(n_pos_call),
                              n=len(sub), n_groups=len(sub), target=dataset,
                              repeat=int(repeat), seed=rep_seed, params=p))
        rows.append(MetricRow(benchmark="gene_detection_native",
                              metric="frac_called_significant",
                              value=float(n_pos_call / max(1, len(sub))),
                              n=len(sub), n_groups=len(sub), target=dataset,
                              repeat=int(repeat), seed=rep_seed, params=p))

        # Threshold-free: the headline. ELLA's significance call saturates at
        # APEX depth (264/264), so the ranking is the only part of the p-value
        # that can separate anything, and this is where it is measured.
        _, sweep = threshold_sweep(sub, truth, score_col=score_col,
                                   benchmark="detection_sweep_native",
                                   repeat=int(repeat), seed=rep_seed)
        for r in sweep:
            r.params.update(p)
        rows += sweep

        # At the method's own cut. For ELLA that is pv_fdr <= 0.05, expressed on
        # the score column as -log10(0.05).
        thr = float(-np.log10(SIG_CUTOFF)) if score_col.endswith("neglog10p") \
            else dataset_threshold(dataset)
        _, fixed = detection_metrics_at(sub, truth, thr, score_col=score_col,
                                        require_pval=False,
                                        benchmark="detection_at_threshold_native",
                                        repeat=int(repeat), seed=rep_seed)
        for r in fixed:
            r.params.update(p, threshold_source="method_own_rule")
        rows += fixed

        if subtype_col and subtype_col in sub.columns:
            counts, agree = subtype_agreement_rows(
                sub, truth, call_col=subtype_col,
                benchmark="subtype_agreement_native",
                repeat=int(repeat), seed=rep_seed, extra_params=p)
            rows += agree
            out = results_dir / "tables"
            out.mkdir(parents=True, exist_ok=True)
            counts.to_csv(out / "{}__{}__subtype_crosstab_native_r{}.csv"
                          .format(dataset, base.get("method", "gene_level"), repeat))
    return rows


def run_gene_table_suite(gene_table: pd.DataFrame, truth, dataset: str, method: str,
                         score_col: str = "ella_neglog10p",
                         subtype_col: Optional[str] = None,
                         control_table: Optional[pd.DataFrame] = None,
                         control_fingerprint: str = "",
                         subset: str = "val", seed: int = SEED,
                         run_label: str = "", scale_label: str = "",
                         n_cells: Optional[int] = None,
                         extra_params: Optional[Dict[str, Any]] = None,
                         results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Score a gene-level method and write a shard, mirroring ``run_suite``.

    ``latent_dim`` is written as 0 rather than left blank: the method genuinely
    has no latent space, and a zero says so where an empty cell would look like
    a missing value.
    """
    t0 = time.time()
    run_id = new_run_id()
    ts = utcnow()
    params = dict(extra_params or {})
    params.setdefault("method", method)
    params["score_col"] = score_col

    # A gene-level method does not necessarily use the control. ELLA's test is a
    # likelihood ratio against its own analytic null -- intensity proportional to
    # r, i.e. radially homogeneous (ELLA.py:54,67) -- so its p-value, its ROC and
    # its pattern labels are computed without the shuffled points ever being
    # read. Stamping `control_fingerprint` on those rows would claim they were
    # scored against that draw, which is exactly the claim
    # `check_control_consistency` is meant to be able to trust.
    #
    # So the fingerprint goes only where the control was actually used: the
    # calibration arm, which is a separate experiment run *on* the shuffled
    # points rather than *against* them.
    det_params = dict(params, control_used=False,
                      null="method_internal")
    rows = gene_table_rows(gene_table, truth, dataset, score_col=score_col,
                           subtype_col=subtype_col, seed=seed, params=det_params,
                           results_dir=results_dir)
    if control_table is not None:
        rows += calibration_rows(control_table,
                                 params=dict(params, control_used=True,
                                             control_fingerprint=control_fingerprint))

    stamped = stamp(rows, run_id=run_id, timestamp=ts, dataset=dataset,
                    subset=subset, method=method, parent_method="",
                    latent_dim=0, code_version=code_version(),
                    run_label=run_label,
                    scale_label=scale_label or infer_scale_label(n_cells),
                    n_cells=n_cells if n_cells is not None else "",
                    n_obs_scored=int(gene_table["gene_name"].nunique()))
    write_shard(stamped, run_id, results_dir=results_dir)
    write_manifest({
        "dataset": dataset, "subset": subset, "method": method,
        "kind": "gene_level", "score_col": score_col,
        "n_genes": int(gene_table["gene_name"].nunique()),
        "n_repeats": int(gene_table["repeat"].nunique())
        if "repeat" in gene_table.columns else 1,
        "control_fingerprint": control_fingerprint,
        "has_control_arm": control_table is not None,
        "run_label": run_label, "scale_label": scale_label or "",
        "n_cells": n_cells,
        "elapsed_s": round(time.time() - t0, 1),
        "params": params,
    }, run_id, results_dir=results_dir)
    return pd.DataFrame(stamped)
