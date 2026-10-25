"""Fit the pattern and subtype classifiers on a given representation.

The shipped ``rf_pattern_nonpattern.pkl`` was fitted to one specific latent
space; its 15 features *are* that model's coordinates. A differently-trained
model -- a seed replicate, a leave-one-out model, the no-DAPI ablation --
produces a different coordinate system even at the same dimensionality, so
scoring it with the original classifier measures how well the new embedding
happens to align with the old one rather than how well it separates patterns.
At a different dimensionality reuse is impossible outright. Every method in a
comparison therefore gets its own classifier, including the original model.

The recipes below reproduce the published ones, whose hyperparameters are
recorded in the shipped pickle filenames:

* ``rf_pattern_nonpattern_no_foci_strong_patterns_homemade_random_10-80spots_maxdepth32_maxfeatures5``
* ``multiclass_rf_all_patterns_0.8prop_maxdepth32_maxfeatures5``
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..config import SEED
from ..results import MetricRow
from ..metrics.probe import rf_probe
from .genes import rep_matrix

from ..config import _benchmark
SIM_DIR = _benchmark.DATASETS
TRAIN_REAL = _benchmark.SIMULATED_PATTERNED.name
TRAIN_RANDOM = _benchmark.SIMULATED_SHUFFLED.name

#: Published recipe constants.
DEFAULT_N_SPOTS = (10, 80)
DEFAULT_MIN_PROP = 0.8
DEFAULT_EXCLUDE = ("foci",)
DEFAULT_MAX_DEPTH = 32
DEFAULT_MAX_FEATURES = 5
DEFAULT_N_ESTIMATORS = 100

#: The shipped multiclass classifier carries this class alongside the patterns.
RANDOM_CLASS = "random"


def attach_control_keys(adata, control_set=None, path_col: str = "original_image_paths",
                        out_col: str = "control_key",
                        min_coverage: float = 0.99) -> Dict[str, Any]:
    """Link rows to a ControlSet and record which shuffle they belong to.

    The key is ``Path(obs[path_col]).stem``, which is exactly the ``obs_key``
    that ``controls.extract_from_coord_tree`` derives from the shuffled
    ``gene_coords_zoomed/*.csv`` filenames. Verified: every control key maps to
    a row of the randomised AnnData.
    """
    stems = [Path(str(p)).stem for p in adata.obs[path_col]]
    adata.obs[out_col] = stems
    report = {"n_rows": len(stems)}
    if control_set is None:
        return report
    keys = set(control_set.points["obs_key"].unique())
    hit = np.fromiter((s in keys for s in stems), dtype=bool, count=len(stems))
    report.update({"n_matched": int(hit.sum()),
                   "coverage": float(hit.mean()) if len(stems) else 0.0,
                   "control_fingerprint": control_set.fingerprint})
    if report["coverage"] < min_coverage:
        raise ValueError(
            "only {:.1%} of rows are covered by control {!r} (need {:.0%}). "
            "These rows were shuffled with a different draw, so scores against "
            "this control would not be comparable."
            .format(report["coverage"], control_set.name, min_coverage))
    adata.uns["control_fingerprint"] = control_set.fingerprint
    adata.uns["control_name"] = control_set.name
    return report


def load_training_pair(sim_dir: Path = SIM_DIR, real_name: str = TRAIN_REAL,
                       random_name: str = TRAIN_RANDOM,
                       control_set=None, min_coverage: float = 0.99
                       ) -> Tuple[ad.AnnData, ad.AnnData]:
    """The simulated (real, shuffled) training pair.

    Not row-aligned in the observation sense: ``obs_names`` are positional
    integers, so row *i* of each file is the same template and proportion but a
    *different rotation draw* (the real half indexes ``coordinate_parsed/``,
    the shuffled half ``coordinate_parsed_with_coords_shuffled/``). Harmless
    here -- the binary classifier needs a balanced *set* of patterned and random
    examples, not matched pairs -- but do not treat the two halves as paired.

    The real file is a strict subset of the shuffled one (805454 vs 805549
    rows), so both are intersected on ``obs_names`` and reordered to match --
    the same fix already present in the published training notebook.
    ``random_or_pattern`` is the constant ``"pattern"`` in *both* files, so the
    real/control identity comes from which file a row came from and is written
    to ``obs["origin"]`` here.

    Pass ``control_set`` to bind the shuffled half to a canonical
    :class:`~revbench.controls.ControlSet`. Both halves then carry
    ``obs["control_key"]`` (the image stem) and the shuffled half records the
    fingerprint in ``uns``, so a classifier retrained for another model is
    provably fitted against the same shuffled points.

    Note the two halves are **not** paired per observation -- the real tree and
    the shuffled tree were rotated independently (e.g. ``rotated172`` against
    ``rotated171`` for the same cell). That is fine here: the binary classifier
    needs a balanced *set* of patterned and random examples, not matched pairs.
    """
    real = ad.read_h5ad(Path(sim_dir) / real_name)
    rand = ad.read_h5ad(Path(sim_dir) / random_name)
    shared = real.obs_names.intersection(rand.obs_names)
    real = real[shared].copy()
    rand = rand[shared].copy()
    real.obs["origin"] = "pattern"
    rand.obs["origin"] = "random"
    attach_control_keys(real, None)
    report = attach_control_keys(rand, control_set, min_coverage=min_coverage)
    rand.uns["control_report"] = report
    return real, rand


def _training_mask(obs: pd.DataFrame, n_spots: Sequence[int], min_prop: float,
                   exclude_patterns: Sequence[str]) -> np.ndarray:
    mask = np.ones(len(obs), dtype=bool)
    if "n_spots" in obs.columns and n_spots is not None:
        lo, hi = n_spots
        # Raw counts, deliberately: the stored n_spots_interval labels omit
        # "60-70", so every label from the 7th bin up is shifted a decade.
        mask &= (obs["n_spots"].values >= lo) & (obs["n_spots"].values <= hi)
    if "prop" in obs.columns and min_prop is not None:
        mask &= obs["prop"].values >= min_prop
    if exclude_patterns and "pattern" in obs.columns:
        mask &= ~obs["pattern"].astype(str).isin(list(exclude_patterns)).values
    return mask


def _balance_by(obs: pd.DataFrame, key: str, seed: int) -> np.ndarray:
    """Indices giving every level of ``key`` the size of its smallest level."""
    rng = np.random.RandomState(seed)
    labels = obs[key].astype(str).values
    n = min(int((labels == lv).sum()) for lv in np.unique(labels))
    keep = []
    for lv in np.unique(labels):
        idx = np.flatnonzero(labels == lv)
        keep.append(rng.choice(idx, size=n, replace=False) if idx.size > n else idx)
    return np.sort(np.concatenate(keep))


QC_MAX_ROWS = 40_000


def _qc_rows(X, y, groups, benchmark, seed, n_splits, n_repeats, params,
             qc_max_rows=None):
    # The full balanced training set is ~180k rows; refitting it once per fold
    # costs minutes for a number that is stable at a fraction of the size.
    # That subsample is a demo-scale cost control, so a caller that explicitly
    # wants the full-set QC passes its own cap; `qc_subsampled_to` appears in
    # params only when the cap actually bit, which is how a reader tells a
    # full-set QC from a capped one.
    cap = QC_MAX_ROWS if qc_max_rows is None else int(qc_max_rows)
    if X.shape[0] > cap:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(X.shape[0], size=cap, replace=False))
        X, y = X[idx], y[idx]
        groups = None if groups is None else groups[idx]
        params = dict(params, qc_subsampled_to=cap)
    return rf_probe(X, y, groups=groups,
                    cv_scheme="stratified_group" if groups is not None else "stratified",
                    n_splits=n_splits, n_repeats=n_repeats, seed=seed,
                    max_depth=params.get("max_depth"),
                    n_estimators=params.get("n_estimators", DEFAULT_N_ESTIMATORS),
                    benchmark=benchmark, target=params.get("target", ""),
                    extra_params=params,
                    # the held-out confusion matrix of this method's own
                    # classifier: pattern-vs-random for the binary one, the
                    # subtype grid for the multiclass one
                    confusion=True)


def train_pattern_classifier(
    real, control, rep: Optional[str] = None,
    n_spots: Sequence[int] = DEFAULT_N_SPOTS,
    min_prop: float = DEFAULT_MIN_PROP,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE,
    max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
    max_features: Optional[int] = DEFAULT_MAX_FEATURES,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    balance: bool = True, seed: int = SEED, n_jobs: int = 1,
    fit_n_jobs: int = -1,
    qc: bool = True, qc_splits: int = 3, qc_repeats: int = 1,
    qc_max_rows: Optional[int] = None,
) -> Tuple[RandomForestClassifier, List[MetricRow]]:
    """Binary pattern-vs-random forest on ``rep``.

    ``fit_n_jobs`` parallelises fitting; ``n_jobs`` is what the returned
    estimator carries, and defaults to 1 because ``predict_proba`` is called
    once per gene inside ``classify_all_genes``, where a multi-job forest is
    slower -- which is why the published code keeps a ``_singlejob`` pickle.
    """
    mr = _training_mask(real.obs, n_spots, min_prop, exclude_patterns)
    mc = _training_mask(control.obs, n_spots, min_prop, exclude_patterns)
    Xr, Xc = rep_matrix(real, rep)[mr], rep_matrix(control, rep)[mc]
    obs = pd.concat([real.obs[mr], control.obs[mc]], ignore_index=True)
    X = np.vstack([Xr, Xc])
    obs["random_or_pattern"] = obs["origin"].astype(str)

    if balance:
        idx = _balance_by(obs, "random_or_pattern", seed)
        X, obs = X[idx], obs.iloc[idx].reset_index(drop=True)

    y = obs["random_or_pattern"].values
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
        bootstrap=True, oob_score=False, random_state=seed, n_jobs=fit_n_jobs)
    clf.fit(X, y)
    # Fit in parallel, predict single-threaded: predict_proba runs once per gene
    # inside classify_all_genes, where a multi-job forest is slower.
    clf.n_jobs = n_jobs

    params = {"recipe": "pattern_nonpattern", "n_spots": list(n_spots),
              "min_prop": min_prop, "exclude_patterns": list(exclude_patterns),
              "max_depth": max_depth, "max_features": max_features,
              "n_estimators": n_estimators, "balanced": balance,
              "n_train": int(X.shape[0]), "latent_dim": int(X.shape[1]),
              "target": "random_or_pattern"}
    rows: List[MetricRow] = []
    if qc:
        groups = obs["cell_id"].astype(str).values if "cell_id" in obs.columns else None
        rows = _qc_rows(X, y, groups, "pattern_classifier_qc", seed,
                        qc_splits, qc_repeats, params,
                        qc_max_rows=qc_max_rows)
    return clf, rows


def train_subtype_classifier(
    real, control=None, rep: Optional[str] = None, min_prop: float = DEFAULT_MIN_PROP,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE,
    max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
    max_features: Optional[int] = DEFAULT_MAX_FEATURES,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    balance: bool = True, seed: int = SEED, n_jobs: int = 1,
    fit_n_jobs: int = -1,
    qc: bool = True, qc_splits: int = 3, qc_repeats: int = 1,
    qc_max_rows: Optional[int] = None,
) -> Tuple[RandomForestClassifier, List[MetricRow]]:
    """Multiclass subtype forest on ``rep``.

    ``control`` is optional but should normally be given: the shipped
    ``multiclass_rf.pkl`` carries a ``random`` class alongside the six
    patterns, and the published subtype assignment
    (``classifiers.py::classify_all_genes_for_pattern_subtypes``) does
    ``predict_proba(...).drop("random", axis=1)`` -- so a classifier trained on
    patterns alone raises ``KeyError: ['random'] not found in axis`` there.
    Passing the shuffled control reproduces the published class set.
    """
    mask = _training_mask(real.obs, None, min_prop, exclude_patterns)
    X = rep_matrix(real, rep)[mask]
    obs = real.obs[mask].reset_index(drop=True)
    if control is not None:
        cmask = _training_mask(control.obs, None, min_prop, exclude_patterns)
        cobs = control.obs[cmask].reset_index(drop=True).copy()
        cobs["pattern"] = RANDOM_CLASS
        X = np.vstack([X, rep_matrix(control, rep)[cmask]])
        obs = pd.concat([obs, cobs], ignore_index=True)
    if balance:
        idx = _balance_by(obs, "pattern", seed)
        X, obs = X[idx], obs.iloc[idx].reset_index(drop=True)
    y = obs["pattern"].astype(str).values

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
        bootstrap=True, oob_score=False, random_state=seed, n_jobs=fit_n_jobs)
    clf.fit(X, y)
    # Fit in parallel, predict single-threaded: predict_proba runs once per gene
    # inside classify_all_genes, where a multi-job forest is slower.
    clf.n_jobs = n_jobs

    params = {"recipe": "subtype_multiclass", "min_prop": min_prop,
              "exclude_patterns": list(exclude_patterns), "max_depth": max_depth,
              "max_features": max_features, "n_estimators": n_estimators,
              "balanced": balance, "n_train": int(X.shape[0]),
              "latent_dim": int(X.shape[1]), "n_classes": int(np.unique(y).size),
              "target": "pattern"}
    rows: List[MetricRow] = []
    if qc:
        groups = obs["cell_id"].astype(str).values if "cell_id" in obs.columns else None
        rows = _qc_rows(X, y, groups, "subtype_classifier_qc", seed,
                        qc_splits, qc_repeats, params,
                        qc_max_rows=qc_max_rows)
    return clf, rows
