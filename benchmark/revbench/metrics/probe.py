"""Supervised probes and negative controls.

``rf_probe`` replaces
``make_paper_figures/src/benchmark_functions.py::benchmark_classify_patterns``.
``negative_control_rows`` replaces the ``testIfItCanClassifyMetadata`` defined
inline in ``fig2_technical_aspects_of_subcellspace.ipynb`` cell 61.

Differences from the published code, all deliberate:
  * takes X explicitly rather than reading ``adata.X``;
  * ``StratifiedGroupKFold(shuffle=True, random_state=...)`` over ``n_repeats``
    -- the published call left ``shuffle=False``, under which ``random_state``
    is ignored, so it could not have varied even if a seed had been passed;
  * RF ``n_jobs=1`` with parallelism at the CV level instead of ``n_jobs=-1``
    on both, which oversubscribes badly at 220k rows;
  * returns flat MetricRows, one per fold, so mean +/- sd falls out.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn import metrics as skm

from ..config import SEED
from ..results import MetricRow

#: Above this many classes, per-class F1 rows are suppressed.
MAX_PER_CLASS_ROWS = 25

_CV = {"stratified_group": StratifiedGroupKFold,
       "stratified": StratifiedKFold,
       "group": GroupKFold}


def _cpu_budget(n_splits: int) -> int:
    return max(1, min(n_splits, (os.cpu_count() or 2) - 1))


def _splitter(scheme, n_splits, seed):
    if scheme == "group":
        return GroupKFold(n_splits=n_splits)          # has no shuffle/random_state
    return _CV[scheme](n_splits=n_splits, shuffle=True, random_state=seed)


def _labels_nested_in_groups(y, groups) -> bool:
    """True when each group carries exactly one label (label nested in group)."""
    df = pd.DataFrame({"g": groups, "y": y})
    return bool((df.groupby("g")["y"].nunique() == 1).all())


def _shuffle_labels(y, groups, rng):
    """Permute labels to build the chance floor, respecting the CV structure.

    Two cases, and using the wrong one silently corrupts the floor:

    * label *nested* in group (each cell has one label) -- permute the
      group->label mapping. Shuffling rows would let a fold see the group's
      true label through its other rows, making the floor easier than chance.
    * label *crossed* with group (each cell carries every pattern, which is the
      case for ``pattern`` x ``cell_id`` here) -- permute rows. A group-level
      permutation would collapse each cell to one label, leaving folds
      single-class and kappa undefined.
    """
    out = np.asarray(y).copy()
    if groups is None or not _labels_nested_in_groups(y, groups):
        rng.shuffle(out)
        return out
    uniq = np.unique(groups)
    first = pd.Series(y).groupby(pd.Series(groups)).first()
    vals = first.loc[uniq].values.copy()
    rng.shuffle(vals)
    mapping = dict(zip(uniq, vals))
    return np.asarray([mapping[g] for g in groups])


def rf_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    cv_scheme: str = "stratified_group",
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = SEED,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    task: str = "auto",
    label_shuffle: bool = False,
    benchmark: str = "probe",
    target: str = "",
    extra_params: Optional[Dict[str, Any]] = None,
    confusion: bool = False,
) -> List[MetricRow]:
    """Cross-validated random-forest probe, one MetricRow per fold and metric."""
    X = np.asarray(X)
    y = np.asarray(y)
    if cv_scheme not in _CV:
        raise ValueError("cv_scheme must be one of {}".format(sorted(_CV)))
    if cv_scheme in ("stratified_group", "group") and groups is None:
        raise ValueError("cv_scheme={!r} needs groups".format(cv_scheme))

    if task == "auto":
        task = "regression" if (y.dtype.kind == "f" and np.unique(y).size > 20) \
               else "classification"

    rows: List[MetricRow] = []
    for repeat in range(max(1, n_repeats)):
        rs = seed + repeat
        conf = {}          # (true, pred) -> count, accumulated across folds
        rng = np.random.RandomState(rs)
        y_use = _shuffle_labels(y, groups, rng) if label_shuffle else y

        params = {"cv_scheme": cv_scheme, "n_splits": n_splits,
                  "max_depth": max_depth, "n_estimators": n_estimators,
                  "label_shuffle": label_shuffle, "task": task}
        params.update(extra_params or {})

        splitter = _splitter(cv_scheme, n_splits, rs)
        split_args = (X, y_use, groups) if cv_scheme != "stratified" else (X, y_use)
        for fold, (tr, te) in enumerate(splitter.split(*split_args)):
            if task == "classification":
                est = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    random_state=rs, n_jobs=_cpu_budget(n_splits))
            else:
                est = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    random_state=rs, n_jobs=_cpu_budget(n_splits))
            est.fit(X[tr], y_use[tr])
            pred = est.predict(X[te])
            yt = y_use[te]
            p = dict(params, fold=fold)
            n, ng = int(te.size), int(np.unique(groups[te]).size) if groups is not None else 0

            def add(metric, value, class_label=""):
                rows.append(MetricRow(
                    benchmark=benchmark, metric=metric, value=float(value),
                    n=n, n_groups=ng, target=target, class_label=class_label,
                    repeat=repeat, seed=rs, params=p))

            if task == "classification":
                add("accuracy", skm.accuracy_score(yt, pred))
                add("kappa", skm.cohen_kappa_score(yt, pred))
                add("f1_weighted", skm.f1_score(yt, pred, average="weighted", zero_division=0))
                add("f1_macro", skm.f1_score(yt, pred, average="macro", zero_division=0))
                add("precision_weighted", skm.precision_score(yt, pred, average="weighted", zero_division=0))
                add("recall_weighted", skm.recall_score(yt, pred, average="weighted", zero_division=0))
                classes = np.unique(yt)
                # Per-class rows only for low-cardinality targets: `cell_id`
                # has hundreds of levels and would swamp the results table
                # with rows nobody reads.
                if classes.size <= MAX_PER_CLASS_ROWS:
                    per = skm.f1_score(yt, pred, average=None, labels=classes,
                                       zero_division=0)
                    for cls, f1 in zip(classes, per):
                        add("f1", f1, class_label=str(cls))
                else:
                    p["n_classes"] = int(classes.size)
                if confusion and classes.size <= MAX_PER_CLASS_ROWS:
                    # accumulated across folds, emitted once per repeat: in
                    # K-fold every observation is held out exactly once, so a
                    # repeat yields one complete matrix. Emitting per fold
                    # instead would make `summarize` average partial matrices,
                    # which is not a confusion matrix of anything.
                    for t_, p_ in zip(yt, pred):
                        k = (str(t_), str(p_))
                        conf[k] = conf.get(k, 0) + 1
            else:
                add("r2", skm.r2_score(yt, pred))
                add("mse", skm.mean_squared_error(yt, pred))
                add("mae", skm.mean_absolute_error(yt, pred))

        if conf:
            total = sum(conf.values())
            cp = dict(params, matrix="confusion", folds_pooled=n_splits)
            for (t_, p_), c in sorted(conf.items()):
                rows.append(MetricRow(
                    benchmark=benchmark, metric="confusion_count", value=float(c),
                    n=total, n_groups=0, target=target,
                    class_label="%s|%s" % (t_, p_),      # "<true>|<predicted>"
                    repeat=repeat, seed=rs, params=cp))
    return rows


def negative_control_rows(
    X: np.ndarray,
    obs: pd.DataFrame,
    targets: Optional[Dict[str, str]] = None,
    group_key: str = "cell_id",
    include_shuffled: bool = True,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = SEED,
    max_depths: Sequence[Optional[int]] = (5, None),
) -> List[MetricRow]:
    """Can the latent predict things it should not?

    Each target gets the CV scheme it actually requires. ``cell_id`` stays
    ungrouped on purpose: grouping on ``cell_id`` while predicting it makes the
    score 0 by construction -- a vacuous pass, not evidence. Its reading rests
    on the ratio to the shuffled floor, and ``cv_scheme`` is recorded on every
    row so the two can never be conflated.

    ``max_depth`` is swept rather than fixed at the notebook's magic 5: a
    negative control that only passes at one depth is not a negative control.
    """
    from ..config import NEGATIVE_CONTROLS
    targets = targets or NEGATIVE_CONTROLS
    groups = obs[group_key].values if group_key in obs.columns else None

    rows: List[MetricRow] = []
    for target, scheme in targets.items():
        if target not in obs.columns:
            continue
        y = obs[target].values
        if hasattr(y, "categories") or str(getattr(y, "dtype", "")) == "category":
            y = np.asarray(y.astype(str))
        y = np.asarray(y)
        keep = pd.notna(y)
        if keep.sum() < n_splits * 2:
            continue
        g = groups[keep] if (groups is not None and scheme != "stratified") else None
        for depth in max_depths:
            common = dict(cv_scheme=scheme, n_splits=n_splits, n_repeats=n_repeats,
                          seed=seed, max_depth=depth, benchmark="negative_control",
                          target=target)
            rows += rf_probe(X[keep], y[keep], groups=g, **common)
            if include_shuffled:
                # Distinct benchmark name, not just a params flag: sharing the
                # name would let summarize() average the floor into the real
                # number and silently destroy the control.
                floor = dict(common, benchmark="negative_control_floor")
                rows += rf_probe(X[keep], y[keep], groups=g, label_shuffle=True,
                                 **floor)
    return rows
