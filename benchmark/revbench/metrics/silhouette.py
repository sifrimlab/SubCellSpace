"""Silhouette on the latent representation.

Replaces ``make_paper_figures/src/benchmark_functions.py::compute_silhouette``,
which defaults to ``use_rep="X_umap"`` -- i.e. it scores a stochastic 2-D
projection whose ``n_neighbors`` differs between notebooks, not the embedding.
The ``one_vs_rest`` branch is vendored from the ``~/CVAE/src/`` copy, which the
public repo's version is missing.

Silhouette is O(n^2), so subsampling is required at 220k rows regardless;
repeating the subsample turns that cost into the error bar.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.metrics import silhouette_score

from ..config import SEED
from ..results import MetricRow


def _subsample(n, labels, groups, sample_size, sample_by, rng):
    if sample_size is None or sample_size >= n:
        return np.arange(n)
    if sample_by == "groups":
        if groups is None:
            raise ValueError("sample_by='groups' needs groups")
        uniq = np.unique(groups)
        # take whole groups until the row budget is met
        rng.shuffle(uniq)
        keep, total = [], 0
        for g in uniq:
            idx = np.flatnonzero(groups == g)
            keep.append(idx)
            total += idx.size
            if total >= sample_size:
                break
        return np.sort(np.concatenate(keep))
    return np.sort(rng.choice(n, size=sample_size, replace=False))


def silhouette_rows(
    X: np.ndarray,
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    one_vs_rest: bool = True,
    sample_size: Optional[int] = 20000,
    sample_by: str = "rows",
    n_repeats: int = 5,
    seed: int = SEED,
    metric: str = "euclidean",
    target: str = "pattern",
    benchmark: str = "silhouette",
) -> List[MetricRow]:
    """Global silhouette, plus one-vs-rest per class, over repeated subsamples.

    ``sample_by="groups"`` samples whole ``cell_id``s. That matters here
    specifically: the simulated data holds several rotations of the same cell,
    so row-wise sampling puts near-duplicates in one computation and inflates
    cohesion. Register both variants and report them side by side.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X has {} rows, labels {}".format(X.shape[0], labels.shape[0]))
    cats = np.unique(labels[labels == labels])
    if cats.size < 2:
        raise ValueError("need >= 2 classes, got {}".format(cats.size))

    rows: List[MetricRow] = []
    for repeat in range(max(1, n_repeats)):
        rs = seed + repeat
        rng = np.random.RandomState(rs)
        idx = _subsample(X.shape[0], labels, groups, sample_size, sample_by, rng)
        Xs, ys = X[idx], labels[idx]
        params = {"sample_by": sample_by, "sample_size": sample_size,
                  "metric": metric, "n_drawn": int(idx.size)}
        n_groups = int(np.unique(groups[idx]).size) if groups is not None else 0

        if np.unique(ys).size < 2:
            continue
        rows.append(MetricRow(
            benchmark=benchmark, metric="silhouette",
            value=float(silhouette_score(Xs, ys, metric=metric)),
            n=int(idx.size), n_groups=n_groups, target=target,
            repeat=repeat, seed=rs, params=params))

        if one_vs_rest:
            for cat in cats:
                mask = ys == cat
                if mask.sum() < 2 or (~mask).sum() < 2:
                    value = float("nan")
                else:
                    value = float(silhouette_score(Xs, mask.astype(int), metric=metric))
                rows.append(MetricRow(
                    benchmark=benchmark, metric="silhouette_ovr", value=value,
                    n=int(idx.size), n_groups=n_groups, target=target,
                    class_label=str(cat), repeat=repeat, seed=rs, params=params))
    return rows
