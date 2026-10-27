"""``run_suite``: the notebook-facing entry point."""
from __future__ import annotations

import dataclasses
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import benchmarks_builtin  # noqa: F401  (registers the built-ins)
from .config import (GROUP_KEY, LABEL_KEY, RESULTS_DIR, SEED,
                     code_version, derive_metadata_columns, env_versions)
from .registry import BENCHMARK_META, BENCHMARKS, applicable
from .representations import Representation, common_valid
from .results import (MetricRow, infer_scale_label, new_run_id, stamp, utcnow,
                      write_manifest, write_shard)


@dataclasses.dataclass
class BenchmarkContext:
    """Everything a benchmark needs, already restricted to the scored rows."""
    rep: Representation
    reps: Dict[str, Representation]
    obs: pd.DataFrame
    row_mask: np.ndarray
    dataset: str
    subset: str = "val"
    group_key: str = GROUP_KEY
    label_key: str = LABEL_KEY
    seed: int = SEED
    n_repeats: int = 3
    n_splits: int = 5
    sample_size: Optional[int] = 20000
    smoke: bool = False
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Where a benchmark that writes a side artefact (the subtype crosstab CSV)
    # must put it. Without this a benchmark reached for the module-level
    # RESULTS_DIR and wrote outside the results_dir its caller asked for.
    results_dir: Path = RESULTS_DIR

    @property
    def X(self) -> np.ndarray:
        return self.rep.X[self.row_mask]


def _subsample_mask(obs: pd.DataFrame, mask: np.ndarray, n: int,
                    label_key: str, group_key: str, seed: int) -> np.ndarray:
    """Cut to ~n rows, keeping whole groups and label balance."""
    idx = np.flatnonzero(mask)
    if n is None or idx.size <= n:
        return mask
    rng = np.random.RandomState(seed)
    sub = obs.iloc[idx]
    if group_key in sub.columns:
        groups = sub[group_key].values
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        keep, total = [], 0
        for g in uniq:
            sel = idx[groups == g]
            keep.append(sel)
            total += sel.size
            if total >= n:
                break
        chosen = np.concatenate(keep)
    else:
        chosen = rng.choice(idx, size=n, replace=False)
    out = np.zeros_like(mask)
    out[chosen] = True
    return out


def run_suite(
    adata,
    representations: Sequence[Representation],
    dataset: str,
    subset: str = "val",
    benchmarks: Optional[Sequence[str]] = None,
    obs_filter: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    group_key: str = GROUP_KEY,
    label_key: str = LABEL_KEY,
    n_repeats: int = 3,
    n_splits: int = 5,
    seed: int = SEED,
    sample_size: Optional[int] = 20000,
    subsample: Optional[int] = None,
    smoke: bool = False,
    strict_coverage: float = 0.95,
    extra: Optional[Dict[str, Any]] = None,
    results_dir: Path = RESULTS_DIR,
    dry_run: bool = False,
    continue_on_error: bool = True,
    run_label: str = "",
    scale_label: str = "",
    n_cells: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Score every representation on every applicable benchmark.

    All representations are scored on the *same* rows -- the intersection of
    their valid masks -- so a method covering fewer observations cannot look
    better by being evaluated on an easier subset. ``strict_coverage`` makes a
    collapsed intersection fail loudly; in particular an attempted row join
    between two methods whose observations differ will trip it rather than
    silently comparing 0.9% of the data.

    ``smoke=True`` is a single switch shrinking every knob, so "runnable on a
    subsample" is structural rather than ten separate promises.
    """
    t0 = time.time()
    if smoke:
        subsample = subsample or 2000
        n_repeats, n_splits, sample_size = 1, 3, min(sample_size or 1000, 1000)

    obs = derive_metadata_columns(adata.obs)

    mask = common_valid(list(representations))
    n_valid_each = {r.name: int(r.valid.sum()) for r in representations}
    largest = max(n_valid_each.values()) if n_valid_each else 0

    if obs_filter is None and subset in ("train", "val") and "train_or_val" in obs.columns:
        obs_filter = lambda o: (o["train_or_val"] == subset).values  # noqa: E731
    if obs_filter is not None:
        mask = mask & np.asarray(obs_filter(obs), dtype=bool)

    if subsample:
        mask = _subsample_mask(obs, mask, subsample, label_key, group_key, seed)

    n_scored = int(mask.sum())
    coverage = n_scored / largest if largest else 0.0
    if largest and not subsample and obs_filter is None and coverage < strict_coverage:
        raise ValueError(
            "scored rows ({}) are only {:.1%} of the largest representation ({}). "
            "If you are comparing methods whose observations differ, do not row-join "
            "them -- score each with representations.standalone() on a matched "
            "cell_id split instead.".format(n_scored, coverage, largest))
    if n_scored == 0:
        raise ValueError("no rows left to score")

    names = benchmarks or applicable(dataset)
    unknown = [b for b in names if b not in BENCHMARKS]
    if unknown:
        raise KeyError("unknown benchmarks: {} (have {})"
                       .format(unknown, sorted(BENCHMARKS)))

    plan = [(rep.name, b) for rep in representations for b in names]
    if dry_run:
        print("rows scored: {} of {} ({:.1%} of largest rep)".format(
            n_scored, adata.n_obs, coverage))
        for m, b in plan:
            print("  {:<28s} {}".format(m, b))
        return pd.DataFrame()

    run_id = new_run_id()
    ts = utcnow()
    version = code_version()
    rep_map = {r.name: r for r in representations}
    obs_scored = obs.iloc[np.flatnonzero(mask)].reset_index(drop=True)
    all_rows: List[Dict[str, Any]] = []

    for rep in representations:
        ctx = BenchmarkContext(
            rep=rep, reps=rep_map, obs=obs_scored, row_mask=mask,
            dataset=dataset, subset=subset, group_key=group_key,
            label_key=label_key, seed=seed, n_repeats=n_repeats,
            n_splits=n_splits, sample_size=sample_size, smoke=smoke,
            extra=dict(extra or {}), results_dir=results_dir)
        for name in names:
            meta = BENCHMARK_META[name]
            missing = [k for k in meta["requires"] if k not in ctx.extra]
            if missing:
                continue
            try:
                rows = BENCHMARKS[name](ctx)
                # Run-wide facts that belong on every row: a threshold applied
                # by pre-filtering the input (min_obs) is otherwise invisible in
                # the store, because a benchmark can only report what it was
                # handed. setdefault, so a benchmark's own key always wins.
                if extra_params:
                    for r in rows:
                        for k, v in extra_params.items():
                            r.params.setdefault(k, v)
            except Exception as exc:  # noqa: BLE001
                if not continue_on_error:
                    raise
                rows = [MetricRow(benchmark=name, metric="error", value=float("nan"),
                                  n=n_scored, params={"error": repr(exc),
                                                      "traceback": traceback.format_exc()[-800:]})]
            all_rows += stamp(
                rows, run_id=run_id, timestamp=ts, dataset=dataset, subset=subset,
                method=rep.name, parent_method=rep.parent or "",
                latent_dim=rep.latent_dim, code_version=version,
                run_label=run_label, scale_label=scale_label or infer_scale_label(n_cells),
                n_cells=n_cells if n_cells is not None else "",
                n_obs_scored=n_scored)

    write_shard(all_rows, run_id, results_dir=results_dir)
    write_manifest({
        "dataset": dataset, "subset": subset, "benchmarks": list(names),
        "n_scored": n_scored, "coverage_of_largest": coverage,
        "valid_per_rep": n_valid_each, "smoke": smoke, "seed": seed,
        "n_repeats": n_repeats, "n_splits": n_splits, "subsample": subsample,
        "sample_size": sample_size, "elapsed_s": round(time.time() - t0, 1),
        "run_label": run_label, "scale_label": scale_label or infer_scale_label(n_cells),
        "n_cells": n_cells,
        "representations": {r.name: {"source": r.source, "latent_dim": r.latent_dim,
                                     "coverage": r.coverage, "parent": r.parent,
                                     "params": r.params}
                            for r in representations},
    }, run_id, results_dir=results_dir)
    return pd.DataFrame(all_rows)
