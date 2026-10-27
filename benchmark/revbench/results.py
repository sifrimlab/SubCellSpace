"""The tidy results contract: one row per (benchmark, metric, repeat, class).

Results are written as CSV shards, one per ``run_suite`` call, then compacted
into ``results/benchmarks.csv``. CSV rather than parquet because being able to
``git diff`` the number behind a figure is worth more than columnar I/O on a
table with fewer than 1e5 rows.
"""
from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, code_version, env_versions

SCHEMA = 2

#: Column order of every results file. Identity columns first, then the
#: measurement, then provenance.
#:
#: ``scale_label`` / ``n_cells`` / ``n_obs_scored`` exist because benchmarks are
#: developed on subsamples and finished on the full data, and once a number is
#: in a table the difference is invisible. A demo row and a full-dataset row
#: must never be silently comparable, so the scale is a first-class column that
#: ``summarize`` groups on rather than a detail buried in ``params_json``.
#: ``run_label`` names the notebook run that produced the row, so two runs of
#: the same notebook can be diffed against each other.
COLUMNS: List[str] = [
    "run_id", "timestamp", "run_label", "dataset", "subset",
    "scale_label", "n_cells", "n_obs_scored",
    "method", "parent_method", "latent_dim",
    "benchmark", "target", "metric", "class_label",
    "value", "n", "n_groups", "repeat", "seed",
    "params_json", "code_version",
]

#: Rows written before scale stamping existed (schema 1) carry no scale. They
#: are labelled rather than left blank, so an unlabelled row is visibly
#: unlabelled instead of looking like a full-scale one.
UNKNOWN_SCALE = "unlabelled"


@dataclasses.dataclass(frozen=True)
class MetricRow:
    """One measurement. Identity columns are stamped on later by the suite."""
    benchmark: str
    metric: str
    value: float
    n: int
    n_groups: int = 0
    target: str = ""
    class_label: str = ""
    repeat: int = -1
    seed: int = -1
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["params_json"] = json.dumps(d.pop("params"), sort_keys=True, default=str)
        return d


def infer_scale_label(n_cells) -> str:
    """A scale label when the caller did not supply one.

    Never returns "": an unlabelled row must *look* unlabelled, because the whole
    point of the column is that a demo row cannot pass for a full one.
    """
    return UNKNOWN_SCALE if n_cells is None else "n_cells=%d" % int(n_cells)


def new_run_id() -> str:
    return uuid.uuid4().hex[:8]


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stamp(rows: Sequence[MetricRow], **identity: Any) -> List[Dict[str, Any]]:
    """Attach run/method identity to primitive rows and normalise the schema."""
    out = []
    version = identity.pop("code_version", None) or code_version()
    identity.setdefault("scale_label", UNKNOWN_SCALE)
    identity.setdefault("run_label", "")
    for row in rows:
        d = row.as_dict()
        d.update(identity)
        d["code_version"] = version
        out.append({col: d.get(col, "") for col in COLUMNS})
    return out


def _runs_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "runs"


def write_shard(rows: Sequence[Dict[str, Any]], run_id: str,
                results_dir: Path = RESULTS_DIR) -> Path:
    """Write one run's rows. Shards avoid concurrent-append corruption and make
    re-running a broken benchmark a matter of deleting one file."""
    runs = _runs_dir(results_dir)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{run_id}.csv"
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    df.to_csv(path, index=False)
    return path


def write_manifest(manifest: Dict[str, Any], run_id: str,
                   results_dir: Path = RESULTS_DIR) -> Path:
    runs = _runs_dir(results_dir)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{run_id}.json"
    payload = dict(manifest)
    payload.setdefault("schema", SCHEMA)
    payload.setdefault("run_id", run_id)
    payload.setdefault("env", env_versions())
    payload.setdefault("code_version", code_version())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def load_all(results_dir: Path = RESULTS_DIR,
             include_compacted: bool = True) -> pd.DataFrame:
    """Every result row ever written, shards plus the compacted file."""
    frames = []
    compacted = Path(results_dir) / "benchmarks.csv"
    if include_compacted and compacted.is_file():
        frames.append(pd.read_csv(compacted))
    runs = _runs_dir(results_dir)
    if runs.is_dir():
        for path in sorted(runs.glob("*.csv")):
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    for f in frames:
        # schema-1 shards predate scale stamping; say so instead of leaving NaN,
        # which would silently group with whatever else is missing a label
        if "scale_label" not in f.columns:
            f["scale_label"] = UNKNOWN_SCALE
        for c in ("run_label", "n_cells", "n_obs_scored"):
            if c not in f.columns:
                f[c] = ""
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=[c for c in COLUMNS if c != "timestamp"])


def compact(results_dir: Path = RESULTS_DIR, keep_shards: bool = True) -> Path:
    """Fold the shards into ``benchmarks.csv``. Explicit, so the cited file
    only changes when you mean it to."""
    df = load_all(results_dir)
    out = Path(results_dir) / "benchmarks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if not keep_shards:
        for path in _runs_dir(results_dir).glob("*.csv"):
            path.unlink()
    return out


#: ``scale_label`` is in the grouping deliberately: averaging a demo row and a
#: full-dataset row into one mean is the exact mistake the column exists to
#: prevent, and a silent mean is worse than two rows.
DEFAULT_GROUPING = ("dataset", "subset", "scale_label", "benchmark", "method",
                    "target", "metric", "class_label")


def summarize(df: pd.DataFrame,
              by: Sequence[str] = DEFAULT_GROUPING) -> pd.DataFrame:
    """Mean +/- sd across repeats.

    ``sd`` is NaN when a single repeat was run, never 0 -- a silent zero error
    bar in a figure is worse than a missing one.
    """
    if df.empty:
        return df
    by = [c for c in by if c in df.columns]
    grouped = df.groupby(by, dropna=False)["value"]
    out = grouped.agg(mean="mean", sd="std", n_repeats="count",
                      vmin="min", vmax="max").reset_index()
    out.loc[out["n_repeats"] < 2, "sd"] = np.nan
    out["sem"] = out["sd"] / np.sqrt(out["n_repeats"])
    out["ci95_lo"] = out["mean"] - 1.96 * out["sem"]
    out["ci95_hi"] = out["mean"] + 1.96 * out["sem"]
    return out
