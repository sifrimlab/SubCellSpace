"""Registering, dumping, loading and aligning embeddings.

Canonical convention: an embedding lives in ``adata.obsm[<method_name>]``.
Nothing downstream of this module reads ``adata.X`` or ``adata.obsm["latent"]``.

A warning about cross-method row joins
--------------------------------------
The SubCellSpace simulated AnnData and the PointNet coordinate directory both
describe ~220k observations built from the same 317 cells and ~74k templates
(72,054 shared, 97%), but their rotation instances were drawn independently:
only 1,927 of 220,650 observations (0.9%) share a ``(template, rotation)`` key.
A per-observation join between those two methods is therefore NOT valid.
Compare them by pinning the same ``cell_id`` split and scoring each on its own
observations -- see ``standalone()``. ``align()`` will surface the problem
anyway, by reporting a coverage near zero.
"""
from __future__ import annotations

import dataclasses
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DUMP_SCHEMA = 1

#: Legal values for ``index_kind``.
INDEX_KINDS = ("obs_names", "canonical_key", "template", "cell_gene", "image_stem")

# Two filename dialects for the same simulated observation:
#   CVAE     cell-edge_0it_100spots_icell149_0it_prop0.90_gene_expression_rotated124_blurred1.5
#   PointNet cell-edge_0it_100spots_icell149_0it_prop0.90_coords_rotated19
_CVAE_RE = re.compile(r"^(?P<template>.+?)_gene_expression_rotated(?P<rot>\d+)")
_POINTCLOUD_RE = re.compile(r"^(?P<template>.+?)_coords_rotated(?P<rot>\d+)")


def parse_stem(stem: str) -> Optional[Tuple[str, int]]:
    """``(template, rotation)`` from either filename dialect, else None."""
    for pattern in (_CVAE_RE, _POINTCLOUD_RE):
        m = pattern.match(stem)
        if m:
            return m.group("template"), int(m.group("rot"))
    return None


def canonical_keys(stems: Sequence[str], kind: str = "canonical_key") -> np.ndarray:
    """Normalise either dialect to a comparable key.

    ``canonical_key`` -> ``"<template>__rot<angle>"`` (one observation)
    ``template``      -> ``"<template>"``            (one simulated cell/gene image)
    """
    if kind not in ("canonical_key", "template"):
        raise ValueError("kind must be 'canonical_key' or 'template'")
    out = []
    for stem in stems:
        parsed = parse_stem(str(stem))
        if parsed is None:
            out.append(str(stem))
            continue
        template, rot = parsed
        out.append(template if kind == "template" else "{}__rot{}".format(template, rot))
    return np.asarray(out, dtype=object)


def adata_keys(adata, index_kind: str, join_key: Optional[str] = None) -> pd.Index:
    """The join keys for an AnnData under a given ``index_kind``."""
    if index_kind == "obs_names":
        return pd.Index(adata.obs_names.astype(str))
    if index_kind == "cell_gene":
        need = ("original_obs_index", "original_gene_name")
        missing = [c for c in need if c not in adata.obs.columns]
        if missing:
            raise KeyError("cell_gene keys need obs columns {}".format(missing))
        return pd.Index(adata.obs["original_obs_index"].astype(str)
                        + "_" + adata.obs["original_gene_name"].astype(str))
    if index_kind == "image_stem":
        # The plain filename stem of obs["original_image_paths"]. This is the
        # natural key for simulated data: the coordinate files a point model
        # reads are named "<stem>_gene-coords.csv", so a dump keyed this way
        # joins straight onto the AnnData with no normalisation.
        col = join_key or "original_image_paths"
        if col not in adata.obs.columns:
            raise KeyError("index_kind='image_stem' needs obs[{!r}]".format(col))
        return pd.Index([Path(str(p)).stem for p in adata.obs[col]])
    if index_kind in ("canonical_key", "template"):
        col = join_key or "original_image_paths"
        if col not in adata.obs.columns:
            raise KeyError(
                "index_kind={!r} needs obs[{!r}]; the simulated AnnData's "
                "obs_names are positional integers and cannot be joined on."
                .format(index_kind, col))
        stems = [Path(str(p)).stem for p in adata.obs[col]]
        return pd.Index(canonical_keys(stems, kind=index_kind))
    raise ValueError("index_kind must be one of {}".format(INDEX_KINDS))


# --- the representation ------------------------------------------------------

@dataclasses.dataclass
class Representation:
    """An embedding aligned to a host AnnData (or standing on its own).

    ``X`` is always full length and in the host's row order; missing rows are
    NaN and flagged in ``valid``. Rows are dropped once, by the suite, on the
    intersection across every method -- never at ingest, because dropping early
    is what makes cross-method numbers silently incomparable.
    """
    name: str
    X: np.ndarray
    valid: np.ndarray
    source: str
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    parent: Optional[str] = None
    obs: Optional[pd.DataFrame] = None   # only set for standalone reps

    @property
    def latent_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0

    def __repr__(self) -> str:
        return ("Representation(name={!r}, shape={}, valid={}/{}, source={!r})"
                .format(self.name, self.X.shape, int(self.valid.sum()),
                        self.valid.size, self.source))


def compute_valid(X: np.ndarray, treat_zero_as_missing: bool = True) -> np.ndarray:
    """Rows that carry a real measurement.

    An all-zero row is the documented encoding for "outside the
    ``10 < n_spots < 101`` window", i.e. not measured -- not a bug. This rule
    lives here so that every method inherits it, including methods that have no
    concept of a spot-count window. It is currently expressed only implicitly,
    as ``nonzero_mask`` inside
    ``make_paper_figures/src/adata_utils.py::reshape_and_filter_anndata_with_index_and_mapping``.
    """
    X = np.asarray(X)
    finite = np.isfinite(X).all(axis=1)
    if treat_zero_as_missing:
        return finite & ~(X == 0).all(axis=1)
    return finite


def _make(name, X, source, params, treat_zero_as_missing, parent=None, obs=None):
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    if X.ndim != 2:
        raise ValueError("embedding must be 2-D, got shape {}".format(X.shape))
    valid = compute_valid(X, treat_zero_as_missing)
    params = dict(params or {})
    params.setdefault("treat_zero_as_missing", treat_zero_as_missing)
    return Representation(name=name, X=X, valid=valid, source=source,
                          params=params, parent=parent, obs=obs)


def from_adata(adata, spec: str, name: Optional[str] = None,
               treat_zero_as_missing: bool = True) -> Representation:
    """Register an embedding already in the AnnData.

    ``spec`` is ``"obsm:<key>"``, ``"layer:<key>"`` or ``"X"``. The ``"X"``
    form exists because the simulated file has no layers at all: its ``X`` is
    already the (220687, 15) latent and ``obsm["latent"]`` is the same array.
    """
    if spec == "X":
        X = adata.X
    elif spec.startswith("obsm:"):
        key = spec.split(":", 1)[1]
        if key not in adata.obsm:
            raise KeyError("obsm[{!r}] not found; have {}"
                           .format(key, list(adata.obsm.keys())))
        X = adata.obsm[key]
    elif spec.startswith("layer:"):
        key = spec.split(":", 1)[1]
        if key not in adata.layers:
            raise KeyError("layers[{!r}] not found; have {}"
                           .format(key, list(adata.layers.keys())))
        X = adata.layers[key]
        if np.asarray(X).ndim == 3:
            raise ValueError(
                "layers[{!r}] is 3-D (n_cells, n_genes, d). Explode it first "
                "with make_paper_figures/src/adata_utils.py::"
                "reshape_and_filter_anndata_with_index_and_mapping, then "
                "register the result.".format(key))
    else:
        raise ValueError("spec must be 'X', 'obsm:<key>' or 'layer:<key>'")
    if hasattr(X, "toarray"):
        X = X.toarray()
    return _make(name or spec.replace(":", "_"), X, "adata:" + spec, {},
                 treat_zero_as_missing)


def standalone(X: np.ndarray, obs: pd.DataFrame, name: str,
               source: str = "standalone",
               treat_zero_as_missing: bool = True,
               params: Optional[Dict[str, Any]] = None) -> Representation:
    """A representation with its own observations and no host AnnData.

    This is the correct vehicle for a method whose observations are not the
    same rows as SubCellSpace's -- the PointNet-VAE above all. Score it on a
    split defined by the same ``cell_id`` partition and compare the metrics,
    rather than forcing a row join that does not exist.
    """
    if len(obs) != np.asarray(X).shape[0]:
        raise ValueError("obs has {} rows, X has {}".format(len(obs), np.asarray(X).shape[0]))
    return _make(name, X, source, params, treat_zero_as_missing,
                 obs=obs.reset_index(drop=True))


def attach(adata, rep: Representation) -> None:
    """Write the representation into ``adata.obsm[rep.name]``."""
    if rep.obs is not None:
        raise ValueError("standalone representations have no host AnnData")
    if rep.X.shape[0] != adata.n_obs:
        raise ValueError("representation has {} rows, adata has {}"
                         .format(rep.X.shape[0], adata.n_obs))
    adata.obsm[rep.name] = rep.X


def common_valid(reps: Sequence[Representation]) -> np.ndarray:
    """Rows valid in every representation.

    The suite computes this once and every benchmark runs on that identical
    row set, so a method that covers fewer rows cannot look better by being
    scored on an easier subset.
    """
    if not reps:
        raise ValueError("no representations given")
    lengths = {r.X.shape[0] for r in reps}
    if len(lengths) != 1:
        raise ValueError("representations have differing lengths: {}".format(lengths))
    mask = np.ones(reps[0].X.shape[0], dtype=bool)
    for rep in reps:
        mask &= rep.valid
    return mask


# --- on-disk dumps -----------------------------------------------------------

def write_dump(path, X: np.ndarray, obs_index: Sequence[str], index_kind: str,
               method: str, dataset: str,
               extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``<name>.npz`` (arrays) plus ``<name>.meta.json`` (provenance).

    npz because a foreign method's env is usually bare numpy+torch and needs no
    extra dependency to write one; the string index is stored as fixed-width
    unicode so the read side can set ``allow_pickle=False``. Metadata goes in a
    sidecar JSON so it stays diffable and committable while the array does not.
    """
    if index_kind not in INDEX_KINDS:
        raise ValueError("index_kind must be one of {}".format(INDEX_KINDS))
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    keys = np.asarray([str(k) for k in obs_index], dtype=np.str_)
    if keys.shape[0] != X.shape[0]:
        raise ValueError("obs_index has {} entries, X has {} rows"
                         .format(keys.shape[0], X.shape[0]))
    np.savez_compressed(path, X=X, obs_index=keys)
    meta = {
        "schema": DUMP_SCHEMA, "method": method, "dataset": dataset,
        "index_kind": index_kind, "n": int(X.shape[0]),
        "latent_dim": int(X.shape[1]),
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    meta.update(extra or {})
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str))
    return path


def read_dump(path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        X = np.asarray(npz["X"], dtype=np.float32)
        keys = np.asarray([str(k) for k in npz["obs_index"]], dtype=object)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return X, keys, meta


def align(adata, X: np.ndarray, obs_index: Sequence[str], index_kind: str,
          on_missing: str = "nan", join_key: Optional[str] = None
          ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Place dump rows into the host's row order.

    Duplicate keys on either side always raise: a repeated key means the same
    observation was embedded twice and there is no correct answer.
    """
    left = adata_keys(adata, index_kind, join_key)
    right = pd.Index(np.asarray([str(k) for k in obs_index], dtype=object))
    X = np.asarray(X, dtype=np.float32)

    if left.has_duplicates:
        dup = left[left.duplicated()].unique()[:3].tolist()
        raise ValueError("duplicate join keys in adata under index_kind={!r}: {}"
                         .format(index_kind, dup))
    if right.has_duplicates:
        dup = right[right.duplicated()].unique()[:3].tolist()
        raise ValueError("duplicate join keys in dump: {}".format(dup))

    pos = pd.Series(np.arange(len(right)), index=right)
    matched = pos.reindex(left)
    found = matched.notna().values
    n_matched = int(found.sum())

    report = {
        "index_kind": index_kind,
        "n_left": int(len(left)), "n_right": int(len(right)),
        "n_matched": n_matched,
        "n_missing": int(len(left) - n_matched),
        "n_extra": int(len(right) - n_matched),
        "coverage": float(n_matched / len(left)) if len(left) else 0.0,
    }

    if on_missing == "error" and report["n_missing"]:
        raise ValueError("{} of {} host rows unmatched (coverage {:.3f})"
                         .format(report["n_missing"], len(left), report["coverage"]))
    if report["n_extra"]:
        warnings.warn("{} dump rows not present in the host AnnData; dropped"
                      .format(report["n_extra"]), stacklevel=2)
    if report["coverage"] < 0.05:
        warnings.warn(
            "coverage is only {:.1%}. If these are two different methods' "
            "simulated observations, a per-observation join is not valid -- "
            "see the module docstring and use standalone()."
            .format(report["coverage"]), stacklevel=2)

    out = np.full((len(left), X.shape[1]), np.nan, dtype=np.float32)
    idx = matched.values[found].astype(int)
    out[found] = X[idx]
    return out, found, report


def from_dump(adata, path, name: Optional[str] = None,
              index_kind: Optional[str] = None, on_missing: str = "nan",
              join_key: Optional[str] = None,
              treat_zero_as_missing: bool = True) -> Representation:
    """Load a dump and align it to ``adata``. Use only when the dump describes
    the same observations as the host; otherwise use ``standalone()``."""
    X, keys, meta = read_dump(path)
    kind = index_kind or meta.get("index_kind") or "obs_names"
    aligned, found, report = align(adata, X, keys, kind,
                                   on_missing=on_missing, join_key=join_key)
    params = {"align": report}
    params.update({k: v for k, v in meta.items() if k not in ("schema",)})
    rep = _make(name or meta.get("method") or Path(path).stem, aligned,
                str(path), params, treat_zero_as_missing)
    rep.valid = rep.valid & found
    return rep
