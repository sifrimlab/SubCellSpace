"""The shared randomised control.

Every method in a comparison must be scored against the *same* shuffled points.
A given draw can, by chance, look patterned; if each method shuffles for itself,
part of any measured difference is just which accidental arrangements that
method happened to draw. Since we compare image-based and point-based models --
which consume different representations of the same underlying points -- the
point set is stored once here and served to both.

This is not only a fairness preference. The shuffle is drawn with
``np.random.choice`` and **no seed is set anywhere in its call path**
(``make_paper_figures/src/shuffle.py:24``,
``CVAE/source_functions/randomization.py:37``), so the published control cannot
be regenerated. The stored coordinates are the only source of truth.

Two caveats this module records rather than hides:

* ``np.random.choice`` is called *with replacement*, so a shuffled cloud may
  contain duplicate pixels (measured: 6% of clouds, 0.3% of points). Rendering
  to an image collapses duplicates; a point cloud keeps them. ``n_duplicates``
  is reported so the asymmetry is visible.
* ``in_dapi`` must be recomputed at the shuffled position. A relocated
  transcript's nucleus membership changes, so copying the flag from the real
  point would be wrong.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
import warnings
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CACHE_DIR, code_version

#: Rendering convention of the published pipeline (create_new_input_row).
RENDER_SHAPE = (100, 100)
RENDER_SIGMA = 1.5

POINT_COLUMNS = ["obs_key", "row", "col", "in_dapi"]


@dataclasses.dataclass(frozen=True)
class ControlSet:
    """One canonical set of shuffled points, shared by every method.

    ``points`` has one row per shuffled transcript:
    ``obs_key`` (``"<cell>__<gene>"`` for real data, the image stem for
    simulated), ``row``/``col`` as int16 pixel indices, ``in_dapi`` bool.
    """
    name: str
    points: pd.DataFrame
    fingerprint: str
    source: str
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    @property
    def n_obs(self) -> int:
        return int(self.points["obs_key"].nunique())

    def for_obs(self, obs_key: str) -> pd.DataFrame:
        return self.points[self.points["obs_key"] == obs_key]

    def __repr__(self) -> str:
        return ("ControlSet(name={!r}, obs={}, points={}, fingerprint={!r})"
                .format(self.name, self.n_obs, self.n_points, self.fingerprint[:12]))


def fingerprint(points: pd.DataFrame) -> str:
    """Stable sha1 over the canonically-sorted point table.

    Two methods scored against different shuffles will carry different
    fingerprints, which is what makes the guarantee checkable rather than
    assumed.
    """
    cols = ["obs_key", "row", "col"]
    df = points[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)
    h = hashlib.sha1()
    h.update(",".join(cols).encode())
    h.update(np.ascontiguousarray(df["obs_key"].astype(str).values.astype("U")).tobytes())
    h.update(np.ascontiguousarray(df["row"].to_numpy(dtype=np.int16)).tobytes())
    h.update(np.ascontiguousarray(df["col"].to_numpy(dtype=np.int16)).tobytes())
    return h.hexdigest()


def make_control(name: str, points: pd.DataFrame, source: str,
                 params: Optional[Dict[str, Any]] = None) -> ControlSet:
    """Normalise a point table and attach its fingerprint."""
    df = points.copy()
    missing = [c for c in POINT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError("point table missing {}".format(missing))
    df["obs_key"] = df["obs_key"].astype(str)
    for c in ("row", "col"):
        df[c] = df[c].to_numpy(dtype=np.int16)
    df["in_dapi"] = df["in_dapi"].astype(bool)
    df = df[POINT_COLUMNS].reset_index(drop=True)

    dup = int(len(df) - len(df.drop_duplicates(["obs_key", "row", "col"])))
    p = dict(params or {})
    p.update({"n_points": int(len(df)), "n_obs": int(df["obs_key"].nunique()),
              "n_duplicate_points": dup,
              "frac_duplicate_points": float(dup / max(1, len(df))),
              "render_sigma": RENDER_SIGMA, "code_version": code_version()})
    return ControlSet(name=name, points=df, fingerprint=fingerprint(df),
                      source=source, params=p)


# --- rendering (what an image-based model consumes) --------------------------

def render_image(rows, cols, sigma: float = RENDER_SIGMA,
                 shape: Tuple[int, int] = RENDER_SHAPE) -> np.ndarray:
    """Render points to the 100x100 image the CVAE was trained on.

    Reproduces the published convention exactly: set each occupied pixel to 1,
    then Gaussian-blur with sigma 1.5. Note that setting rather than
    accumulating is what collapses duplicate points -- kept deliberately, since
    departing from it would stop reproducing the stored control.
    """
    from skimage.filters import gaussian
    img = np.zeros(shape, dtype=float)
    r = np.asarray(rows, dtype=int)
    c = np.asarray(cols, dtype=int)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    img[r[ok], c[ok]] = 1
    return gaussian(img, sigma)


def render_all(cs: ControlSet, obs_keys: Optional[Sequence[str]] = None
               ) -> Dict[str, np.ndarray]:
    """Render every (or selected) observation of a control set."""
    keys = obs_keys if obs_keys is not None else cs.points["obs_key"].unique()
    wanted = set(map(str, keys))
    out = {}
    for key, grp in cs.points.groupby("obs_key", sort=False):
        if key in wanted:
            out[key] = render_image(grp["row"].values, grp["col"].values)
    return out


# --- persistence -------------------------------------------------------------

def to_parquet(cs: ControlSet, path) -> Path:
    """Write the control to parquet plus a JSON sidecar."""
    path = Path(path)
    if path.suffix != ".parquet":
        path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    cs.points.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(
        {"name": cs.name, "fingerprint": cs.fingerprint, "source": cs.source,
         "params": cs.params}, indent=2, sort_keys=True, default=str))
    return path


def _cache_is_stale(cs, request):
    """True when a cached control was built with a different request.

    The cache is keyed on the control's *name*, so without this check a call
    with different ``max_cells`` / ``fields`` / ``with_in_dapi`` would silently
    return the previously cached point set and quietly answer a different
    question than the one asked.
    """
    got = {k: cs.params.get(k) for k in request}
    for k, want in request.items():
        if got.get(k) != want:
            return True, "{}: cached {!r}, requested {!r}".format(k, got.get(k), want)
    return False, ""


def from_parquet(path) -> ControlSet:
    path = Path(path)
    points = pd.read_parquet(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    cs = ControlSet(name=meta["name"], points=points,
                    fingerprint=meta["fingerprint"], source=meta["source"],
                    params=meta.get("params", {}))
    check = fingerprint(points)
    if check != cs.fingerprint:
        raise ValueError("control parquet at {} does not match its recorded "
                         "fingerprint ({} vs {}) -- the file has changed"
                         .format(path, check[:12], cs.fingerprint[:12]))
    return cs


def control_path(name: str, cache_dir=CACHE_DIR) -> Path:
    return Path(cache_dir) / "control_{}.parquet".format(name)


def load_control(name: str, cache_dir=CACHE_DIR) -> ControlSet:
    p = control_path(name, cache_dir)
    if not p.is_file():
        raise FileNotFoundError(
            "no cached control named {!r} at {}. Build one with "
            "extract_from_h5ad() or extract_from_coord_tree().".format(name, p))
    return from_parquet(p)


# --- point-cloud export ------------------------------------------------------

def to_pointcloud_parquet(cs: ControlSet, path) -> Path:
    """Export in the shape a point-cloud model needs.

    Emits ``row``, ``col`` and the nucleus flag under **both** names --
    ``in_dapi`` (what every CSV on disk uses) and ``nucleus`` (what
    ``point_clouds/data.py:117`` actually tests for). The reader falls back to
    all-False when it finds neither, silently dropping the nucleus channel, so
    writing both is what keeps the competitor from being handicapped.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = cs.points.copy()
    df["nucleus"] = df["in_dapi"]
    df["row"] = df["row"].astype(float)
    df["col"] = df["col"].astype(float)
    df.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(
        {"name": cs.name, "fingerprint": cs.fingerprint,
         "columns": list(df.columns), "n_points": int(len(df)),
         "note": "row/col are float pixel indices; nucleus == in_dapi"},
        indent=2, sort_keys=True))
    return path


# --- extraction from a randomised h5ad ---------------------------------------

SHUFFLED_FIELDS = ("subcell_row_shuffled", "subcell_col_shuffled")
REAL_FIELDS = ("subcell_row", "subcell_col")


def _dapi_lookup(f, cell_idx, rows, cols):
    """in_dapi recomputed at the shuffled position, one cell at a time.

    Never reads obsm/dapi_mask whole -- it is 12.3 GB on the APEX file.
    """
    mask = np.asarray(f["obsm/dapi_mask"][cell_idx]).reshape(RENDER_SHAPE) > 0
    r = np.clip(np.asarray(rows, dtype=int), 0, RENDER_SHAPE[0] - 1)
    c = np.clip(np.asarray(cols, dtype=int), 0, RENDER_SHAPE[1] - 1)
    return mask[r, c]


def _form_fields(dt):
    """``{field_name: form_spec}`` for the record array in an awkward h5 group."""
    form = json.loads(dt.attrs["form"])
    inner = form.get("content", form)
    return dict(zip(inner.get("fields") or [], inner.get("contents") or []))


def _list_slice(dt, spec, start, stop):
    """Per-cell offsets and the flat data slice for a ListOffsetArray field."""
    off = np.asarray(dt["{}-offsets".format(spec["form_key"])][start:stop + 1])
    data_key = spec["content"]["form_key"]
    lo, hi = int(off[0]), int(off[-1])
    data = np.asarray(dt["{}-data".format(data_key)][lo:hi])
    return off - off[0], data


def _decode_strings(dt, spec, start, stop):
    """Per-cell string lists, decoded without a Python loop over transcripts.

    A string field is a list-of-lists-of-char. Reconstructing 321M Python
    strings one at a time is the thing that makes this file painful, so the
    characters are gathered into a fixed-width buffer and viewed as bytes in
    one vectorised step.
    """
    outer = np.asarray(dt["{}-offsets".format(spec["form_key"])][start:stop + 1])
    inner_spec = spec["content"]
    s_lo, s_hi = int(outer[0]), int(outer[-1])
    soff = np.asarray(dt["{}-offsets".format(inner_spec["form_key"])][s_lo:s_hi + 1])
    c_lo, c_hi = int(soff[0]), int(soff[-1])
    chars = np.asarray(dt["{}-data".format(inner_spec["content"]["form_key"])][c_lo:c_hi])
    rel = soff - c_lo
    lengths = np.diff(rel)
    if lengths.size == 0:
        return outer - outer[0], np.empty(0, dtype=object)
    width = int(lengths.max())
    buf = np.zeros((lengths.size, width), dtype=np.uint8)
    grid = np.arange(width)[None, :]
    valid = grid < lengths[:, None]
    buf[valid] = chars[(rel[:-1][:, None] + grid)[valid]]
    names = buf.view("S{}".format(width)).ravel()
    return outer - outer[0], names


def _indexed_option_slice(dt, spec, start, stop):
    """Per-cell arrays for an IndexedOptionArray field; None where missing.

    The shuffled coordinates are stored this way on the APEX file: an index per
    cell into an inner list array, negative where that cell has no shuffle.
    On the ileum file the same fields are a plain ListOffsetArray, so both
    shapes have to be supported.
    """
    idx = np.asarray(dt["{}-index".format(spec["form_key"])][start:stop])
    inner = spec["content"]
    present = idx >= 0
    out = [None] * len(idx)
    if not present.any():
        return out
    jlo, jhi = int(idx[present].min()), int(idx[present].max())
    off = np.asarray(dt["{}-offsets".format(inner["form_key"])][jlo:jhi + 2])
    lo, hi = int(off[0]), int(off[-1])
    data = np.asarray(dt["{}-data".format(inner["content"]["form_key"])][lo:hi])
    rel = off - lo
    for i, j in enumerate(idx):
        if j < 0:
            continue
        k = int(j) - jlo
        out[i] = data[rel[k]:rel[k + 1]]
    return out


def _split(offsets, data):
    return [data[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _read_chunk(dt, fields, gene_spec, start, stop):
    """Per-cell (genes, rows, cols) for cells [start, stop)."""
    g_off, g_names = _decode_strings(dt, gene_spec, start, stop)
    genes = _split(g_off, g_names)
    cols = []
    for spec in fields:
        if spec["class"] == "IndexedOptionArray":
            cols.append(_indexed_option_slice(dt, spec, start, stop))
        else:
            off, data = _list_slice(dt, spec, start, stop)
            cols.append(_split(off, data))
    return genes, cols[0], cols[1]


def has_shuffled_points(path) -> bool:
    """Whether a file retains the shuffled coordinates (not just their embedding).

    Only some randomised files kept them: of the four Xenium regions, only
    Region_1 did. The rest carry ``layers["random_subcell_embedding"]`` but
    their ``detected_transcripts`` stops at ``subcell_col``, so those shuffles
    are unrecoverable.
    """
    import h5py
    with h5py.File(path, "r") as f:
        dt = f.get("obsm/detected_transcripts")
        if dt is None:
            return False
        form = dt.attrs.get("form")
        return bool(form and all(fld in str(form) for fld in SHUFFLED_FIELDS))


def extract_from_h5ad(path, name: str, cache_dir=CACHE_DIR,
                      fields: Sequence[str] = SHUFFLED_FIELDS,
                      cell_chunk: int = 2048, force: bool = False,
                      max_cells: Optional[int] = None,
                      with_in_dapi: bool = True) -> ControlSet:
    """Pull a stored point set out of a randomised AnnData.

    ``fields`` selects which coordinates to read: the shuffled control by
    default, or ``REAL_FIELDS`` for the measured points (what a point-based
    model needs as its non-control input).

    Reads the underlying awkward node datasets directly, sliced per cell chunk.
    Neither ``read_h5ad(backed="r")`` nor a whole-array ``read_elem`` is used:
    the first materialises 24.7 GB of masks, and the second would re-read all
    321,668,970 records on every chunk.
    """
    import h5py
    from anndata.experimental import read_elem

    path = Path(path)
    fields = tuple(fields)
    request = {"fields": list(fields), "n_cells_requested": max_cells,
               "in_dapi": ("recomputed_at_shuffled_position" if with_in_dapi else "absent")}
    cached = control_path(name, cache_dir) if cache_dir else None
    if cached is not None and cached.is_file() and not force:
        cs = from_parquet(cached)
        stale, why = _cache_is_stale(cs, request)
        if not stale:
            return cs
        warnings.warn("cached control {!r} was built with a different request "
                      "({}); rebuilding.".format(name, why), stacklevel=2)
    if fields == SHUFFLED_FIELDS and not has_shuffled_points(path):
        raise ValueError(
            "{} has no shuffled coordinates in obsm['detected_transcripts']; "
            "its shuffle is unrecoverable and cannot be shared.".format(path.name))

    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        cell_names = np.asarray(obs.index.astype(str))
        n_cells = len(cell_names) if max_cells is None else min(max_cells, len(cell_names))
        dt = f["obsm/detected_transcripts"]
        specs = _form_fields(dt)
        missing = [x for x in fields if x not in specs]
        if missing:
            raise KeyError("fields {} not in detected_transcripts (have {})"
                           .format(missing, sorted(specs)))
        field_specs = [specs[x] for x in fields]

        keys, rows, cols, flags = [], [], [], []
        n_missing_cells = 0
        for start in range(0, n_cells, cell_chunk):
            stop = min(start + cell_chunk, n_cells)
            genes, rr, cc = _read_chunk(dt, field_specs, specs["gene"], start, stop)
            for local in range(stop - start):
                r, c, g = rr[local], cc[local], genes[local]
                if r is None or c is None or len(r) == 0:
                    n_missing_cells += 1
                    continue
                cell = cell_names[start + local]
                gs = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in g])
                keys.append(np.char.add(np.char.add(cell, "__"), gs))
                rows.append(np.asarray(r)); cols.append(np.asarray(c))
                flags.append(_dapi_lookup(f, start + local, r, c) if with_in_dapi
                             else np.zeros(len(r), dtype=bool))

        points = (pd.DataFrame({"obs_key": np.concatenate(keys),
                                "row": np.concatenate(rows),
                                "col": np.concatenate(cols),
                                "in_dapi": np.concatenate(flags)})
                  if keys else pd.DataFrame(columns=POINT_COLUMNS))

    cs = make_control(name, points, source=str(path),
                      params={"n_cells": n_cells, "cell_chunk": cell_chunk,
                              "n_cells_requested": max_cells, "fields": list(fields),
                              "n_cells_without_points": n_missing_cells,
                              "in_dapi": ("recomputed_at_shuffled_position"
                                          if with_in_dapi else "absent")})
    if cached is not None:
        to_parquet(cs, cached)
    return cs



# --- extraction from a shuffled coordinate tree ------------------------------

def _read_one_obs(csv_path, dapi_path):
    """Points for one observation, with in_dapi at the shuffled position.

    ``np.loadtxt`` rather than ``pd.read_csv``: these are 10-100 row files and
    pandas' per-call overhead dominates (measured 0.88 ms vs 0.08 ms each,
    which is ~10 minutes over the 805,549-file tree).
    """
    from skimage import io

    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    if arr.size == 0:
        return None
    rows = arr[:, 0].astype(int)
    cols = arr[:, 1].astype(int)
    if dapi_path is None:
        in_dapi = np.zeros(len(rows), dtype=bool)
    else:
        mask = np.asarray(io.imread(dapi_path)) > 0
        in_dapi = mask[np.clip(rows, 0, mask.shape[0] - 1),
                       np.clip(cols, 0, mask.shape[1] - 1)]
    return rows, cols, in_dapi


def extract_from_coord_tree(tree, name: str, cache_dir=CACHE_DIR,
                            coords_dir: str = "gene_coords_zoomed",
                            dapi_dir: str = "dapi_zoomed",
                            limit: Optional[int] = None, with_in_dapi: bool = True,
                            n_jobs: int = 1, force: bool = False,
                            stems: Optional[Sequence[str]] = None) -> ControlSet:
    """Read a shuffled simulated tree (one CSV per observation).

    These trees hold the shuffled control for the *classifier-training* stage:
    ``coordinate_parsed_with_coords_shuffled/`` carries ``gene_coords_zoomed/``
    (``row,col``, integer pixel indices), ``dapi_zoomed/`` and
    ``cell_mask_zoomed/``, 1:1 with the randomised AnnData.

    The CSVs have no nucleus column, so ``in_dapi`` is recomputed from the
    co-located DAPI image at the shuffled position.

    ``with_in_dapi`` is the expensive half: one random-access image read per
    observation. Only a *point*-based model needs the flag -- an image model
    renders from the coordinates and takes its DAPI channel separately. Full
    tree: ~2 min without, ~11 min with.
    """
    tree = Path(tree)
    cached = control_path(name, cache_dir) if cache_dir else None
    if cached is not None and cached.is_file() and not force:
        return from_parquet(cached)

    cdir, ddir = tree / coords_dir, tree / dapi_dir
    if not cdir.is_dir():
        raise FileNotFoundError("no {} under {}".format(coords_dir, tree))

    suffix = "_gene-coords.csv"
    if stems is not None:
        # An explicit subset: build paths directly instead of walking a
        # 542,425-entry directory to throw almost all of it away.
        stems = [str(x) for x in stems]
        csvs = [cdir / (st + suffix) for st in stems]
        keep = [i for i, c in enumerate(csvs) if c.is_file()]
        if len(keep) != len(csvs):
            missing = len(csvs) - len(keep)
            warnings.warn("{} of {} requested stems have no coordinate file"
                          .format(missing, len(csvs)), stacklevel=2)
        csvs = [csvs[i] for i in keep]
        stems = [stems[i] for i in keep]
        return _build_from_csvs(csvs, stems, ddir, name, tree, with_in_dapi,
                                n_jobs, cached, limit)
    # Read in directory order, not sorted order. These are 805,549 tiny files on
    # a filesystem where the cost is seek latency, and directory order measured
    # ~17x faster (0.17 vs 2.9 ms per observation). The fingerprint sorts
    # internally, so the ControlSet is identical either way. Sorting is imposed
    # only when `limit` is set, so a subset is a deterministic one.
    csvs = [e for e in cdir.iterdir() if e.name.endswith(".csv")]
    if limit is not None:
        csvs = sorted(csvs)[:limit]
    stems = [e.name[:-len(suffix)] if e.name.endswith(suffix) else e.stem for e in csvs]
    return _build_from_csvs(csvs, stems, ddir, name, tree, with_in_dapi,
                            n_jobs, cached, limit)


def _build_from_csvs(csvs, stems, ddir, name, tree, with_in_dapi, n_jobs,
                     cached, limit):
    # Naming is deterministic (<stem>_DAPI.tif next to <stem>_gene-coords.csv),
    # so paths are constructed rather than indexed -- listing the sibling
    # directory would mean walking 805,549 entries for no gain.
    if with_in_dapi and ddir.is_dir():
        dapis = [(ddir / (st + "_DAPI.tif")) for st in stems]
        dapis = [d if d.is_file() else None for d in dapis]
    else:
        dapis = [None] * len(stems)
    missing_dapi = sum(1 for d in dapis if d is None) if with_in_dapi else len(stems)

    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, batch_size=256)(
            delayed(_read_one_obs)(c, d) for c, d in zip(csvs, dapis))
    else:
        results = [_read_one_obs(c, d) for c, d in zip(csvs, dapis)]

    keys, rows, cols, flags = [], [], [], []
    for stem, res in zip(stems, results):
        if res is None:
            continue
        r, c, fl = res
        keys.append(np.full(len(r), stem, dtype=object))
        rows.append(r); cols.append(c); flags.append(fl)

    points = (pd.DataFrame({"obs_key": np.concatenate(keys),
                            "row": np.concatenate(rows),
                            "col": np.concatenate(cols),
                            "in_dapi": np.concatenate(flags)})
              if keys else pd.DataFrame(columns=POINT_COLUMNS))

    cs = make_control(name, points, source=str(tree),
                      params={"n_files_read": len(csvs),
                              "n_missing_dapi": missing_dapi,
                              "in_dapi": ("recomputed_at_shuffled_position"
                                          if with_in_dapi else "absent"),
                              "limit": limit, "n_jobs": n_jobs})
    if cached is not None and limit is None:
        to_parquet(cs, cached)
    return cs


# --- consistency guard -------------------------------------------------------

def check_control_consistency(df: pd.DataFrame, raise_on_mismatch: bool = True
                              ) -> Dict[str, Any]:
    """Verify every compared row used the same shuffled points.

    Reads ``control_fingerprint`` out of ``params_json``. Rows that carry no
    fingerprint are reported separately: they were scored before the shared
    control existed, or against the stored embedding, and are not comparable to
    fingerprinted rows without checking.
    """
    if "params_json" not in df.columns:
        raise KeyError("expected a results frame with params_json")
    fp = df["params_json"].str.extract(r'"control_fingerprint":\s*"([0-9a-f]+)"')[0]
    work = df.assign(_fp=fp)
    # A control belongs to a dataset: the APEX shuffle and the ileum shuffle are
    # different point sets and *should* differ. The property that matters is that
    # no single dataset was scored against two different shuffles.
    group = "dataset" if "dataset" in work.columns else None
    per = {}
    bad = {}
    for key, sub in (work.groupby(group) if group else [("all", work)]):
        known = sub["_fp"].dropna().unique().tolist()
        per[str(key)] = known
        if len(known) > 1:
            bad[str(key)] = sub.dropna(subset=["_fp"]).groupby(["_fp", "method"]).size().to_dict()
    report = {"n_rows": int(len(df)),
              "n_without_fingerprint": int(fp.isna().sum()),
              "fingerprints_by_dataset": per,
              "consistent": not bad}
    if bad and raise_on_mismatch:
        raise ValueError(
            "these datasets were scored against more than one shuffled control: "
            "{}. Scores from different shuffles are not comparable -- rebuild the "
            "affected runs against one ControlSet.".format(bad))
    return report


# --- simulated metadata ------------------------------------------------------

_SIM_PAT = {
    "pattern":  re.compile(r"^([a-z\-]+)_"),
    "cell_id":  re.compile(r"(?:icell|cellID)(\d+)"),
    "n_spots":  re.compile(r"(\d+)spots"),
    "rotation": re.compile(r"rotated(\d+)"),
    "prop":     re.compile(r"prop(\d\.\d+)"),
}


def simulated_metadata(keys: Sequence[str]) -> pd.DataFrame:
    """Recover per-observation metadata from simulated filename stems.

    A point-based model's dump carries only the observation key, but the
    benchmarks need ``pattern`` (the label) and ``cell_id`` (the grouping
    variable). Both are encoded in the filename, so they are parsed back rather
    than carried through the adapter.
    """
    out = {"obs_key": [str(k) for k in keys]}
    for col, rx in _SIM_PAT.items():
        vals = []
        for k in out["obs_key"]:
            m = rx.search(str(k))
            vals.append(m.group(1) if m else None)
        if col in ("n_spots", "rotation"):
            out[col] = pd.to_numeric(vals, errors="coerce")
        elif col == "prop":
            out[col] = pd.to_numeric(vals, errors="coerce")
        else:
            out[col] = vals
    df = pd.DataFrame(out)
    df["cell_id"] = df["cell_id"].astype(str)
    return df
