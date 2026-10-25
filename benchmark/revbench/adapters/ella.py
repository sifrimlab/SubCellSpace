"""ELLA adapter -- runs in ELLA's own environment, not in ``subcellspace_env``.

    /mnt/data/miniconda3/envs/ella/bin/python src/revbench/adapters/ella.py \
        --input  /mnt/data/david/benchmarks/ELLA/inputs/apex_real \
        --work   /mnt/data/david/benchmarks/ELLA/work/apex_real \
        --out    /mnt/data/david/benchmarks/ELLA/dumps/ella__apex_real \
        --stages register,profiles,fit --n-jobs 32

Like ``adapters/pointcloud_vae.py`` this imports **nothing** from ``revbench``:
it sees numpy 2.4 / pandas 3.0 / sklearn 1.9, while the harness sees numpy
1.23 / pandas 1.x / sklearn 1.2. The two halves agree only on the file formats,
and the ``schema`` field in each sidecar is what keeps the duplicated
``write_dump`` below honest against ``representations.py``'s original.

It emits two things, because ELLA is benchmarked in two lanes:

* ``<out>_genes.csv``    -- ELLA's native per-gene test (p-value, score, pattern)
* ``<out>_profiles.npz`` -- a per-(cell, gene) radial profile, built from ELLA's
  own ``df_registered.d_c_s``. This is *our* construction from ELLA's coordinate
  system, not ELLA's inference, and the sidecar says so in ``lane``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DUMP_SCHEMA = 1
INDEX_KINDS = ("obs_names", "canonical_key", "template", "cell_gene", "image_stem")


def write_dump(path, X, obs_index, index_kind, method, dataset, extra=None):
    """Copy of ``representations.write_dump``. Kept in sync by ``schema``."""
    import numpy as np
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
    meta = {"schema": DUMP_SCHEMA, "method": method, "dataset": dataset,
            "index_kind": index_kind, "n": int(X.shape[0]),
            "latent_dim": int(X.shape[1]),
            "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    meta.update(extra or {})
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str))
    return path


# --- input -------------------------------------------------------------------

def load_input(input_dir):
    """Rebuild ELLA's input dict from the harness's export."""
    import numpy as np
    import pandas as pd

    input_dir = Path(input_dir)
    meta = json.loads((input_dir / "meta.json").read_text())
    expr = pd.read_parquet(input_dir / "expr.parquet")

    with np.load(input_dir / "cell_poly.npz", allow_pickle=False) as npz:
        cells = [str(c) for c in npz["cells"]]
        offsets = np.asarray(npz["offsets"])
        verts = np.asarray(npz["verts"], dtype=np.float64)
    cell_poly = {c: verts[offsets[i]:offsets[i + 1]] for i, c in enumerate(cells)}

    # ELLA indexes results by cell type first; this project has one type per
    # dataset, but the structure is kept general so an annotated dataset needs
    # no code change here.
    cell_type = meta.get("cell_type", "cell")
    types = sorted(expr["type"].astype(str).unique())
    data = {
        "types": types,
        "cells": {t: sorted(expr.loc[expr["type"] == t, "cell"].astype(str).unique())
                  for t in types},
        "genes": {t: sorted(expr.loc[expr["type"] == t, "gene"].astype(str).unique())
                  for t in types},
        "cells_all": sorted(expr["cell"].astype(str).unique()),
        "cell_poly": cell_poly,
        "expr": expr,
    }
    missing = [c for c in data["cells_all"] if c not in cell_poly]
    if missing:
        raise KeyError("{} cells have no polygon (first: {})"
                       .format(len(missing), missing[:3]))
    return data, meta, cell_type


# --- lane 2: the radial profile ----------------------------------------------

def radial_profiles(df_registered, bins=10):
    """Per-observation histogram of ELLA's registered radius ``d_c_s``.

    Normalised to sum 1, so the vector carries *shape* and not magnitude --
    deliberately, because ``n_spots`` is one of the negative controls the
    representation is scored against, and a count feature would hand it over.
    """
    import numpy as np
    import pandas as pd

    df = df_registered
    key = (df["obs_key"].astype(str) if "obs_key" in df.columns
           else df["cell"].astype(str) + "__" + df["gene"].astype(str))
    r = np.clip(df["d_c_s"].to_numpy(dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(r, edges[1:-1], right=False), 0, bins - 1)

    keys, inverse = np.unique(key.to_numpy(), return_inverse=True)
    counts = np.zeros((len(keys), bins), dtype=np.float64)
    np.add.at(counts, (inverse, idx), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    profiles = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    n_out = int((df["d_c_s"].to_numpy(dtype=np.float64) > 1.0).sum())
    return keys, profiles, {"n_molecules_outside_polygon": n_out,
                            "frac_molecules_outside_polygon":
                                float(n_out / max(1, len(df)))}


# --- lane 1: the per-gene table ----------------------------------------------

def gene_table(ella, labels_by_type=None):
    """ELLA's native results, one row per (cell type, gene)."""
    import numpy as np
    import pandas as pd

    rows = []
    for t in ella.type_list:
        genes = list(ella.gene_list_dict[t])
        pv_fdr = np.asarray(ella.pv_fdr_tl.get(t, []), dtype=float)
        pv_cau = np.asarray(ella.pv_cauchy_tl.get(t, []), dtype=float)
        scores = np.asarray(ella.scores.get(t, []), dtype=float)
        labels = (labels_by_type or {}).get(t)
        for i, g in enumerate(genes):
            rows.append({
                "cell_type": t, "gene_name": str(g),
                "pv_fdr": float(pv_fdr[i]) if i < len(pv_fdr) else float("nan"),
                "pv_cauchy": float(pv_cau[i]) if i < len(pv_cau) else float("nan"),
                "ella_score": float(scores[i]) if i < len(scores) else float("nan"),
                "ella_pattern": ("pattern_{}".format(int(labels[i]))
                                 if labels is not None and i < len(labels)
                                 and labels[i] >= 0 else "random"),
            })
    return pd.DataFrame(rows)


# --- sharding ----------------------------------------------------------------
# ELLA has no built-in parallelism and at this project's scale its two serial
# loops dominate: registration runs at ~24 cells/s and preparation is per
# (gene, cell). Both are exactly shardable, and that is a property of the code
# rather than an assumption -- `register_cells` touches one cell's rows and its
# polygon, `nhpp_prepare` touches one gene's rows, and the Newton fit is a
# deterministic per-gene solve (ELLA.py:369, :468, :616).
#
# What is NOT shardable is everything after the fit. `compute_pv` imputes NaN
# log-likelihoods with `np.nanmin` over the *whole panel* (ELLA.py:845), so a
# batch would use a different imputation value than a full run. Estimation is
# therefore assembled and run once in the parent, over all genes, exactly as a
# serial run does. Only prepare+fit is distributed.

def _subset_data(data, cells=None, genes=None):
    """A data dict restricted to some cells and/or genes."""
    expr = data["expr"]
    if cells is not None:
        keep = set(map(str, cells))
        expr = expr[expr["cell"].astype(str).isin(keep)]
    if genes is not None:
        keep_g = set(map(str, genes))
        expr = expr[expr["gene"].astype(str).isin(keep_g)]
    types = sorted(expr["type"].astype(str).unique())
    cells_all = sorted(expr["cell"].astype(str).unique())
    return {
        "types": types,
        "cells": {t: sorted(expr.loc[expr["type"] == t, "cell"].astype(str).unique())
                  for t in types},
        "genes": {t: sorted(expr.loc[expr["type"] == t, "gene"].astype(str).unique())
                  for t in types},
        "cells_all": cells_all,
        "cell_poly": {c: data["cell_poly"][c] for c in cells_all},
        "expr": expr,
    }


def _shard_cwd(work, tag):
    """Each worker gets its own directory: ELLA writes checkpoints to a literal
    './output', so shards sharing a cwd would race on the same file."""
    d = Path(work) / "shards" / str(tag)
    d.mkdir(parents=True, exist_ok=True)
    os.chdir(d)
    return d


def register_shard(payload):
    """Register one batch of cells; returns that batch's ``df_registered``."""
    data, dataset, work, tag = payload
    from ELLA import ELLA
    _shard_cwd(work, "reg_%s" % tag)
    e = ELLA(dataset=dataset)
    e.load_data(data_dict=data)
    e.register_cells()
    return e.df_registered


def fit_shard(payload):
    """Prepare + fit one batch of genes.

    Returns the raw fit only -- ``A_est``, ``B_est``, ``mll_est`` -- and never a
    p-value, because the p-value stage is panel-wide.
    """
    data, df_reg, dataset, work, tag = payload
    import numpy as np
    from ELLA import ELLA
    _shard_cwd(work, "fit_%s" % tag)
    e = ELLA(dataset=dataset)
    e.load_data(data_dict=data)
    e.load_registered_cells(registered_dict={"df_registered": df_reg})
    e.nhpp_prepare()
    e.nhpp_fit()
    t = e.type_list[0]
    return (list(map(str, e.gene_list_dict[t])),
            np.asarray(e.A_est[t]), np.asarray(e.B_est[t]), np.asarray(e.mll_est[t]))


def run_register(data, dataset, work, n_jobs, log=print):
    """``df_registered`` for every cell, sharded over processes if asked."""
    import pandas as pd
    from ELLA import ELLA

    if n_jobs <= 1:
        e = ELLA(dataset=dataset)
        e.load_data(data_dict=data)
        e.register_cells()
        return e.df_registered

    from concurrent.futures import ProcessPoolExecutor
    cells = list(data["cells_all"])
    n = max(1, min(n_jobs, len(cells)))
    batches = [b for b in (cells[i::n] for i in range(n)) if b]
    payload = [(_subset_data(data, cells=b), dataset, work, i)
               for i, b in enumerate(batches)]
    log("registering %d cells over %d shards" % (len(cells), len(payload)))
    with ProcessPoolExecutor(max_workers=n) as ex:
        parts = list(ex.map(register_shard, payload))
    out = pd.concat(parts)
    # Restore the serial row order. `register_cells` walks `cell_list_all`, which
    # `load_data` sorts, so a serial run's frame is in sorted-cell order; a concat
    # of interleaved shards is not. `nhpp_prepare` then sums each gene's radii in
    # frame order, and float addition is not associative -- a different order
    # perturbs the log-likelihood in the last bits, which is enough to flip the
    # knife-edge `T <= 0` boundary test that decides whether a p-value is computed
    # or drawn at random. Sorting here makes the sharded fit bit-identical to the
    # serial one.
    out = out.assign(_k=out["cell"].astype(str)).sort_values(
        "_k", kind="stable").drop(columns="_k")
    return out


def run_fit(data, df_registered, dataset, work, n_jobs, log=print):
    """Prepare + fit every gene, then assemble one panel-wide ELLA object.

    The returned object has ``A_est`` / ``B_est`` / ``mll_est`` for the full
    panel in ``gene_list_dict`` order and has *not* had ``weighted_density_est``
    or ``compute_pv`` called on it -- those are the caller's, because they must
    see the whole panel and because the p-value stage is re-run per seed.
    """
    import numpy as np
    import pandas as pd
    from ELLA import ELLA

    t = data["types"][0]
    genes = [str(g) for g in data["genes"][t]]

    if n_jobs <= 1:
        e = ELLA(dataset=dataset)
        e.load_data(data_dict=data)
        e.load_registered_cells(registered_dict={"df_registered": df_registered})
        e.nhpp_prepare()
        e.nhpp_fit()
        return e

    from concurrent.futures import ProcessPoolExecutor
    n = max(1, min(n_jobs, len(genes)))
    batches = [b for b in (genes[i::n] for i in range(n)) if b]
    by_gene = df_registered.groupby("gene", observed=True)
    payload = []
    for i, b in enumerate(batches):
        present = [g for g in b if g in by_gene.groups]
        if not present:
            continue
        payload.append((_subset_data(data, genes=b),
                        pd.concat([by_gene.get_group(g) for g in present]),
                        dataset, work, i))
    log("prepare+fit for %d genes over %d shards" % (len(genes), len(payload)))
    with ProcessPoolExecutor(max_workers=n) as ex:
        shards = list(ex.map(fit_shard, payload))

    rows = {}
    for gs, A, B, M in shards:
        for j, g in enumerate(gs):
            rows[g] = (A[j], B[j], M[j])
    order = [g for g in genes if g in rows]
    missing = [g for g in genes if g not in rows]
    if missing:
        log("WARNING: %d genes produced no fit (first: %s)" % (len(missing), missing[:3]))

    e = ELLA(dataset=dataset)
    e.load_data(data_dict=_subset_data(data, genes=order))
    e.gene_list_dict = {t: order}
    e.type_list = [t]
    e.ng_dict = {t: len(order)}
    e.A_est = {t: np.stack([rows[g][0] for g in order])}
    e.B_est = {t: np.stack([rows[g][1] for g in order])}
    e.mll_est = {t: np.stack([rows[g][2] for g in order])}
    return e


def estimate(ella, seed, pattern_k, log=print):
    """``weighted_density_est`` + ``compute_pv`` at one seed, plus clustering.

    ELLA's p-value is **stochastic by construction**: for a (gene, kernel) whose
    likelihood-ratio statistic sits at the parameter boundary (T <= 0) it draws
    ``1 - Uniform(0, 0.5)`` (ELLA.py:855). That is the standard randomised
    p-value that keeps a boundary-constrained LRT calibrated -- but ELLA never
    seeds the global RNG, so two runs on identical input disagree. Measured on a
    300-cell APEX slice: 32.3% of (gene, kernel) cells are at the boundary,
    263 of 264 genes get a different p-value between two runs, and one gene's
    significance call flips.

    So the seed is set here, and the caller runs this at several seeds. That is
    the only way a ROC computed from these p-values means anything.
    """
    import numpy as np
    np.random.seed(int(seed))
    ella.weighted_density_est()
    ella.compute_pv()
    labels, err = None, None
    try:
        ella.pattern_clustering()
        ella.pattern_labeling(K=pattern_k)
        labels = {t: np.asarray(v) for t, v in ella.labels_dict.items()}
    except Exception as exc:  # noqa: BLE001
        # Too few significant genes to cluster is a legitimate outcome, and the
        # shuffled-control arm is expected to hit it. Record rather than crash.
        err = repr(exc)
        log("pattern labelling skipped: %r" % (exc,))
    return labels, err


# --- full-scale staged modes -------------------------------------------------
# The default mode holds the whole dataset in one process, which is right for
# 2,031 cells and wrong for all 150,151: the APEX file carries 321,668,970
# transcript records, and one `df_registered` over all of them is tens of GB.
# These three modes split the same pipeline so each step's memory is bounded by
# a block of cells or a shard of genes, and each writes a checkpoint that
# `scripts/run_apex_full.py` can resume from.
#
#   register-block : one cell block -> radial profiles + df_registered,
#                    partitioned by gene shard so the fit stage reads contiguously
#   fit-shard      : one gene shard, all blocks -> A_est / B_est / mll_est
#   estimate       : all shards -> panel-wide weighted_density_est + compute_pv
#
# The split is exact for the same reason the in-process sharding is: registration
# is per cell, preparation is per gene, the Newton fit is a deterministic
# per-gene solve, and everything panel-wide happens in `estimate`.

def gene_shard_of(gene, panel_index, n_shards):
    """Stable gene -> shard. Position in the panel, not `hash()`, which is
    salted per process and would scatter a gene differently in every block."""
    return panel_index[str(gene)] % n_shards


def mode_register_block(a, log=print):
    """Register one cell block; emit profiles and gene-partitioned registrations."""
    import numpy as np
    import pandas as pd

    data, meta, _ = load_input(a.input)
    panel = json.loads(Path(a.panel_file).read_text())
    panel_index = {str(g): i for i, g in enumerate(panel)}
    tag = a.block_tag or Path(a.input).name

    t0 = time.time()
    df = run_register(data, meta.get("arm", "run"), a.work, a.n_jobs, log=log)
    t_reg = time.time() - t0

    keys, X, info = radial_profiles(df, bins=a.bins)
    write_dump(Path(a.out).with_name(Path(a.out).name + "_profiles.npz"), X, keys,
               a.index_kind or meta.get("index_kind", "cell_gene"), a.method,
               meta.get("dataset", ""),
               extra={"lane": "derived_radial_profile", "bins": a.bins,
                      "arm": meta.get("arm", ""), "block": tag,
                      "frame": meta.get("frame", ""),
                      "radius_source": "ELLA df_registered.d_c_s",
                      "normalisation": "sum_to_one_no_count_feature", **info})

    # The fit needs radius and counts, not coordinates -- and not `obs_key`,
    # which is just "<cell>__<gene>" and at ~316M rows is the largest column on
    # disk. The profiles above already consumed it.
    slim = df[["type", "cell", "gene", "umi", "sc_total", "d_c_s"]].copy()
    slim["_shard"] = [gene_shard_of(g, panel_index, a.gene_shards)
                      if str(g) in panel_index else -1 for g in slim["gene"]]
    slim = slim[slim["_shard"] >= 0]

    out_root = Path(a.emit_registered)
    written = {}
    for shard, grp in slim.groupby("_shard", sort=True):
        d = out_root / ("shard_%03d" % int(shard))
        d.mkdir(parents=True, exist_ok=True)
        f = d / ("%s.parquet" % tag)
        tmp = f.with_suffix(".parquet.tmp")
        grp.drop(columns="_shard").to_parquet(tmp, index=False)
        tmp.replace(f)              # atomic: a half-written shard must not look done
        written[int(shard)] = int(len(grp))

    return {"block": tag, "n_cells": meta.get("n_cells"),
            "n_molecules": int(len(df)), "n_observations": int(len(keys)),
            "register_s": round(t_reg, 1), "shards_written": len(written),
            "rows_per_shard": written, **info}


def _bare_ella(dataset, type_name, genes, cells, df_registered):
    """An ELLA instance carrying only what prepare+fit reads.

    `load_data` derives `type_list`, `gene_list_dict`, `cell_list_dict`,
    `cells_all`, `nt`, `nc_all`, `ng_dict`, `nc_dict`, `ntanbin_dict` and `nk`
    from `expr`, and stores `cell_poly` and `data_df` for `register_cells`
    (ELLA.py:306). Registration is already done here, so those last two are not
    just unnecessary -- carrying the polygons and coordinates into every fit
    worker is the difference between a few GB and tens of them. The derived
    fields are set explicitly instead.
    """
    from ELLA import ELLA
    e = ELLA(dataset=dataset)
    e.type_list = [type_name]
    e.gene_list_dict = {type_name: list(genes)}
    e.cell_list_dict = {type_name: list(cells)}
    e.cell_list_all = list(cells)
    e.nt = 1
    e.nc_all = len(cells)
    e.ng_dict = {type_name: len(genes)}
    e.nc_dict = {type_name: len(cells)}
    e.ntanbin_dict = {type_name: e.max_ntanbin}
    e.nk = len(e.beta_kernel_param_list)
    e.df_registered = df_registered
    return e


def mode_fit_shard(a, log=print):
    """Prepare + fit every gene in one shard, across all blocks."""
    import numpy as np
    import pandas as pd

    shard_dir = Path(a.shard_dir)
    files = sorted(shard_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no block parquet in %s" % shard_dir)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Sorted by cell so a shard's row order does not depend on how blocks were
    # cut: float summation order changes the log-likelihood in the last bits,
    # which is enough to flip the knife-edge boundary test in compute_pv.
    df = df.assign(_k=df["cell"].astype(str)).sort_values("_k", kind="stable").drop(columns="_k")

    t = str(df["type"].iloc[0])
    genes = sorted(df["gene"].astype(str).unique())
    cells = sorted(df["cell"].astype(str).unique())
    log("shard %s: %d blocks, %d genes, %d cells, %d molecules"
        % (shard_dir.name, len(files), len(genes), len(cells), len(df)))

    t0 = time.time()
    e = _bare_ella(a.method, t, genes, cells, df)
    e.nhpp_prepare()
    e.nhpp_fit()
    took = time.time() - t0

    final = Path(a.out)
    if final.suffix != ".npz":
        final = final.with_suffix(".npz")
    final.parent.mkdir(parents=True, exist_ok=True)
    # The in-flight name must itself end in .npz: savez_compressed silently
    # appends the extension otherwise, the rename below then finds nothing, and
    # the whole shard is redone on every resume.
    tmp = final.with_name(final.stem + ".inflight.npz")
    np.savez_compressed(tmp, genes=np.asarray(genes, dtype=np.str_),
                        A=np.asarray(e.A_est[t]), B=np.asarray(e.B_est[t]),
                        mll=np.asarray(e.mll_est[t]),
                        cell_type=np.asarray([t], dtype=np.str_))
    tmp.replace(final)
    return {"shard": shard_dir.name, "n_genes": len(genes), "n_cells": len(cells),
            "n_molecules": int(len(df)), "fit_s": round(took, 1)}


def mode_estimate(a, log=print):
    """Panel-wide estimation over every shard's fit, at several seeds."""
    import numpy as np
    import pandas as pd

    panel = [str(g) for g in json.loads(Path(a.panel_file).read_text())]
    rows, ctype = {}, None
    for f in sorted(Path(a.fit_dir).glob("shard_*.npz")):
        if f.name.endswith(".inflight.npz"):
            continue
        with np.load(f, allow_pickle=False) as z:
            gs = [str(g) for g in z["genes"]]
            A, B, M = z["A"], z["B"], z["mll"]
            ctype = str(z["cell_type"][0])
        for j, g in enumerate(gs):
            rows[g] = (A[j], B[j], M[j])
    order = [g for g in panel if g in rows]
    missing = [g for g in panel if g not in rows]
    if missing:
        log("WARNING: %d panel genes have no fit (first: %s)" % (len(missing), missing[:5]))
    log("estimating over %d genes from %d shard fits" % (len(order), len(rows) and len(list(Path(a.fit_dir).glob('shard_*.npz')))))

    e = _bare_ella(a.method, ctype, order, ["_"], None)
    e.A_est = {ctype: np.stack([rows[g][0] for g in order])}
    e.B_est = {ctype: np.stack([rows[g][1] for g in order])}
    e.mll_est = {ctype: np.stack([rows[g][2] for g in order])}

    mll = e.mll_est[ctype]
    T = -2.0 * (mll[:, -1][:, None] - mll)
    boundary = float(((T <= 0) | np.isnan(T)).mean())

    tables, err = [], None
    for r in range(max(1, a.repeats)):
        labels, e_err = estimate(e, a.seed + r, a.pattern_k, log=log)
        err = err or e_err
        tab = gene_table(e, labels)
        tab["repeat"], tab["seed"] = r, a.seed + r
        tables.append(tab)
    table = pd.concat(tables, ignore_index=True)
    table["n_cells"] = int(a.n_cells or 0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_name(out.name + "_genes.csv")
    table.to_csv(str(csv) + ".tmp", index=False)
    Path(str(csv) + ".tmp").replace(csv)

    sig = table.groupby("repeat")["pv_fdr"].apply(lambda s: int((s <= 0.05).sum()))
    return {"n_genes": len(order), "n_genes_missing": len(missing),
            "frac_boundary_gene_kernel_cells": round(boundary, 4),
            "n_significant_per_repeat": [int(v) for v in sig],
            "pattern_labeling_error": err, "gene_table": str(csv)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--stages", default="register,profiles,fit")
    ap.add_argument("--method", default="ella")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--pattern-k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--repeats", type=int, default=5,
                    help="p-value stages to run, at seed, seed+1, ... "
                         "ELLA's p-value is stochastic; one draw is not a result")
    ap.add_argument("--index-kind", default=None)
    # --- full-scale staged modes (see scripts/run_apex_full.py) --------------
    ap.add_argument("--mode", default="run",
                    choices=["run", "register-block", "fit-shard", "estimate"])
    ap.add_argument("--panel-file", default=None,
                    help="JSON list of genes; fixes the gene->shard map")
    ap.add_argument("--gene-shards", type=int, default=64)
    ap.add_argument("--emit-registered", default=None,
                    help="register-block: root for gene-partitioned registrations")
    ap.add_argument("--block-tag", default=None)
    ap.add_argument("--shard-dir", default=None, help="fit-shard: one shard_NNN dir")
    ap.add_argument("--fit-dir", default=None, help="estimate: dir of shard .npz fits")
    ap.add_argument("--n-cells", type=int, default=0, help="recorded in the gene table")
    ap.add_argument("--report", default=None, help="write the stage summary here")
    a = ap.parse_args(argv)

    # pattern_clustering draws an elbow plot and there is no display here.
    # Forced, not setdefault: a Jupyter kernel exports
    # MPLBACKEND=module://matplotlib_inline.backend_inline, a child process
    # inherits it, and this env has no matplotlib_inline -- so ELLA's
    # `import matplotlib.pyplot` raises before any of its code runs.
    os.environ["MPLBACKEND"] = "Agg"
    if a.n_jobs > 1:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")

    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd

    NEEDS = {"run": ("input",), "register-block": ("input", "panel_file", "emit_registered"),
             "fit-shard": ("shard_dir",), "estimate": ("fit_dir", "panel_file")}
    for opt in NEEDS[a.mode]:
        if not getattr(a, opt, None):
            ap.error("--%s is required for --mode %s" % (opt.replace("_", "-"), a.mode))

    if a.mode != "run":
        handler = {"register-block": mode_register_block,
                   "fit-shard": mode_fit_shard,
                   "estimate": mode_estimate}[a.mode]
        cwd = os.getcwd()
        os.chdir(work)
        try:
            summary = handler(a)
        finally:
            os.chdir(cwd)
        summary = {"mode": a.mode, **summary,
                   "versions": {"python": sys.version.split()[0],
                                "numpy": np.__version__, "pandas": pd.__version__},
                   "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
        if a.report:
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
        print(json.dumps(summary, indent=2, default=str))
        return 0

    data, meta, cell_type = load_input(a.input)
    index_kind = a.index_kind or meta.get("index_kind") or "cell_gene"
    dataset = meta.get("dataset", "")
    arm = meta.get("arm", "run")
    timings, notes, profile_info = {}, {}, {}

    cwd = os.getcwd()
    os.chdir(work)
    try:
        t0 = time.time()
        df_registered = run_register(data, arm, work, a.n_jobs)
        timings["register_cells_s"] = round(time.time() - t0, 2)

        if "profiles" in stages:
            t0 = time.time()
            keys, X, profile_info = radial_profiles(df_registered, bins=a.bins)
            write_dump(out.with_name(out.name + "_profiles.npz"), X, keys,
                       index_kind, a.method, dataset,
                       extra={"lane": "derived_radial_profile",
                              "bins": a.bins, "arm": arm,
                              "frame": meta.get("frame", ""),
                              "control_fingerprint": meta.get("control_fingerprint", ""),
                              "radius_source": "ELLA df_registered.d_c_s",
                              "normalisation": "sum_to_one_no_count_feature",
                              "n_cells": meta.get("n_cells"), **profile_info})
            timings["profiles_s"] = round(time.time() - t0, 2)

        if "fit" in stages:
            t0 = time.time()
            ella = run_fit(data, df_registered, arm, work, a.n_jobs)
            timings["prepare_and_fit_s"] = round(time.time() - t0, 2)

            t0 = time.time()
            tables, boundary = [], None
            t = ella.type_list[0]
            mll = np.asarray(ella.mll_est[t])
            T = -2.0 * (mll[:, -1][:, None] - mll)
            boundary = float(((T <= 0) | np.isnan(T)).mean())
            for r in range(max(1, a.repeats)):
                seed = a.seed + r
                labels, err = estimate(ella, seed, a.pattern_k)
                if err:
                    notes.setdefault("pattern_labeling_error", err)
                tab = gene_table(ella, labels)
                tab["repeat"], tab["seed"] = r, seed
                tables.append(tab)
            table = pd.concat(tables, ignore_index=True)
            table["n_cells"] = int(meta.get("n_cells", 0))
            table.to_csv(str(out.with_name(out.name + "_genes.csv")), index=False)
            timings["estimate_s"] = round(time.time() - t0, 2)

            sig = table.groupby("repeat")["pv_fdr"].apply(lambda s: int((s <= 0.05).sum()))
            notes.update({
                "n_genes": int(table["gene_name"].nunique()),
                "n_significant_per_repeat": [int(v) for v in sig],
                "frac_boundary_gene_kernel_cells": round(boundary, 4),
            })

            lam = getattr(ella, "weighted_lam_est", None)
            if lam:
                np.savez_compressed(
                    str(out.with_name(out.name + "_intensity.npz")),
                    **{str(k): np.asarray(v, dtype=np.float32) for k, v in lam.items()})
    finally:
        os.chdir(cwd)

    run = {"schema": 1, "method": a.method, "dataset": dataset, "arm": arm,
           "stages": sorted(stages), "index_kind": index_kind,
           "n_jobs": a.n_jobs, "bins": a.bins, "pattern_k": a.pattern_k,
           "seed": a.seed, "repeats": a.repeats,
           "input_meta": meta, "timings_s": timings, "notes": notes,
           "versions": {"python": sys.version.split()[0],
                        "numpy": np.__version__, "pandas": pd.__version__},
           "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    run.update(profile_info)
    Path(str(out) + "_run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True, default=str))
    print(json.dumps({"out": str(out), "timings_s": timings, "notes": notes,
                      **profile_info}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
