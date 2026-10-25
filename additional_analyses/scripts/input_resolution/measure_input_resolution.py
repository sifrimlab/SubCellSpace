#!/usr/bin/env python
# Provenance for additional_analyses/input_resolution.ipynb.
# Writes cells_{apex,ileum}.csv, displacement_{apex,ileum}.csv, displacement_hist_{apex,ileum}.npz,
# occupancy_{apex,ileum}.csv and boundary_{apex,ileum}.csv.gz (the files under input_resolution/
# in the data bundle). Expects the two published objects from the main SubCellSpace data record
# (10.5281/zenodo.19708995): hek293T_Xenium_adata_subcell_including_randomized_embeddings.h5ad and
# merfish_SI_adata_subcell_including_randomized_embeddings.h5ad; edit the paths in DATASETS / OUT below.
"""How much spatial information does the 100x100 input grid actually discard?

Produces the cached tables read by input_resolution.ipynb. The h5ads store the whole
coordinate chain per transcript -- raw microns, native camera pixels, and the 100x100
integers the model finally sees -- so the cost of the 100x100 grid can be measured.

The chain, verified to reproduce the stored integers exactly (the `gate` stage):

    subcell_row = rint((round(local_row) - cell_bbox[0]) * 100/bbox_height)

so there are two quantisations, not one. `round(local_row)` snaps a transcript to the
native camera pixel; `rint(... * 100/h)` snaps it to our grid. Because a segmented cell
is smaller than 100 px across, the second step is on a FINER lattice than the first --
the grid adds less positional error than the camera already has.

Reads the awkward array's flat leaf datasets straight out of HDF5 rather than through
anndata: the APEX file is 12.7 GB and holds 321,668,970 transcripts, which do not need
materialising to be summarised.

Writes to /path/to/work/resolution/work/.
"""
import argparse
import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

OUT = Path("/path/to/work/resolution/work")
DATA = {
    "apex": dict(
        path=Path("/mnt/data/david/subcellspace_data/apex_validation_data/"
                  "adata_subcell_including_randomized_embeddings.h5ad"),
        label="APEX / HEK (Xenium)", um_per_px=0.2125),
    "ileum": dict(
        path=Path("/mnt/data/david/subcellspace_data/small_intestine_data/"
                  "adata_subcell_including_randomized_embeddings.h5ad"),
        label="ileum (MERFISH)", um_per_px=0.108),
}
GRID = 100
SEED = 101


def fields(dt):
    """field name -> (offsets dataset, leaf data dataset), read off the awkward form."""
    form = json.loads(dt.attrs["form"])
    def leaf(c):
        while "content" in c:
            c = c["content"]
        return c["form_key"]
    return {f: (c["form_key"] + "-offsets", leaf(c) + "-data")
            for f, c in zip(form["fields"], form["contents"])}


def string_field(dt, name):
    """(outer offsets, inner offsets, char data) for a string field.

    The node numbering differs between the two files -- the ileum's record puts `gene`
    at a different field index -- so these must be read off the form, never hardcoded.
    """
    form = json.loads(dt.attrs["form"])
    c = dict(zip(form["fields"], form["contents"]))[name]
    outer = c["form_key"]
    inner = c["content"]["form_key"]
    chars = c["content"]["content"]["form_key"]
    return (dt[outer + "-offsets"], dt[inner + "-offsets"], dt[chars + "-data"])


def obs_col(f, key):
    g = f["obs"][key]
    if isinstance(g, h5py.Group):                     # categorical
        cats = np.array([x.decode() if isinstance(x, bytes) else x
                         for x in g["categories"][:]])
        return cats[g["codes"][:]]
    v = g[:]
    if v.dtype.kind in "SO":
        return np.array([x.decode() if isinstance(x, bytes) else x for x in v])
    return v


def cell_table(f):
    """Per-cell native bounding box, from uns/measured_df keyed by (tile, corrID).

    `cell_bbox` is the crop window that actually gets resized to 100x100, stored as a
    string "(min_row, min_col, max_row, max_col)". Area alone understates elongated
    cells, so the bbox is the honest measure of how many native pixels a cell spans.
    """
    cid = obs_col(f, "cell_id")
    tile = obs_col(f, "tile_nr").astype(int)
    md = f["uns"]["measured_df"]
    lut = {}
    for t in md.keys():
        g = md[t]
        bb = [ast.literal_eval(x.decode() if isinstance(x, bytes) else x)
              for x in g["cell_bbox"][:]]
        extra = {k: g[k][:] for k in ("Area", "MajorAxisLength", "MinorAxisLength")
                 if k in g}
        for j, (c, b) in enumerate(zip(g["corresponding_cell"][:], bb)):
            lut[(int(t), int(c))] = (b, {k: v[j] for k, v in extra.items()})
    rows = []
    for i, (c, t) in enumerate(zip(cid, tile)):
        b, ex = lut.get((int(t), int(str(c).split("_")[-1])), (None, {}))
        if not b:
            rows.append(dict(i=i, cell_id=c, bbox_h=np.nan, bbox_w=np.nan)); continue
        rows.append(dict(i=i, cell_id=c, bbox_h=b[2] - b[0], bbox_w=b[3] - b[1],
                         b0=b[0], b1=b[1], **ex))
    d = pd.DataFrame(rows)
    d["scale_h"] = GRID / d.bbox_h          # >1 means the grid oversamples the cell
    d["scale_w"] = GRID / d.bbox_w
    return d


def stage_cells(name):
    cfg = DATA[name]
    with h5py.File(cfg["path"], "r") as f:
        d = cell_table(f)
    d["um_per_grid_px_h"] = d.bbox_h * cfg["um_per_px"] / GRID
    d["um_per_grid_px_w"] = d.bbox_w * cfg["um_per_px"] / GRID
    d.to_csv(OUT / f"cells_{name}.csv", index=False)
    ok = d.dropna(subset=["bbox_h"])
    print(f"[{name}] {len(d):,} cells ({len(ok):,} with a bbox)")
    print(f"  bbox height  median {ok.bbox_h.median():6.1f} px   p99 {ok.bbox_h.quantile(.99):6.1f}   max {ok.bbox_h.max():6.0f}")
    print(f"  bbox width   median {ok.bbox_w.median():6.1f} px   p99 {ok.bbox_w.quantile(.99):6.1f}   max {ok.bbox_w.max():6.0f}")
    print(f"  fraction of cells with EITHER axis over {GRID} px: "
          f"{((ok.bbox_h > GRID) | (ok.bbox_w > GRID)).mean():.4f}")
    print(f"  um per grid pixel: median {np.median([ok.um_per_grid_px_h.median(), ok.um_per_grid_px_w.median()]):.4f}"
          f"   vs one native pixel {cfg['um_per_px']:.4f}")
    return d


def stage_displacement(name, chunk=20_000_000):
    """Positional error the two quantisations introduce, per transcript, in microns.

    err_camera : |round(local) - local|            the native pixel grid
    err_grid   : the extra error from snapping to 100x100, measured back in native px
    Both are reported per axis and as a 2-D magnitude.
    """
    cfg = DATA[name]
    cells = pd.read_csv(OUT / f"cells_{name}.csv")
    with h5py.File(cfg["path"], "r") as f:
        dt = f["obsm"]["detected_transcripts"]
        fl = fields(dt)
        off = dt[fl["local_row"][0]][:]
        lr_d, lc_d = dt[fl["local_row"][1]], dt[fl["local_col"][1]]
        n = off[-1]
        # per-transcript cell index, so each transcript knows its own bbox
        counts = np.diff(off)
        cell_of = np.repeat(np.arange(len(counts)), counts)
        b0 = cells.b0.values; b1 = cells.b1.values
        zh = (GRID / cells.bbox_h.values); zw = (GRID / cells.bbox_w.values)

        acc = {k: [] for k in ("cam_r", "cam_c", "grid_r", "grid_c")}
        kept = 0
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            lr = lr_d[s:e]; lc = lc_d[s:e]
            ci = cell_of[s:e]
            good = np.isfinite(zh[ci]) & np.isfinite(zw[ci])
            lr, lc, ci = lr[good], lc[good], ci[good]
            rr, rc = np.round(lr), np.round(lc)
            cam_r = np.abs(rr - lr); cam_c = np.abs(rc - lc)
            ur = (rr - b0[ci]) * zh[ci]; uc = (rc - b1[ci]) * zw[ci]
            # snap to the grid, then express the move back in native pixels
            grid_r = np.abs(np.rint(ur) - ur) / zh[ci]
            grid_c = np.abs(np.rint(uc) - uc) / zw[ci]
            for k, v in (("cam_r", cam_r), ("cam_c", cam_c),
                         ("grid_r", grid_r), ("grid_c", grid_c)):
                acc[k].append(_hist(v))
            kept += len(lr)
            print(f"  [{name}] {e:,}/{n:,}", flush=True)
    summ = {k: _merge(v) for k, v in acc.items()}
    u = cfg["um_per_px"]
    rows = []
    for k, h in summ.items():
        q = _quant(h, (.5, .9, .99, 1.0))
        rows.append(dict(dataset=name, source=k, n=kept,
                         med_px=q[0], p90_px=q[1], p99_px=q[2], max_px=q[3],
                         med_um=q[0] * u, p90_um=q[1] * u, p99_um=q[2] * u,
                         max_um=q[3] * u))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"displacement_{name}.csv", index=False)
    np.savez_compressed(OUT / f"displacement_hist_{name}.npz",
                        **{k: v for k, v in summ.items()}, um_per_px=u, n=kept)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return d


BINS = np.linspace(0, 1.5, 1501)          # native pixels; 0.001 px resolution


def _hist(v):
    return np.histogram(np.clip(v, 0, BINS[-1] - 1e-9), bins=BINS)[0]


def _merge(hs):
    return np.sum(hs, axis=0)


def _quant(h, qs):
    c = np.cumsum(h) / h.sum()
    mid = (BINS[:-1] + BINS[1:]) / 2
    return [mid[np.searchsorted(c, q)] if q < 1.0 else mid[np.flatnonzero(h)[-1]]
            for q in qs]


def _gene_ids(i, gso, so, ch):
    """Per-transcript gene identity for one cell, without decoding to str."""
    a, b = gso[i], gso[i + 1]
    cs = so[a:b + 1]
    raw = ch[cs[0]:cs[-1]]
    bnd = np.r_[0, np.cumsum(np.diff(cs))]
    return [bytes(raw[bnd[k]:bnd[k + 1]]) for k in range(len(bnd) - 1)]


def stage_occupancy(name, n_cells=1500):
    """Do the transcripts of one (cell, gene) observation even collide on the grid?

    This is the literal form of the question: a 100x100 frame has 10,000 pixels and an
    observation carries tens of transcripts, so the grid should hold the point set
    essentially losslessly. Reported over all observations and over the window the
    detection track actually models (10 < n_spots < 101).
    """
    cfg = DATA[name]
    rng = np.random.RandomState(SEED)
    with h5py.File(cfg["path"], "r") as f:
        dt = f["obsm"]["detected_transcripts"]
        fl = fields(dt)
        off = dt[fl["local_row"][0]][:]
        gso_d, so, ch = string_field(dt, "gene")
        gso = gso_d[:]
        has_isg = "is_gene" in fl
        cells = rng.choice(len(off) - 1, min(n_cells, len(off) - 1), replace=False)
        rows = []
        for i in sorted(cells):
            s, e = off[i], off[i + 1]
            if e == s:
                continue
            sr = dt[fl["subcell_row"][1]][s:e]
            sc = dt[fl["subcell_col"][1]][s:e]
            g = _gene_ids(i, gso, so, ch)
            d = pd.DataFrame(dict(gene=g, row=sr, col=sc))
            if has_isg:
                d = d[dt[fl["is_gene"][1]][s:e].astype(bool)]
            for gn, grp in d.groupby("gene", sort=False):
                px = pd.Series(list(zip(grp.row.values, grp.col.values)))
                mult = px.map(px.value_counts())
                # `n_pixels` says how many distinct pixels were used; `n_conflicted` counts
                # the transcripts actually sharing one, which is the rate a reader wants --
                # two transcripts on one pixel lose one position but involve two spots
                rows.append((len(grp), px.nunique(), int((mult > 1).sum())))
    O = pd.DataFrame(rows, columns=["n_spots", "n_pixels", "n_conflicted"])
    O.to_csv(OUT / f"occupancy_{name}.csv", index=False)
    for lbl, sel in (("all", O.n_spots > 0),
                     ("modelled 10<n<101", (O.n_spots > 10) & (O.n_spots < 101))):
        d = O[sel]
        print(f"[{name}] {lbl}: n={len(d):,}  conflict-free observations "
              f"{(d.n_conflicted == 0).mean():.4f}  TRANSCRIPTS IN CONFLICT "
              f"{d.n_conflicted.sum() / d.n_spots.sum():.5f}"
              f"  median occupancy {d.n_spots.median() / (GRID * GRID):.5f}")
    return O


def stage_boundary(name, n_cells=2500):
    """Could a one-pixel error move a transcript into another compartment?

    Distance to the nuclear boundary per transcript, and the flip rate under a full
    one-pixel shift -- the union over the 4-neighbourhood, so an upper bound rather
    than a typical case.
    """
    from scipy.ndimage import distance_transform_edt
    cfg = DATA[name]
    rng = np.random.RandomState(SEED)
    with h5py.File(cfg["path"], "r") as f:
        dt = f["obsm"]["detected_transcripts"]
        fl = fields(dt)
        off = dt[fl["local_row"][0]][:]
        dm = f["obsm"]["dapi_mask"]
        cells = rng.choice(len(off) - 1, min(n_cells, len(off) - 1), replace=False)
        out, used = [], 0
        for i in sorted(cells):
            s, e = off[i], off[i + 1]
            if e == s:
                continue
            m = (np.asarray(dm[i], float).reshape(GRID, GRID) > 0)
            if m.all() or not m.any():          # no boundary to be near
                continue
            din, dout = distance_transform_edt(m), distance_transform_edt(~m)
            sr = np.clip(dt[fl["subcell_row"][1]][s:e], 0, GRID - 1)
            sc = np.clip(dt[fl["subcell_col"][1]][s:e], 0, GRID - 1)
            inside = m[sr, sc]
            dist = np.where(inside, din[sr, sc], dout[sr, sc])
            flip = np.zeros(len(sr), bool)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                flip |= (m[np.clip(sr + dr, 0, GRID - 1),
                           np.clip(sc + dc, 0, GRID - 1)] != inside)
            out.append(pd.DataFrame(dict(dist=dist, inside=inside, flip=flip)))
            used += 1
    D = pd.concat(out, ignore_index=True)
    D.to_csv(OUT / f"boundary_{name}.csv.gz", index=False, compression="gzip")
    print(f"[{name}] {len(D):,} transcripts over {used:,} cells with a nuclear boundary")
    print(f"  distance to boundary: median {D.dist.median():.2f} grid px   "
          f"within 1 px {(D.dist <= 1).mean():.4f}   flips under 1-px shift {D.flip.mean():.4f}")
    return D


STAGES = {"cells": stage_cells, "displacement": stage_displacement,
          "occupancy": stage_occupancy, "boundary": stage_boundary}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["cells", "displacement"])
    ap.add_argument("--datasets", nargs="+", default=["apex", "ileum"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for st in a.stages:
        for ds in a.datasets:
            print(f"=== {st} / {ds} ===", flush=True)
            STAGES[st](ds)
