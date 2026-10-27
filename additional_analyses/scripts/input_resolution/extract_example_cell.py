#!/usr/bin/env python
"""Extract the one APEX-seq cell drawn in input_resolution.ipynb (section 3, panels a-b).

Writes example_cell_apex.npz next to the other cached tables, so the notebook does not
have to open the 12.7 GB h5ad. Input: the published APEX-seq / HEK293T Xenium object
from the main SubCellSpace data record (10.5281/zenodo.19708995),
`hek293T_Xenium_adata_subcell_including_randomized_embeddings.h5ad`, and `cells_apex.csv`
written by measure_input_resolution.py.

    python extract_example_cell.py --h5ad /path/to/hek293T_..._embeddings.h5ad \
        --cells /path/to/input_resolution/cells_apex.csv --out /path/to/input_resolution/
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

GRID = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cells = pd.read_csv(a.cells).dropna(subset=["bbox_h"])
    with h5py.File(a.h5ad, "r") as f:
        dt = f["obsm"]["detected_transcripts"]
        form = json.loads(dt.attrs["form"])

        def leaf(c):
            while "content" in c:
                c = c["content"]
            return c["form_key"]
        fl = {fd: (c["form_key"], leaf(c)) for fd, c in zip(form["fields"], form["contents"])}
        off = dt[fl["local_row"][0] + "-offsets"][:]
        # a cell with a mid-sized box and enough transcripts to look at
        cand = cells[(cells.bbox_h.between(80, 95)) & (cells.bbox_w.between(80, 95))]
        i = int(cand.i.iloc[len(cand) // 2])
        s, e = off[i], off[i + 1]
        b0, b1 = cells.set_index("i").loc[i, ["b0", "b1"]]
        h, w = cells.set_index("i").loc[i, ["bbox_h", "bbox_w"]]
        lr = dt[fl["local_row"][1] + "-data"][s:e]; lc = dt[fl["local_col"][1] + "-data"][s:e]
        sr = dt[fl["subcell_row"][1] + "-data"][s:e]; sc = dt[fl["subcell_col"][1] + "-data"][s:e]
        mask = np.asarray(f["obsm"]["dapi_mask"][i], float).reshape(GRID, GRID) > 0
        # one model input is ONE gene in one cell, not every transcript in the cell --
        # pick a gene sitting in the middle of the modelled 10 < n_spots < 101 window
        gso = dt[fl["gene"][0] + "-offsets"]
        gform = dict(zip(form["fields"], form["contents"]))["gene"]
        so = dt[gform["content"]["form_key"] + "-offsets"]
        ch = dt[gform["content"]["content"]["form_key"] + "-data"]
        g0, g1 = gso[i], gso[i + 1]
        cs = so[g0:g1 + 1]; raw = ch[cs[0]:cs[-1]]; bnd = np.r_[0, np.cumsum(np.diff(cs))]
        gid = np.array([bytes(raw[bnd[k]:bnd[k + 1]]) for k in range(len(bnd) - 1)])
        uq, cnt = np.unique(gid, return_counts=True)
        ok = uq[(cnt > 10) & (cnt < 101)]
        gsel = gid == ok[len(ok) // 2]

    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "example_cell_apex.npz", cell=i, b0=b0, b1=b1, h=h, w=w,
                        local_row=lr, local_col=lc, subcell_row=sr, subcell_col=sc,
                        dapi_mask=mask, gene_selected=gsel)
    print(f"cell {i}: {len(lr):,} transcripts, {int(gsel.sum())} in the selected gene")


if __name__ == "__main__":
    main()
