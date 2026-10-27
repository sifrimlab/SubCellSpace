#!/usr/bin/env python
# Provenance for additional_analyses/composite_patterns.ipynb.
# Produces latents.npy and metadata.csv (shipped in the data bundle under composite_patterns/):
# extracts the composite simulations from combined_patterns.zip, selects the pure anchors from
# the original simulation tree, and encodes both with the published model in one pass.
# Expects the original simulation tree and the zip at the paths below, and a GPU; it is not
# turnkey and is kept so the step can be inspected.
"""Embed composite (two-pattern) simulations alongside their pure parents.

Real RNAs can combine two localisation classes. The composite simulations
(`combined_patterns.zip`, simulated in 2024 with the training-data generator) place two
patterns in the same cell; this script produces the latents that quantify where such
composites land in the embedding.

Both constituents of every composite were simulated at pattern-proportion 0.90;
composition varies only through the spot counts n1:n2, so the mixing fraction
f2 = n2/(n1+n2) correlates with total spot count by construction (r ~ 0.80). That is
a property of the design, not a confound to correct for.

Scope, fixed with the user:
  * six constituent patterns -- `foci` and `nuclear-edge` are excluded as ambiguous,
    which leaves C(6,2) = 15 pairs;
  * the 64 held-out cell ids only, so no observation comes from a cell the model
    trained on (composites were generated over all 317);
  * no classifier anywhere -- the question is about the geometry of the embedding.

Pure anchors come from `all_pattern_more_random`, the only rotated tree holding all
six patterns (the `unbalanced` benchmark split has no protrusion), and include
`random` on top of the six constituents. Composite and pure observations are encoded
in one pass so they share a space exactly.

One thing measurement settled early: position along the axis joining two pure
centroids is dominated by spot count, not by pattern -- pure `perinuclear` alone
slides from t = -0.06 below 20 spots to t = 1.59 above 70, which is the whole
between-centroid distance. Every comparison downstream is therefore made at matched
spot count.
"""
import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]   # the SubCellSpace repository
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
from revbench import cvae_encode as ce  # noqa: E402

SIM = Path("/mnt/data/david/simulated_cells/with_dapi")
ZIP = SIM / "combined_patterns.zip"
PURE_TREE = SIM / "even_less_spots/coordinate_parsed_zoomed_again/all_pattern_more_random"
PURE_DAPI = SIM / "even_less_spots/rotated_dapi_zoomed_again"
TEST_IDS = REPO / "training_subcellspace/create_simulations/test_cell_ids.csv"
MODEL = REPO / "SubCellSpace"

OUT = Path("/path/to/work/composite/work")
COMP_DIR = OUT / "sim"                       # extracted composite tiffs
ZGENE = "combined_patterns/coordinate_parsed/"
ZDAPI = "combined_patterns/coordinate_parsed/dapi_zoomed/"

# the six constituents composites are built from
PATTERNS = ["cell-edge", "extranuclear", "intranuclear",
            "pericellular", "perinuclear", "protrusion"]
# `random` is never a constituent, but it has to be in the pure reference set: a
# composite of two complementary patterns (intranuclear + extranuclear, say) is
# spots everywhere, which is what `random` *is*. Leaving it out forces such a
# composite onto a wrong nearest class and makes the comparison meaningless.
ANCHORS = PATTERNS + ["random"]
DROP = {"foci", "nuclear-edge"}

COMP_RX = re.compile(
    r"^(?P<combo>[a-z\-x]+)_(?P<it>\d+)it_(?P<n1>\d+)spots1_(?P<n2>\d+)spots2"
    r"_cellID(?P<cell_id>\d+)_prop1-(?P<p1>[\d.]+)_prop2-(?P<p2>[\d.]+)"
    r"_gene_expression_rotated(?P<rotation>\d+)_blurred1\.5$")
PURE_RX = re.compile(
    r"^(?P<pattern>[a-z\-]+)_(?P<it>\d+)it_(?P<n_spots>\d+)spots_icell(?P<cell_id>\d+)"
    r"_\d+it_prop(?P<prop>[\d.]+)_gene_expression_rotated(?P<rotation>\d+)_blurred1\.5$")


def split_combo(combo):
    """`extranuclearxcell-edge` -> ('extranuclear', 'cell-edge').

    Split on the `x` that leaves two known pattern names either side. Needed
    because `extranuclear` and `cell-edge` contain no `x` but `foci` pairs and the
    separator do, so a naive `combo.split('x')` mangles them.
    """
    for a in PATTERNS + sorted(DROP):
        if combo.startswith(a) and combo[len(a):len(a) + 1] == "x":
            b = combo[len(a) + 1:]
            if b in PATTERNS or b in DROP:
                return a, b
    return None, None


def held_out_ids():
    return {l.strip() for l in TEST_IDS.read_text().splitlines() if l.strip()}


def composite_table(zf, test):
    """Stems of composite observations we keep, with their parsed metadata."""
    dapi, gene = set(), []
    for n in zf.namelist():
        if not n.endswith(".tif"):
            continue
        if n.startswith(ZDAPI):
            dapi.add(n[len(ZDAPI):-len("_DAPI.tif")])
        elif n.startswith(ZGENE):
            gene.append(n[len(ZGENE):-4])

    rows, skipped = [], {"unparsed": 0, "dropped_pattern": 0,
                         "not_held_out": 0, "no_dapi": 0}
    for s in gene:
        m = COMP_RX.match(s)
        if not m:
            skipped["unparsed"] += 1
            continue
        a, b = split_combo(m["combo"])
        if a is None:
            skipped["unparsed"] += 1
            continue
        if a in DROP or b in DROP:
            skipped["dropped_pattern"] += 1
            continue
        if m["cell_id"] not in test:
            skipped["not_held_out"] += 1
            continue
        if s not in dapi:
            skipped["no_dapi"] += 1
            continue
        n1, n2 = int(m["n1"]), int(m["n2"])
        rows.append(dict(
            obs_key=s, kind="composite", pattern=m["combo"],
            pattern_a=a, pattern_b=b, n1=n1, n2=n2, n_spots=n1 + n2,
            f2=(n2 / (n1 + n2)) if (n1 + n2) else np.nan,
            degenerate=(n1 == 0 or n2 == 0),
            cell_id=m["cell_id"], rotation=int(m["rotation"]), it=int(m["it"]),
            prop=float(m["p1"])))
    return pd.DataFrame(rows), skipped


def pure_table(test):
    dapi = set(os.listdir(PURE_DAPI))
    rows, skipped = [], {"unparsed": 0, "dropped_pattern": 0,
                         "not_held_out": 0, "no_dapi": 0}
    for f in os.listdir(PURE_TREE):
        if not f.endswith(".tif"):
            continue
        s = f[:-4]
        m = PURE_RX.match(s)
        if not m:
            skipped["unparsed"] += 1
            continue
        if m["pattern"] not in ANCHORS:
            skipped["dropped_pattern"] += 1
            continue
        if m["cell_id"] not in test:
            skipped["not_held_out"] += 1
            continue
        if (s + "_DAPI.tif") not in dapi:
            skipped["no_dapi"] += 1
            continue
        rows.append(dict(
            obs_key=s, kind="pure", pattern=m["pattern"],
            pattern_a=m["pattern"], pattern_b=None,
            n1=int(m["n_spots"]), n2=0, n_spots=int(m["n_spots"]),
            f2=np.nan, degenerate=False,
            cell_id=m["cell_id"], rotation=int(m["rotation"]), it=int(m["it"]),
            prop=float(m["prop"])))
    return pd.DataFrame(rows), skipped


def extract(zf, stems):
    """Pull only the composite tiffs we need, gene and DAPI, skipping what is there."""
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    (COMP_DIR / "dapi_zoomed").mkdir(exist_ok=True)
    todo = []
    for s in stems:
        if not (COMP_DIR / f"{s}.tif").exists():
            todo.append(ZGENE + f"{s}.tif")
        if not (COMP_DIR / "dapi_zoomed" / f"{s}_DAPI.tif").exists():
            todo.append(ZDAPI + f"{s}_DAPI.tif")
    print(f"extracting {len(todo):,} files ({len(stems):,} stems) -> {COMP_DIR}", flush=True)
    for i, member in enumerate(todo, 1):
        rel = member[len("combined_patterns/coordinate_parsed/"):]
        target = COMP_DIR / rel
        with zf.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
        if i % 10000 == 0:
            print(f"  {i:,}/{len(todo):,}", flush=True)
    return len(todo)


def check_format(paths, label):
    """Assert the images really are what the pure tree holds, rather than trusting names."""
    from skimage.io import imread
    for p in paths:
        a = imread(str(p))
        assert a.shape == (100, 100), f"{label}: {p} has shape {a.shape}"
        assert a.dtype == np.float64, f"{label}: {p} has dtype {a.dtype}"
        assert np.isfinite(a).all(), f"{label}: {p} has non-finite values"
        assert a.min() >= 0.0, f"{label}: {p} has negative values"
    print(f"  {label}: {len(paths)} files, 100x100 float64, finite, non-negative  OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--smoke", type=int, default=0,
                    help="encode only this many rows per kind, write nothing")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    test = held_out_ids()
    assert len(test) == 64, f"expected 64 held-out cell ids, got {len(test)}"

    zf = zipfile.ZipFile(ZIP)
    comp, comp_skip = composite_table(zf, test)
    pure, pure_skip = pure_table(test)
    print(f"composite: {len(comp):,} observations, {comp['pattern'].nunique()} pairs   skipped {comp_skip}")
    print(f"pure     : {len(pure):,} observations, {pure['pattern'].nunique()} patterns  skipped {pure_skip}")
    assert comp_skip["unparsed"] == 0 and pure_skip["unparsed"] == 0
    assert comp_skip["no_dapi"] == 0 and pure_skip["no_dapi"] == 0
    assert comp["pattern"].nunique() == 15, comp["pattern"].nunique()
    assert sorted(pure["pattern"].unique()) == sorted(ANCHORS)

    if args.smoke:
        comp = comp.groupby("pattern", group_keys=False).head(max(1, args.smoke // 15))
        pure = pure.groupby("pattern", group_keys=False).head(max(1, args.smoke // 6))
        print(f"SMOKE: {len(comp)} composite + {len(pure)} pure")

    if not args.skip_extract:
        extract(zf, comp["obs_key"].tolist())

    comp_gene = [COMP_DIR / f"{s}.tif" for s in comp["obs_key"]]
    comp_dapi = [COMP_DIR / "dapi_zoomed" / f"{s}_DAPI.tif" for s in comp["obs_key"]]
    pure_gene = [PURE_TREE / f"{s}.tif" for s in pure["obs_key"]]
    pure_dapi = [PURE_DAPI / f"{s}_DAPI.tif" for s in pure["obs_key"]]
    for name, ps in (("composite gene", comp_gene), ("composite dapi", comp_dapi),
                     ("pure gene", pure_gene), ("pure dapi", pure_dapi)):
        missing = [p for p in ps[:2000] if not p.exists()]
        assert not missing, f"{name}: {len(missing)} missing, e.g. {missing[0]}"
    print("all referenced files present (checked first 2,000 of each)")

    print("format check:")
    check_format(comp_gene[:5], "composite gene")
    check_format(pure_gene[:5], "pure gene")

    model = ce.load_model(str(MODEL), device=args.device)
    s = ce.model_summary(model, path=MODEL)
    print("model:", {k: s[k] for k in ("latent_dim", "n_channels") if k in s})
    assert s["latent_dim"] == 15, s
    assert s["n_channels"] == 2, s

    print(f"encoding {len(pure):,} pure ...", flush=True)
    z_pure = ce.encode_image_paths(model, [str(p) for p in pure_gene],
                                   [str(p) for p in pure_dapi], mode="mu", progress=True)
    print(f"encoding {len(comp):,} composite ...", flush=True)
    z_comp = ce.encode_image_paths(model, [str(p) for p in comp_gene],
                                   [str(p) for p in comp_dapi], mode="mu", progress=True)
    assert z_pure.shape == (len(pure), 15) and z_comp.shape == (len(comp), 15)

    if args.smoke:
        print("SMOKE OK -- latents", z_pure.shape, z_comp.shape,
              "finite:", bool(np.isfinite(z_pure).all() and np.isfinite(z_comp).all()))
        return

    meta = pd.concat([pure, comp], ignore_index=True)
    Z = np.vstack([z_pure, z_comp]).astype(np.float32)
    assert len(meta) == len(Z)
    np.save(OUT / "latents.npy", Z)
    meta.to_csv(OUT / "metadata.csv", index=False)
    json.dump({"n_pure": int(len(pure)), "n_composite": int(len(comp)),
               "pure_tree": str(PURE_TREE), "model": str(MODEL),
               "device": args.device, "mode": "mu",
               "constituents": PATTERNS, "anchors": ANCHORS,
               "dropped": sorted(DROP)},
              open(OUT / "encode_provenance.json", "w"), indent=2)
    print(f"\nwrote {OUT}/latents.npy {Z.shape} and metadata.csv")


if __name__ == "__main__":
    main()
