"""Constants, paths and binning definitions shared by every benchmark.

Values are vendored from the published code so that reproductions match:
SEED and DISTANCE_THRESHOLD come from ``CVAE/source_functions/config.py``,
MIN_OBS_PER_GENE from the minimum-n_obs analysis in
``Interpretation-multiple-adata-hek_invest_classifier_simulated_data.ipynb``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# --- paths -------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = PKG_DIR.parent

# One source of truth for where the data lives: the benchmark's own config.py.
import sys
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
import config as _benchmark

RESULTS_DIR = _benchmark.RESULTS
INFO_DIR = _benchmark.INFO_FILES
PUBLIC_INFO_DIR = _benchmark.INFO_FILES

# The few published functions this library calls are vendored beside it.
PUBLISHED_SRC = BENCHMARK_DIR / "vendor"

# Derived caches (the exploded APEX pair is ~1.1 GB). Written on demand, not
# part of the data bundle.
CACHE_DIR = _benchmark.DATA_ROOT / "cache"

# --- reproducibility ---------------------------------------------------------

SEED = 101

# --- detection ---------------------------------------------------------------

#: Earth-mover threshold calibrated for the 15-dim SubCellSpace latent at
#: ~1000 observations. NOT portable to a latent of a different dimensionality:
#: the cut is a point on a probability-distribution distance and each method's
#: classifier has its own scale.
#:
#: This comment used to say "use the Youden-J optimum from a threshold sweep for
#: cross-method comparison". **Do not.** That optimum is maximised against the
#: evaluation dataset's own ground-truth panel, so it is fitted on the labels
#: being scored and cannot be computed for a dataset that has none -- which is
#: every dataset this is actually applied to. It answers "how well could this
#: rank if someone had already told you the answers", not "how does this
#: workflow perform on data it has not seen".
#:
#: Derive a per-method threshold on SIMULATED data instead:
#: :mod:`revbench.detection.calibrate` reproduces the procedure that produced
#: these two values (the paper's Fig 4D) and runs it for any method, blind to
#: the real panels.
DISTANCE_THRESHOLD = 0.06
MEDIUM_DISTANCE_THRESHOLD = 0.03

#: The published analyses do not share one threshold. Verified against the saved
#: per-gene tables: the HEK/APEX run used 0.06, the ileum run used 0.03
#: (``passes_earth_thresh`` agrees 33/33 at 0.03 and only 26/33 at 0.06).
DATASET_THRESHOLDS = {
    "apex_hek": 0.06,
    "ileum": 0.03,
}


def dataset_threshold(dataset, default=DISTANCE_THRESHOLD):
    """Earth-mover threshold for a named dataset."""
    return DATASET_THRESHOLDS.get(dataset, default)

#: Minimum observations before a gene is testable.
MIN_OBS_PER_GENE = 8

#: The seven log2FC columns fig4 cell 32 uses for the APEX assignment.
#: ``ER_Lumen_log2FC`` exists in the source table and is deliberately excluded,
#: to reproduce the published label exactly.
APEX_FC_COLUMNS: List[str] = [
    "Cytosol_log2FC",
    "ERM_log2FC",
    "Lamina_log2FC",
    "Nuclear_Pore_log2FC",
    "Nucleolus_log2FC",
    "Nucleus_log2FC",
    "OMM_log2FC",
]

ILEUM_FC_COLUMNS: List[str] = [
    "Villus bottom - logFC",
    "Villus mid - logFC",
    "Villus top - logFC",
]

# --- labels ------------------------------------------------------------------

LABEL_KEY = "pattern"
GROUP_KEY = "cell_id"

#: Targets the latent should NOT encode, with the CV scheme each requires.
#: ``cell_id`` must stay ungrouped: grouping on cell_id while predicting it
#: makes the score 0 by construction, which is a vacuous pass rather than
#: evidence. Its interpretation rests on the ratio to the shuffled floor.
NEGATIVE_CONTROLS: Dict[str, str] = {
    "rotation_interval_45": "stratified_group",
    "n_spots_interval_fixed": "stratified_group",
    "cell_id": "stratified",
}

PATTERN_NAMES: List[str] = [
    "cell-edge", "extranuclear", "intranuclear",
    "pericellular", "perinuclear", "protrusion",
]

# --- binning -----------------------------------------------------------------

#: 45-degree rotation bins. Computed inline in fig2 cell 23; not stored on disk.
ROTATION_45_BINS = [0, 45, 90, 135, 180, 225, 270, 315, float("Inf")]
ROTATION_45_LABELS = ["0-45", "45-90", "90-135", "135-180",
                      "180-225", "225-270", "270-315", "315+"]

N_SPOTS_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float("Inf")]

#: The labels as published in ``make_paper_figures/src/adata_utils.py:75``.
#: "60-70" is missing, so bin (60,70] is labelled "70-80" and every label above
#: is shifted one decade. Kept verbatim for reproducibility only.
N_SPOTS_LABELS_PUBLISHED = ["0-10", "10-20", "20-30", "30-40", "40-50",
                            "50-60", "70-80", "80-90", "90-100", "100+"]

#: Corrected labels. Use these for anything new.
N_SPOTS_LABELS_FIXED = ["0-10", "10-20", "20-30", "30-40", "40-50",
                        "50-60", "60-70", "70-80", "80-90", "90+"]


def derive_metadata_columns(obs: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Add the label columns the benchmarks need but the h5ad does not store.

    Adds ``rotation_interval_45`` (absent on disk; only a 6-bin
    ``rotation_interval`` is stored) and ``n_spots_interval_fixed`` (the
    corrected binning). The published ``n_spots_interval`` is left untouched.
    """
    out = obs if inplace else obs.copy()
    if "rotation" in out.columns:
        out["rotation_interval_45"] = pd.cut(
            out["rotation"].astype(float),
            bins=ROTATION_45_BINS, labels=ROTATION_45_LABELS,
        )
    if "n_spots" in out.columns:
        out["n_spots_interval_fixed"] = pd.cut(
            out["n_spots"].astype(float),
            bins=N_SPOTS_BINS, labels=N_SPOTS_LABELS_FIXED,
        )
    return out


# --- provenance --------------------------------------------------------------

def code_version() -> str:
    """Short git SHA of the repository, ``+dirty`` if uncommitted."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def env_versions() -> Dict[str, str]:
    """Versions of everything whose behaviour can move a number."""
    import sys
    mods = {}
    for name in ("numpy", "scipy", "pandas", "sklearn", "anndata", "scanpy"):
        try:
            mods[name] = __import__(name).__version__
        except Exception:
            mods[name] = "absent"
    mods["python"] = sys.version.split()[0]
    return mods
