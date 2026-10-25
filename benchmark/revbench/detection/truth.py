"""Experimental ground truth for the gene-level detection benchmarks.

Two sources:

* **APEX-seq / HEK** -- proximity-labelling log2 fold-changes across seven
  subcellular compartments. A gene's compartment is the arg-max column.
  Negatives come from a bulk-derived list of genes with no subcellular
  localisation, restricted to the measured panel; genes carrying conflicting
  evidence between the two sources are in neither class.
* **Ileum (Lee et al. 2025)** -- apical/basal polarisation along the villus
  axis. A gene is polarised when all three villus-zone log fold-changes exceed
  the cutoff with the same sign.

Reproduces the definitions used in ``fig4`` (cells 31-36) and ``fig6``
(cells 22-27).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import (APEX_FC_COLUMNS, ILEUM_FC_COLUMNS, INFO_DIR,
                      PUBLIC_INFO_DIR)

ILEUM_SHEET = "17_sIT_MERFISH_volnorm"
ILEUM_HEADER = 4
ILEUM_XLSX_NAME = "13059_2025_3786_MOESM2_ESM.xlsx"
APEX_CSV_NAME = "genes_in_panel_apex_seq_info.csv"
APEX_NEG_NAME = "nonsubcell_genes.txt"


def _find(name: str, *dirs: Path) -> Path:
    """First existing copy of ``name``, searching the given directories."""
    tried = []
    for d in dirs:
        if d is None:
            continue
        p = Path(d) / name
        tried.append(str(p))
        if p.is_file():
            return p
    raise FileNotFoundError("{!r} not found in any of: {}".format(name, tried))


@dataclasses.dataclass(frozen=True)
class GroundTruth:
    """Positives, negatives and continuous truth for one dataset."""
    name: str
    positives: FrozenSet[str]
    negatives: FrozenSet[str]
    labels: Mapping[str, str]      # gene -> compartment / polarity
    scores: Mapping[str, float]    # gene -> continuous truth (log2FC)
    source: str
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        both = self.positives & self.negatives
        if both:
            raise ValueError("{} gene(s) in both classes: {}"
                             .format(len(both), sorted(both)[:5]))

    @property
    def genes(self) -> FrozenSet[str]:
        return self.positives | self.negatives

    def restrict(self, genes: Sequence[str]) -> "GroundTruth":
        """Narrow to a gene set (e.g. the genes actually testable)."""
        keep = set(map(str, genes))
        return dataclasses.replace(
            self,
            positives=frozenset(self.positives & keep),
            negatives=frozenset(self.negatives & keep),
            labels={g: v for g, v in self.labels.items() if g in keep},
            scores={g: v for g, v in self.scores.items() if g in keep},
            params=dict(self.params, restricted_to=len(keep)),
        )

    def __repr__(self) -> str:
        return ("GroundTruth(name={!r}, positives={}, negatives={})"
                .format(self.name, len(self.positives), len(self.negatives)))


def load_apex(info_dir: Optional[Path] = None,
              gene_panel: Optional[Sequence[str]] = None,
              fc_columns: Sequence[str] = tuple(APEX_FC_COLUMNS)) -> GroundTruth:
    """APEX-seq HEK ground truth.

    ``fc_columns`` defaults to the seven compartments used in the published
    assignment. The source table also carries ``ER_Lumen_log2FC``; including it
    changes 2 of 250 assignments, so it is excluded to reproduce the published
    labels exactly and the choice is recorded in ``params``.
    """
    dirs = (info_dir, INFO_DIR, PUBLIC_INFO_DIR) if info_dir else (INFO_DIR, PUBLIC_INFO_DIR)
    csv = _find(APEX_CSV_NAME, *dirs)
    neg_path = _find(APEX_NEG_NAME, *dirs)

    df = pd.read_csv(csv)
    fc_columns = list(fc_columns)
    missing = [c for c in fc_columns if c not in df.columns]
    if missing:
        raise KeyError("missing log2FC columns: {}".format(missing))

    assigned = (df[fc_columns].idxmax(axis=1)
                .str.replace("_log2FC", "", regex=False))
    genes = df["Common_Gene"].astype(str)
    labels = dict(zip(genes, assigned))
    scores = dict(zip(genes, df[fc_columns].max(axis=1).astype(float)))
    positives = set(genes)

    negatives = {g for g in neg_path.read_text().split() if g}

    if gene_panel is not None:
        panel = set(map(str, gene_panel))
        positives &= panel
        negatives &= panel

    # Conflicting evidence between the two sources belongs to neither class.
    ambiguous = positives & negatives
    positives -= ambiguous
    negatives -= ambiguous

    return GroundTruth(
        name="apex_hek",
        positives=frozenset(positives),
        negatives=frozenset(negatives),
        labels={g: v for g, v in labels.items() if g in positives},
        scores={g: v for g, v in scores.items() if g in positives},
        source=str(csv),
        params={"fc_columns": fc_columns, "negatives_file": neg_path.name,
                "n_negatives_file": len(negatives) + len(ambiguous),
                "n_ambiguous_excluded": len(ambiguous),
                "panel_restricted": gene_panel is not None},
    )


def load_ileum(path: Optional[Path] = None, sheet: str = ILEUM_SHEET,
               header: int = ILEUM_HEADER, cutoff: float = 0.5,
               gene_panel: Optional[Sequence[str]] = None,
               tested_genes: Optional[Sequence[str]] = None) -> GroundTruth:
    """Ileum apical/basal ground truth (Lee et al. 2025).

    Polarised = all three villus-zone log fold-changes beyond ``cutoff`` with
    one sign. Negatives follow the published analysis: the complement of the
    polarised set among the genes actually tested (``tested_genes``); without
    that argument the complement is taken over the whole sheet.
    """
    xlsx = Path(path) if path else _find(ILEUM_XLSX_NAME, INFO_DIR, PUBLIC_INFO_DIR)
    df = pd.read_excel(xlsx, sheet_name=sheet, header=header)
    cols = list(ILEUM_FC_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError("missing logFC columns: {}".format(missing))

    df = df.dropna(subset=cols).copy()
    df["gene"] = df["Gene names"].astype(str)
    df["mean_fc"] = df[cols].mean(axis=1)

    apical = (df[cols] > cutoff).all(axis=1)
    basal = (df[cols] < -cutoff).all(axis=1)
    polarised = apical | basal

    labels = dict(zip(df.loc[polarised, "gene"],
                      np.where(apical[polarised], "apical", "basal")))
    scores = dict(zip(df["gene"], df["mean_fc"].astype(float)))

    positives = set(df.loc[polarised, "gene"])
    if tested_genes is not None:
        # Restrict BOTH classes, so the reported counts describe genes that can
        # actually be scored rather than the whole sheet.
        universe = set(map(str, tested_genes))
        positives &= universe
    else:
        universe = set(df["gene"])
    negatives = universe - positives

    if gene_panel is not None:
        panel = set(map(str, gene_panel))
        positives &= panel
        negatives &= panel

    return GroundTruth(
        name="ileum",
        positives=frozenset(positives),
        negatives=frozenset(negatives),
        labels={g: v for g, v in labels.items() if g in positives},
        scores={g: v for g, v in scores.items() if g in positives | negatives},
        source="{}::{}".format(xlsx.name, sheet),
        params={"cutoff": cutoff, "fc_columns": cols, "header": header,
                "negatives": "complement_among_tested" if tested_genes is not None
                             else "complement_in_sheet",
                "panel_restricted": gene_panel is not None},
    )


def truth_frame(gt: GroundTruth, result_df: pd.DataFrame,
                score_col: str = "earth_dist",
                gene_col: str = "gene_name") -> pd.DataFrame:
    """Join a per-gene result table to ground truth.

    Genes with no ground-truth membership are dropped -- they belong to neither
    class and must not be silently counted as positives.
    """
    if score_col not in result_df.columns:
        raise KeyError("{!r} not in result_df; have {}"
                       .format(score_col, list(result_df.columns)))
    out = result_df[[gene_col, score_col]].copy()
    out[gene_col] = out[gene_col].astype(str)
    out["y_true"] = out[gene_col].map(
        lambda g: True if g in gt.positives else (False if g in gt.negatives else None))
    dropped = int(out["y_true"].isna().sum())
    out = out.dropna(subset=["y_true"]).copy()
    out["y_true"] = out["y_true"].astype(bool)
    out["truth_label"] = out[gene_col].map(gt.labels).fillna("")
    out["truth_score"] = out[gene_col].map(gt.scores)
    out.attrs["n_dropped_unlabelled"] = dropped
    out.attrs["truth"] = gt.name
    return out.rename(columns={score_col: "score"})
