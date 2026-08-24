"""Per-experiment configuration and all paths derived from it.

Everything that used to live as a wall of module-level globals at the top of
``processXeniumIntoSingleGeneImages_template.py`` now lives here as a single
``PipelineConfig`` you load from a small YAML file (see ``configs/example.yaml``).

``PipelineConfig`` holds the *inputs*. ``Paths`` holds every output location and
per-tile filename convention derived from those inputs, in one place, so no step
has to hand-build a path string ever again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class ExtraChannel:
    """An optional extra image channel (e.g. a protein stain) to extract per cell.

    ``count_spots`` mirrors the old ``count_spots_on_them`` flag: some channels
    are only cut out per cell, others additionally have their blobs counted.
    """

    path: Path
    count_spots: bool = False

    def __post_init__(self):
        self.path = Path(self.path)

    @property
    def prefix(self) -> str:
        return self.path.stem


class Paths:
    """All output directories, checkpoint files and per-tile filename helpers.

    Built once from a :class:`PipelineConfig`. Holds only plain values (so it
    stays picklable for the multiprocessing steps); it keeps no reference back
    to the config.
    """

    def __init__(self, cfg: "PipelineConfig"):
        self.dapi_prefix = cfg.dapi_path.stem
        self.cellbound2_prefix = cfg.cellbound2_path.stem
        # MERFISH ships Cellbound1/2/3 stains and the old MERFISH template summed 2+3
        # for segmentation. Xenium has no third boundary channel, so this stays None
        # there -- which is what the orphaned cellbound3 references in the old Xenium
        # template were: MERFISH code left behind by a copy-paste.
        self.cellbound3_prefix = (
            cfg.cellbound3_path.stem if cfg.cellbound3_path is not None else None
        )
        self.segmented_prefix = (
            "xenium_segmentation" if cfg.use_xenium_segmentation else "dapi-intra-rna"
        )
        self.extra_channel_prefixes = [ec.prefix for ec in cfg.extra_channels]

        self.basedir = cfg.basedir
        self.img_dir = cfg.basedir / "images"
        self.out_dir = cfg.basedir / "parsed"

        # Top-level checkpoint artefacts
        self.parsed_transcripts_file = self.out_dir / "detected_transcripts_parsed.parquet"
        self.gene_map_file = self.out_dir / "gene_map.json"
        self.adata_file = self.out_dir / "adata.h5ad"
        self.adata_subcell_file = self.out_dir / "adata_subcell.h5ad"

        # Tiled intermediate tree
        self.tiled_dir = self.out_dir / "tiled"
        self.tile_grid_file = self.tiled_dir / "tile_grid.pickle"
        self.dapi_tiled_dir = self.tiled_dir / "dapi_tiles"
        self.cellbound2_tiled_dir = self.tiled_dir / "cellbound2_tiles"
        self.cellbound3_tiled_dir = self.tiled_dir / "cellbound3_tiles"
        self.detected_transcripts_tiled_dir = self.tiled_dir / "detected_transcripts"
        self.labeled_dapi_tiled_dir = self.tiled_dir / "labeled_dapi"
        self.labeled_cellbound_tiled_dir = self.tiled_dir / "labeled_cellbound"
        self.measured_df_dir = self.tiled_dir / "measured_df"
        self.count_matrix_dir = self.tiled_dir / "count_matrices"
        self.count_matrix_spots_dir = self.tiled_dir / "count_matrices_with_spots_counted"

        # Per-cell outputs
        self.single_cells_dir = self.out_dir / "single_cells"
        self.coordinate_parsed_dir = self.out_dir / "coordinate_parsed"

        # Full-resolution Xenium label masks exported from the cells zarr
        self.labeled_dapi_full = self.img_dir / f"{self.dapi_prefix}_labeled.tif"
        self.labeled_cellbound_full = self.img_dir / f"{self.cellbound2_prefix}_labeled.tif"

    # --- per-tile filename conventions ------------------------------------
    def dapi_tile(self, n: int) -> Path:
        return self.dapi_tiled_dir / f"{self.dapi_prefix}_tile{n}.tif"

    def cellbound2_tile(self, n: int) -> Path:
        return self.cellbound2_tiled_dir / f"{self.cellbound2_prefix}_tile{n}.tif"

    def cellbound3_tile(self, n: int):
        """MERFISH only; None when the platform has no third boundary channel."""
        if self.cellbound3_prefix is None:
            return None
        return self.cellbound3_tiled_dir / f"{self.cellbound3_prefix}_tile{n}.tif"

    def labeled_dapi_tile(self, n: int) -> Path:
        return self.labeled_dapi_tiled_dir / f"{self.dapi_prefix}_labeled_tile{n}.tif"

    def labeled_cellbound_tile(self, n: int) -> Path:
        return self.labeled_cellbound_tiled_dir / f"{self.cellbound2_prefix}_labeled_tile{n}.tif"

    def measured_csv(self, n: int) -> Path:
        return self.measured_df_dir / f"{self.dapi_prefix}_{self.segmented_prefix}_measured_tile{n}.csv"

    def transcripts_tile(self, n: int) -> Path:
        return self.detected_transcripts_tiled_dir / f"detected_transcripts_transformed_tile{n}.parquet"

    def count_matrix_tile(self, n: int) -> Path:
        return self.count_matrix_dir / f"count_matrix_tile{n}.csv"

    def count_matrix_concat(self) -> Path:
        return self.count_matrix_dir / "concat_count_matrix.csv"

    def count_matrix_spots_tile(self, n: int) -> Path:
        return self.count_matrix_spots_dir / f"count_matrix_spots_counted_tile{n}.csv"

    def count_matrix_spots_concat(self) -> Path:
        return self.count_matrix_spots_dir / "concat_count_matrix.csv"

    def single_cells_tile_dir(self, n: int) -> Path:
        # Per-cell diagnostic layout (only written when write_single_cell_images).
        return self.single_cells_dir / f"tile{n}"

    def single_cells_archive(self, n: int) -> Path:
        """Compact per-tile checkpoint: per-cell dapi/cell masks (+ extra channels)."""
        return self.single_cells_dir / f"tile{n}_cells.npz"

    def single_cells_transcripts(self, n: int) -> Path:
        """All cells' subcell transcripts for one tile, in a single parquet."""
        return self.single_cells_dir / f"tile{n}_transcripts.parquet"

    def subcell_transcripts_zoomed(self, n: int) -> Path:
        """One tile's resized (zoomed) per-cell transcripts, concatenated."""
        return self.coordinate_parsed_dir / f"tile{n}_transcripts_zoomed.parquet"

    def coordinate_parsed_tile_dir(self, n: int) -> Path:
        return self.coordinate_parsed_dir / f"tile{n}"

    def subcell_npz(self, n: int) -> Path:
        """Compact per-tile checkpoint: sparse embeddings + dapi/cell masks."""
        return self.coordinate_parsed_dir / f"tile{n}_subcell.npz"

    def extra_channel_tiled_dir(self, prefix: str) -> Path:
        return self.tiled_dir / f"{prefix}_tiles"

    def extra_channel_tile(self, prefix: str, n: int) -> Path:
        return self.extra_channel_tiled_dir(prefix) / f"{prefix}_tile{n}.tif"


@dataclass
class PipelineConfig:
    """Everything that varies between experiments.

    Required: ``basedir`` and ``model_path``. Everything else has the same
    defaults the old template shipped with. Image/transcript file locations are
    derived from ``basedir`` but can be overridden explicitly in the YAML.
    """

    basedir: Path
    model_path: Path

    # Which instrument produced the data. This selects the transcript parser and the
    # default input-file layout; the rest of the pipeline is platform-agnostic.
    platform: str = "xenium"

    # Segmentation behaviour. use_xenium_segmentation is Xenium-only (it reuses the
    # instrument's own masks from cells.zarr.zip); MERFISH always segments with cellpose.
    use_xenium_segmentation: bool = True
    keep_unmatched_nuclei: bool = False
    expand_segmentation: int = 0
    seg_diameter: int = 100

    # Storage: when False (default) single-cell crops are kept only as compact
    # per-tile archives that the later steps read. Set True to ALSO dump the old
    # per-cell tif/parquet layout (single_*_masks/, detected_transcripts/,
    # actual_images/) for visual diagnostics -- downstream always reads the archive.
    write_single_cell_images: bool = False

    # Log a per-step storage-added + (averaged/peak) memory summary at the end.
    # Off by default; samples process-tree RSS during each step (needs psutil) and
    # walks the output dir between steps, so it adds a little overhead.
    profile_resources: bool = False

    # When False (default) the bulky tiled images (dapi/cellbound tiles, and in
    # Xenium mode the label-mask tiles + exported full masks) are deleted once the
    # pipeline completes; they are regenerated from source if an early step is
    # re-run. Set True to keep them for diagnostics / visualization.
    keep_tiled_images: bool = False

    # Tiling
    tile_size: Tuple[int, int] = (10000, 10000)

    # Compute
    gpu_id: int = 0
    # None (default) = use the GPU if one is available, otherwise fall back to CPU.
    # Set true to require a GPU, false to force CPU. Without this the pipeline could not
    # run at all on a machine without CUDA, which is most macOS/Windows installs.
    use_gpu: bool = None
    n_jobs_count_matrix: int = 48
    n_jobs_single_cells: int = 12
    n_jobs_count_spots: int = 48

    # Coordinate-parsing / autoencoder input
    coord_target_shape: Tuple[int, int] = (100, 100)
    coord_sigma: float = 1.5

    # A gene is embedded for a cell only when its spot count is strictly between
    # these bounds (spot_count_lower < count < spot_count_upper). Applies to both
    # the real and the randomised-control embeddings.
    spot_count_lower: int = 10
    spot_count_upper: int = 101

    # Randomised control: also embed each gene after scattering its spots to
    # random pixels inside the cell mask, into layers["random_subcell_embedding"].
    # random_seed makes it reproducible.
    compute_random_control: bool = True
    random_seed: int = None

    # Embeddings are batched on the GPU this many images per forward pass. Raise
    # to better saturate a roomy GPU; lower it on CUDA out-of-memory.
    embed_batch_size: int = 256

    # Optional extra channels to extract (and optionally count spots on)
    extra_channels: List[ExtraChannel] = field(default_factory=list)

    # Input file locations (default to the standard Xenium layout under basedir)
    detected_transcripts_file: Path = None
    experiment_path: Path = None
    cells_path: Path = None
    cells_parquet: Path = None
    dapi_path: Path = None
    cellbound2_path: Path = None
    morphology_focus_ome: Path = None
    # MERFISH only
    cellbound3_path: Path = None
    transform_path: Path = None

    def __post_init__(self):
        self.basedir = Path(self.basedir)
        self.model_path = Path(self.model_path)
        self.tile_size = tuple(self.tile_size)
        self.coord_target_shape = tuple(self.coord_target_shape)

        self.platform = str(self.platform).lower()
        if self.platform not in ("xenium", "merfish"):
            raise ValueError(
                f"platform must be 'xenium' or 'merfish', got {self.platform!r}"
            )
        if self.platform == "merfish" and self.use_xenium_segmentation:
            # Silently ignoring this would produce a confusing crash later, when the
            # pipeline went looking for cells.zarr.zip in a MERSCOPE output.
            raise ValueError(
                "use_xenium_segmentation is Xenium-only; set it to false for "
                "platform: merfish (MERFISH is always segmented with cellpose)."
            )

        b = self.basedir
        if self.platform == "xenium":
            self.detected_transcripts_file = Path(
                self.detected_transcripts_file or b / "transcripts.parquet"
            )
            self.experiment_path = Path(self.experiment_path or b / "experiment.xenium")
            self.dapi_path = Path(
                self.dapi_path or b / "images" / "morphology_focus_0000_dapi.tif"
            )
            self.cellbound2_path = Path(
                self.cellbound2_path
                or b / "images" / "morphology_focus_0000_intra-rna.tif"
            )
        else:  # merfish -- MERSCOPE output layout
            self.detected_transcripts_file = Path(
                self.detected_transcripts_file or b / "detected_transcripts.csv"
            )
            self.transform_path = Path(
                self.transform_path
                or b / "images" / "micron_to_mosaic_pixel_transform.csv"
            )
            self.dapi_path = Path(
                self.dapi_path or b / "images" / "mosaic_DAPI_z3.tif"
            )
            self.cellbound2_path = Path(
                self.cellbound2_path or b / "images" / "mosaic_Cellbound2_z3.tif"
            )
            self.cellbound3_path = Path(
                self.cellbound3_path or b / "images" / "mosaic_Cellbound3_z3.tif"
            )
            # Not used on this platform, but Paths still references them.
            self.experiment_path = Path(self.experiment_path or b / "experiment.xenium")

        self.cells_path = Path(self.cells_path or b / "cells.zarr.zip")
        self.cells_parquet = Path(self.cells_parquet or b / "cells.parquet")
        self.morphology_focus_ome = Path(
            self.morphology_focus_ome
            or b / "morphology_focus" / "morphology_focus_0000.ome.tif"
        )

        self.extra_channels = [
            ec if isinstance(ec, ExtraChannel) else ExtraChannel(**ec)
            for ec in self.extra_channels
        ]

        self.paths = Paths(self)

    # Convenience accessors kept for readability in the steps
    @property
    def tile_size_row(self) -> int:
        return self.tile_size[0]

    @property
    def tile_size_col(self) -> int:
        return self.tile_size[1]

    @classmethod
    def from_yaml(cls, path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
