"""The eleven concrete pipeline steps, in execution order.

Each step is a small :class:`PipelineStep`. The actual scientific work is still
done by the unchanged helpers in ``src/`` -- this module only orchestrates them
and decides what can be skipped.

Two kinds of skipping happen:

  * **Step-level** -- ``is_complete`` reports whether the step's checkpoint
    already exists, so the runner can resume from the furthest point reached.
  * **Tile-level** -- inside a step, each tile's work is guarded by its own
    output-file check, so an interrupted step resumes tile-by-tile.

The handful of steps that fan out across tiles with joblib use module-level
worker functions (``_*_tile``) that take only the picklable
:class:`PipelineConfig` -- never the live context with its model/GPU handles.
"""

from __future__ import annotations

import json
import re
from glob import glob

import numpy as np
import pandas as pd
import anndata as ad
import awkward as ak
import torch
import zarr
import imagesize
from skimage import io
from tifffile import TiffFile, imwrite
from skimage.exposure import match_histograms
from skimage.segmentation import expand_labels
from joblib import Parallel, delayed
from tqdm import tqdm

from pyxelperfect.tiling import tile, tileGrid, tileBorder

from .src.make_gene_labeled_image import createGeneMap
from .src.parse_detected_transcripts import (
    parse_xenium_detected_transcripts,
    parse_detected_transcripts as parse_merfish_detected_transcripts,
)
from .src.get_count_matrix import createGeneMatrix, concatCountMatrices
from .src.dual_segment import segmentDualChannel, segmentDapi, combineCellAndDapiInfo
from .src.split_large_images_into_single_cell_images import (
    cutBboxCellsBasedOnDualChannelSegmentation,
)
from .src.count_spots import countSpotsOnExtraChannel, addBlobsToDetectedTranscripts
from .src.add_to_adata import addMeasuredDfsAsObsMetadata
from .src.parse_single_cell_single_gene_coordinate_images import processWithoutZoomingOut

from .config import PipelineConfig
from .context import PipelineContext
from .pipeline import PipelineStep


# ===========================================================================
# Step 1 -- parse detected transcripts
# ===========================================================================
class ParseDetectedTranscripts(PipelineStep):
    name = "parse_transcripts"
    description = "Parse raw transcripts into pixel coordinates"

    def is_complete(self, ctx):
        return ctx.paths.parsed_transcripts_file.is_file()

    def run(self, ctx):
        cfg, p = ctx.config, ctx.paths
        p.out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.platform == "xenium":
            # Xenium stores microns; the experiment file carries the pixel size.
            parsed = parse_xenium_detected_transcripts(
                cfg.detected_transcripts_file, cfg.experiment_path
            )
        else:
            # MERFISH ships an explicit micron -> mosaic-pixel affine transform.
            parsed = parse_merfish_detected_transcripts(
                cfg.detected_transcripts_file, cfg.transform_path
            )
        parsed.to_parquet(p.parsed_transcripts_file, index=False)


# ===========================================================================
# Step 2 -- build the gene -> integer-label map
# ===========================================================================
class BuildGeneMap(PipelineStep):
    name = "build_gene_map"
    description = "Build the {gene_name: label} map from parsed transcripts"

    def is_complete(self, ctx):
        return ctx.paths.gene_map_file.is_file()

    def run(self, ctx):
        p = ctx.paths
        parsed = pd.read_parquet(p.parsed_transcripts_file)
        gene_map = createGeneMap(parsed)
        with open(p.gene_map_file, "w") as f:
            json.dump(gene_map, f, indent=4)


# ===========================================================================
# Step 3 -- export full-res masks, tile every image, build the tile grid
# ===========================================================================
class BuildTileGrid(PipelineStep):
    name = "build_tile_grid"
    description = "Tile DAPI/cellbound/mask images and build the tile grid"

    def is_complete(self, ctx):
        return ctx.paths.tile_grid_file.is_file()

    def run(self, ctx):
        cfg, p = ctx.config, ctx.paths
        p.tiled_dir.mkdir(parents=True, exist_ok=True)

        rowdiv, coldiv, full_rows, full_cols, image_shape = _write_image_tiles(ctx)

        grid = tileGrid(rowdiv, coldiv, full_rows, full_cols, image_shape)
        with open(p.tile_grid_file, "wb") as f:
            import pickle
            pickle.dump(grid, f)
        ctx.grid = grid  # hand it to downstream steps without a reload


# Channel order of the Xenium multi-modal morphology focus image, in the order the
# instrument writes them.
_MORPHOLOGY_CHANNELS = ("dapi", "cellbound", "intra-rna", "intra-prot")


def _ensure_morphology_channels(ctx):
    """Split the multi-channel morphology OME-TIFF into single-channel tiffs.

    Written channel-by-channel on purpose. The file is an OME pyramid whose axes are
    CYX (channels FIRST) -- indexing it as ``img[:, :, n]`` slices columns, not
    channels -- and a full read is ~25 GB for a typical sample, so it is never
    materialised in one go.
    """
    cfg, p, log = ctx.config, ctx.paths, ctx.log
    if cfg.dapi_path.is_file():
        return

    with TiffFile(cfg.morphology_focus_ome) as tf:
        series = tf.series[0]
        axes, shape = series.axes, series.shape

        if "C" not in axes:
            # Older XOA (< 2.0) ships a single-channel DAPI morphology image, so there
            # is no boundary stain to derive cellbound2_path from. Say so here rather
            # than failing later on a path that was never going to exist.
            raise ValueError(
                f"{cfg.morphology_focus_ome} has axes {axes} (shape {shape}) -- no "
                "channel axis, so it looks like a single-channel (DAPI-only) morphology "
                "image from an older Xenium Onboard Analysis version.\n"
                "This pipeline segments on DAPI plus a second stain, so set dapi_path and "
                "cellbound2_path in the config explicitly to the images you have."
            )

        c_axis = axes.index("C")
        n_channels = shape[c_axis]
        if n_channels < len(_MORPHOLOGY_CHANNELS):
            raise ValueError(
                f"{cfg.morphology_focus_ome} has {n_channels} channels (axes {axes}), "
                f"expected {len(_MORPHOLOGY_CHANNELS)} "
                f"({', '.join(_MORPHOLOGY_CHANNELS)}). Set dapi_path and cellbound2_path "
                "explicitly if your morphology image is laid out differently."
            )

        log.info(
            "Splitting morphology OME-TIFF (axes %s, shape %s) into single channels",
            axes, shape,
        )
        # One page per channel is the normal layout here, and reading a single page
        # keeps peak memory at one channel. Otherwise fall back to a lazy zarr view
        # and slice the channel axis out of it -- still one channel at a time.
        one_page_per_channel = len(series.pages) == n_channels
        lazy = None if one_page_per_channel else zarr.open(series.aszarr(), mode="r")

        for i, name in enumerate(_MORPHOLOGY_CHANNELS):
            out = p.img_dir / f"morphology_focus_0000_{name}.tif"
            if out.is_file():
                continue
            if one_page_per_channel:
                channel = series.pages[i].asarray()
            else:
                index = [slice(None)] * len(shape)
                index[c_axis] = i
                channel = lazy[tuple(index)]
            imwrite(out, channel)
            log.info("  wrote %s (%s)", out.name, channel.shape)
            del channel


def _export_xenium_masks(ctx):
    """Export the Xenium nucleus (0) and cell (1) label masks from cells.zarr
    to the full-res label tifs (skipped if they already exist)."""
    cfg, p, log = ctx.config, ctx.paths, ctx.log
    if p.labeled_dapi_full.is_file() and p.labeled_cellbound_full.is_file():
        return
    log.info("Exporting Xenium segmentation masks from %s", cfg.cells_path)
    z = zarr.open(str(cfg.cells_path), mode="r")
    io.imsave(p.labeled_dapi_full, np.array(z["masks"]["0"]))
    io.imsave(p.labeled_cellbound_full, np.array(z["masks"]["1"]))


def _write_image_tiles(ctx):
    """Tile the DAPI/cellbound (and, in Xenium mode, label-mask) images, plus any
    extra channels. Returns (rowdiv, coldiv, full_rows, full_cols, image_shape) of
    the DAPI tiling so the caller can build the tile grid. Regenerable: callable
    again later to recreate tiles deleted by the storage cleanup."""
    cfg, p, log = ctx.config, ctx.paths, ctx.log
    p.tiled_dir.mkdir(parents=True, exist_ok=True)
    p.dapi_tiled_dir.mkdir(exist_ok=True)
    p.cellbound2_tiled_dir.mkdir(exist_ok=True)
    p.img_dir.mkdir(exist_ok=True)

    if cfg.platform == "xenium":
        _ensure_morphology_channels(ctx)
        if cfg.use_xenium_segmentation:
            _export_xenium_masks(ctx)

    log.info("Tiling DAPI")
    rowdiv, coldiv, full_rows, full_cols = tile(
        str(cfg.dapi_path), cfg.tile_size_row, cfg.tile_size_col, out_dir=p.dapi_tiled_dir,
    )

    try:
        tile(str(cfg.cellbound2_path), cfg.tile_size_row, cfg.tile_size_col,
             out_dir=p.cellbound2_tiled_dir)
        log.info("Tiled cellbound2")
    except Exception:
        log.info("No cellbound2 image found -- skipping")

    if cfg.cellbound3_path is not None:
        p.cellbound3_tiled_dir.mkdir(exist_ok=True)
        try:
            tile(str(cfg.cellbound3_path), cfg.tile_size_row, cfg.tile_size_col,
                 out_dir=p.cellbound3_tiled_dir)
            log.info("Tiled cellbound3")
        except Exception:
            log.info("No cellbound3 image found -- segmenting on cellbound2 alone")

    if cfg.use_xenium_segmentation:
        p.labeled_dapi_tiled_dir.mkdir(exist_ok=True)
        p.labeled_cellbound_tiled_dir.mkdir(exist_ok=True)
        tile(str(p.labeled_dapi_full), cfg.tile_size_row, cfg.tile_size_col,
             out_dir=p.labeled_dapi_tiled_dir)
        tile(str(p.labeled_cellbound_full), cfg.tile_size_row, cfg.tile_size_col,
             out_dir=p.labeled_cellbound_tiled_dir)

    for ec in cfg.extra_channels:
        ec_dir = p.extra_channel_tiled_dir(ec.prefix)
        ec_dir.mkdir(exist_ok=True)
        tile(str(ec.path), cfg.tile_size_row, cfg.tile_size_col, out_dir=ec_dir)

    width, height = imagesize.get(cfg.dapi_path)
    return rowdiv, coldiv, full_rows, full_cols, (height, width)


def _image_tiles_present(cfg):
    """True if the tiles that _write_image_tiles produces are on disk."""
    p = cfg.paths
    dirs = [p.dapi_tiled_dir, p.cellbound2_tiled_dir]
    if cfg.use_xenium_segmentation:
        dirs += [p.labeled_dapi_tiled_dir, p.labeled_cellbound_tiled_dir]
    dirs += [p.extra_channel_tiled_dir(ec.prefix) for ec in cfg.extra_channels]
    return all(d.is_dir() and any(d.iterdir()) for d in dirs)


def _ensure_image_tiles(ctx):
    """Regenerate the image tiles from source if the storage cleanup removed them.
    A no-op on a normal forward run (the tiles are still present)."""
    if _image_tiles_present(ctx.config):
        return
    ctx.log.info("Image tiles missing -- regenerating from source")
    _write_image_tiles(ctx)


def _cleanup_tiled_images(ctx):
    """Delete the bulky, regenerable image-tile dirs (and the full-res label masks
    exported for Xenium mode). Leaves the grid, transcripts, count matrices and
    measured_df in place. Cellpose-segmented label tiles are kept (not cheaply
    regenerable)."""
    import shutil
    cfg, p, log = ctx.config, ctx.paths, ctx.log

    targets = [p.dapi_tiled_dir, p.cellbound2_tiled_dir]
    targets += [p.extra_channel_tiled_dir(ec.prefix) for ec in cfg.extra_channels]
    if cfg.use_xenium_segmentation:
        # Regenerable from cells.zarr; in cellpose mode these are the segmentation
        # output, so they would NOT be safe to delete.
        targets += [p.labeled_dapi_tiled_dir, p.labeled_cellbound_tiled_dir]

    for d in targets:
        if d.is_dir():
            shutil.rmtree(d)
            log.info("Removed tiled images: %s", d)
    if cfg.use_xenium_segmentation:
        for f in (p.labeled_dapi_full, p.labeled_cellbound_full):
            if f.is_file():
                f.unlink()
                log.info("Removed full-res label mask: %s", f)


# ===========================================================================
# Step 4 -- assign each detected transcript to its tile
# ===========================================================================
class TileDetectedTranscripts(PipelineStep):
    name = "tile_transcripts"
    description = "Split parsed transcripts into per-tile parquet files"

    def is_complete(self, ctx):
        p = ctx.paths
        if not p.detected_transcripts_tiled_dir.is_dir() or not ctx.grid_available:
            return False
        # A directory alone is not proof: run() creates it up front, so a killed run
        # would otherwise be mistaken for a finished one.
        return all(
            p.transcripts_tile(n).is_file()
            for n in range(1, ctx.grid.n_tiles + 1)
        )

    def run(self, ctx):
        p = ctx.paths
        grid = ctx.grid
        parsed = pd.read_parquet(p.parsed_transcripts_file)
        grid.addDataCoordinates(parsed)
        p.detected_transcripts_tiled_dir.mkdir(exist_ok=True)
        for tile_nr in range(1, grid.n_tiles + 1):
            tile_df = grid.getTileDataCoordinates(tile_nr, rowname="row", colname="col")
            tile_df.to_parquet(p.transcripts_tile(tile_nr), index=False)


# ===========================================================================
# Step 5 -- segment every tile (sequential: shares one GPU)
# ===========================================================================
class SegmentImages(PipelineStep):
    name = "segment"
    description = "Segment nuclei and cells per tile, measure them"

    def is_complete(self, ctx):
        p = ctx.paths
        if not p.measured_df_dir.is_dir() or not ctx.grid_available:
            return False
        # _segment_tile returns early for tiles whose dapi is entirely blank, so a
        # missing measured csv is only conclusive when its labeled dapi is missing too.
        return all(
            p.measured_csv(n).is_file() or not p.labeled_dapi_tile(n).is_file()
            for n in range(1, ctx.grid.n_tiles + 1)
        )

    def run(self, ctx):
        cfg, p = ctx.config, ctx.paths
        _ensure_image_tiles(ctx)  # regenerate tiles if a prior run cleaned them
        grid = ctx.grid

        # Tiling tolerates a missing cellbound image, but cellpose dual-channel
        # segmentation cannot: fail here with something actionable rather than on an
        # unguarded imread several tiles in. Datasets vary -- MERSCOPE runs may ship
        # Cellbound2/3, PolyT, or neither -- so point cellbound2_path at whatever
        # second stain you have.
        # Unconditional: _segment_tile reads the cellbound tile before it branches on
        # the segmentation mode, so an instrument-mask run needs it just as much.
        if not p.cellbound2_tile(1).is_file():
            raise FileNotFoundError(
                f"No cellbound tiles found (expected {p.cellbound2_tile(1)}).\n"
                f"cellbound2_path is {cfg.cellbound2_path}, which does not exist.\n"
                "Every segmentation mode reads this stain (cellpose segments on it, and "
                "the instrument-mask path still crops it), so set cellbound2_path in the "
                "config to the membrane / intra-RNA / PolyT image for this sample."
            )

        p.labeled_dapi_tiled_dir.mkdir(exist_ok=True)
        p.labeled_cellbound_tiled_dir.mkdir(exist_ok=True)
        p.measured_df_dir.mkdir(exist_ok=True)

        # Sequential on purpose: cellpose does not parallelise cleanly on one GPU.
        for tile_nr in tqdm(range(1, grid.n_tiles + 1)):
            _segment_tile(ctx, tile_nr)


def _segment_tile(ctx: PipelineContext, tile_nr: int):
    """Segment one tile. Each output is guarded so an interrupted run resumes."""
    cfg, p = ctx.config, ctx.paths

    dapi_tile = io.imread(p.dapi_tile(tile_nr))
    if np.amax(dapi_tile) == 0:  # empty tile, nothing to segment
        return
    cellbound_tile = io.imread(p.cellbound2_tile(tile_nr))

    # MERFISH: fold in the third boundary stain if it was tiled. This is what the old
    # MERFISH template did; on Xenium cellbound3_tile() is None and this is skipped.
    cb3 = p.cellbound3_tile(tile_nr)
    if cb3 is not None and cb3.is_file():
        cellbound_tile = cellbound_tile + io.imread(cb3)

    labeled_cellbound_path = p.labeled_cellbound_tile(tile_nr)
    labeled_dapi_path = p.labeled_dapi_tile(tile_nr)

    if not labeled_cellbound_path.is_file():
        labeled_cellbound = segmentDualChannel(
            cellbound_tile, dapi_tile, diameter=cfg.seg_diameter, gpu=ctx.use_gpu, device=ctx.device
        )
        if cfg.expand_segmentation > 0:
            labeled_cellbound = expand_labels(labeled_cellbound, cfg.expand_segmentation)
        io.imsave(labeled_cellbound_path, labeled_cellbound, check_contrast=False)
    else:
        labeled_cellbound = io.imread(labeled_cellbound_path)

    if not labeled_dapi_path.is_file():
        labeled_dapi = segmentDapi(
            dapi_tile, diameter=cfg.seg_diameter, gpu=ctx.use_gpu, device=ctx.device
        )
        if cfg.expand_segmentation > 0:
            labeled_dapi = expand_labels(labeled_dapi, cfg.expand_segmentation)
        io.imsave(labeled_dapi_path, labeled_dapi, check_contrast=False)
    else:
        labeled_dapi = io.imread(labeled_dapi_path)

    if not p.measured_csv(tile_nr).is_file():
        measured = combineCellAndDapiInfo(
            labeled_cellbound, labeled_dapi, ctx.grid, tile_nr,
            keep_unmatched_dapis=cfg.keep_unmatched_nuclei,
        )
        if not measured.empty:
            measured.to_csv(p.measured_csv(tile_nr), index=False)


# ===========================================================================
# Step 6 -- per-tile count matrices, then concatenate
# ===========================================================================
class MakeCountMatrix(PipelineStep):
    name = "count_matrix"
    description = "Build per-tile count matrices and concatenate them"

    def is_complete(self, ctx):
        # The concat is the artifact BuildAdata actually reads; the directory on its
        # own only means run() got started.
        return ctx.paths.count_matrix_concat().is_file()

    def run(self, ctx):
        cfg, p = ctx.config, ctx.paths
        _ensure_image_tiles(ctx)  # regenerate tiles if a prior run cleaned them
        grid = ctx.grid
        p.count_matrix_dir.mkdir(exist_ok=True)

        Parallel(n_jobs=cfg.n_jobs_count_matrix, backend="multiprocessing")(
            delayed(_count_matrix_tile)(cfg, tile_nr)
            for tile_nr in range(1, grid.n_tiles + 1)
        )

        concat_df = concatCountMatrices(str(p.count_matrix_dir / "count_matrix_tile*.csv"))
        concat_df.to_csv(p.count_matrix_concat())


def _count_matrix_tile(cfg: PipelineConfig, tile_nr: int):
    """Build the count matrix for one tile and tag transcripts with their cell."""
    p = cfg.paths
    if not p.measured_csv(tile_nr).is_file():
        return
    measured_df = pd.read_csv(p.measured_csv(tile_nr))
    labeled = io.imread(p.labeled_cellbound_tile(tile_nr))
    df = pd.read_parquet(p.transcripts_tile(tile_nr))

    count_matrix_df, adapted_decoded_df = createGeneMatrix(
        labeled, df, rowname="local_row", colname="local_col",
        measured_df=measured_df, key_to_match_in_labeled_image="corresponding_cell",
    )
    count_matrix_df.index = f"tile{tile_nr}_" + count_matrix_df.index

    count_matrix_df.to_csv(p.count_matrix_tile(tile_nr))
    adapted_decoded_df.to_parquet(p.transcripts_tile(tile_nr), index=False)


# ===========================================================================
# Step 7 -- cut every cell out into its own single-cell images
# ===========================================================================
class CutSingleCells(PipelineStep):
    name = "cut_single_cells"
    description = "Cut each cell into a compact per-tile archive (+ optional diagnostic crops)"

    def is_complete(self, ctx):
        d = ctx.paths.single_cells_dir
        if not d.is_dir() or not ctx.grid_available:
            return False
        return all(
            ctx.paths.single_cells_archive(t).is_file()
            for t in range(1, ctx.grid.n_tiles + 1)
        )

    def run(self, ctx):
        cfg, p = ctx.config, ctx.paths
        _ensure_image_tiles(ctx)  # regenerate tiles if a prior run cleaned them
        grid = ctx.grid
        p.single_cells_dir.mkdir(exist_ok=True)

        Parallel(n_jobs=cfg.n_jobs_single_cells, backend="multiprocessing")(
            delayed(_cut_single_cells_tile)(cfg, tile_nr)
            for tile_nr in range(1, grid.n_tiles + 1)
        )


def _to_object_array(arrays):
    """Pack a list of differently-shaped ndarrays into a 1-D object array
    (np.array(list, dtype=object) would try to broadcast equal-shaped ones)."""
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def _save_cells_archive(cfg, tile_nr, records):
    """Persist one tile's single-cell crops compactly: per-cell dapi/cell masks
    (and extra-channel crops) in a compressed npz, and all cells' subcell
    transcripts in a single parquet."""
    p = cfg.paths
    cell_ids = np.array([r["cell_id"] for r in records], dtype=np.int64)
    arrays = {
        "cell_ids": cell_ids,
        "dapi": _to_object_array([r["dapi_mask"] for r in records]),
        "cell": _to_object_array([r["cell_mask"] for r in records]),
    }
    for ec in cfg.extra_channels:
        arrays[f"extra_{ec.prefix}"] = _to_object_array(
            [r["extra"].get(ec.prefix) for r in records]
        )
    np.savez_compressed(p.single_cells_archive(tile_nr), **arrays)

    if records:
        transcripts = pd.concat([r["transcripts"] for r in records], ignore_index=True)
    else:
        transcripts = pd.DataFrame()
    transcripts.to_parquet(p.single_cells_transcripts(tile_nr), index=False)


def _cut_single_cells_tile(cfg: PipelineConfig, tile_nr: int):
    """Cut all cells of one tile into the compact archive (+ optional diagnostics)."""
    p = cfg.paths

    if p.single_cells_archive(tile_nr).is_file():  # per-tile checkpoint
        return

    # The tile may have been empty (never segmented) or matched nothing.
    if (not p.labeled_dapi_tile(tile_nr).is_file()) or (not p.measured_csv(tile_nr).is_file()):
        _save_cells_archive(cfg, tile_nr, [])
        return

    dapi_labeled_tile = io.imread(p.labeled_dapi_tile(tile_nr))
    cellbound_labeled_tile = io.imread(p.labeled_cellbound_tile(tile_nr))
    measured_tile_df = pd.read_csv(p.measured_csv(tile_nr))
    transcripts_tile_df = pd.read_parquet(p.transcripts_tile(tile_nr))

    extra_tiles = [
        (prefix, io.imread(p.extra_channel_tile(prefix, tile_nr)))
        for prefix in p.extra_channel_prefixes
    ]

    # The RGB visualisation crop is purely diagnostic; only build it (and the
    # per-cell tif/parquet layout) when the user wants the inspectable files.
    write_files = cfg.write_single_cell_images
    origin_image = None
    if write_files:
        p.single_cells_tile_dir(tile_nr).mkdir(exist_ok=True)
        dapi_tile = io.imread(p.dapi_tile(tile_nr))
        cellbound_tile = io.imread(p.cellbound2_tile(tile_nr))
        # Match the composite used for segmentation (MERFISH sums the two boundary
        # stains), so the diagnostic image shows what was actually segmented.
        cb3 = p.cellbound3_tile(tile_nr)
        if cb3 is not None and cb3.is_file():
            cellbound_tile = cellbound_tile + io.imread(cb3)
        origin_image = np.zeros((*dapi_tile.shape, 3))
        origin_image[:, :, 0] = dapi_tile / np.amax(dapi_tile)
        origin_image[:, :, 1] = match_histograms(
            cellbound_tile / np.amax(cellbound_tile), dapi_tile / np.amax(dapi_tile)
        )

    # Cells touching the tile border are skipped (handled when tiles are stitched).
    border = tileBorder(cellbound_labeled_tile, tile_nr=tile_nr)

    records = cutBboxCellsBasedOnDualChannelSegmentation(
        measured_tile_df,
        dapi_labeled_image=dapi_labeled_tile,
        cell_labeled_image=cellbound_labeled_tile,
        detected_transcripts=transcripts_tile_df,
        origin_image=origin_image,
        border=border,
        image_prefix=f"tile{tile_nr}_{p.segmented_prefix}_",
        expand=0,
        out_dir=p.single_cells_tile_dir(tile_nr),
        extra_channels=extra_tiles,
        write_files=write_files,
        return_cells=True,
    )

    _save_cells_archive(cfg, tile_nr, records)


# ===========================================================================
# Step 8 -- count spots on extra channels (no-op when there are none)
# ===========================================================================
class CountSpots(PipelineStep):
    name = "count_spots"
    description = "Count blobs on extra channels and fold into the count matrix"

    def is_complete(self, ctx):
        # Only the on-disk checkpoint counts towards resume position. When there
        # are no extra channels this dir is never created and run() is a no-op;
        # the resume point is then driven by the later steps' checkpoints, just
        # like in the original script.
        if not ctx.config.extra_channels:
            # run() is a no-op here and writes nothing, so there is no checkpoint.
            # Report NOT complete: resume_index is max(completed)+1, so returning
            # True would jump the pointer past every earlier step on a fresh run.
            # Re-"running" this step is free, and the pointer is driven by the later
            # steps' checkpoints instead.
            return False
        return ctx.paths.count_matrix_spots_concat().is_file()

    def run(self, ctx):
        cfg, p = ctx.config, ctx.paths
        if not cfg.extra_channels:
            ctx.log.info("No extra channels configured -- skipping spot counting")
            return

        grid = ctx.grid
        p.count_matrix_spots_dir.mkdir(exist_ok=True)

        Parallel(n_jobs=cfg.n_jobs_count_spots, backend="multiprocessing")(
            delayed(_count_spots_tile)(cfg, tile_nr)
            for tile_nr in tqdm(range(1, grid.n_tiles + 1))
        )

        concat_df = concatCountMatrices(
            str(p.count_matrix_spots_dir / "count_matrix_spots_counted_tile*.csv")
        )
        concat_df.to_csv(p.count_matrix_spots_concat())


def _count_spots_tile(cfg: PipelineConfig, tile_nr: int):
    """Count blobs per cell on each spot-counted extra channel for one tile.

    Reads the extra-channel crops and masks from the compact per-tile archive,
    and augments the per-tile transcripts parquet with the detected blobs.
    """
    p = cfg.paths
    if not p.measured_csv(tile_nr).is_file():
        return
    if not p.count_matrix_tile(tile_nr).is_file():
        return
    if not p.single_cells_archive(tile_nr).is_file():
        return

    # The grid is needed by addBlobsToDetectedTranscripts; load it locally so the
    # worker stays picklable (it only receives the config).
    import pickle
    with open(p.tile_grid_file, "rb") as f:
        grid = pickle.load(f)

    count_matrix = pd.read_csv(p.count_matrix_tile(tile_nr), index_col=0)
    measured_df = pd.read_csv(p.measured_csv(tile_nr))
    assert len(count_matrix) == len(measured_df)

    data = np.load(p.single_cells_archive(tile_nr), allow_pickle=True)
    idx_of = {int(c): i for i, c in enumerate(data["cell_ids"])}
    dapi_arrs, cell_arrs = data["dapi"], data["cell"]

    transcripts_df = pd.read_parquet(p.single_cells_transcripts(tile_nr))
    groups = ({int(k): v.copy() for k, v in transcripts_df.groupby("cell_label_assigned")}
              if not transcripts_df.empty else {})

    for ec in cfg.extra_channels:
        if not ec.count_spots:
            continue  # e.g. PolyT is extracted but not counted

        extra_arrs = data[f"extra_{ec.prefix}"]
        channel_count = []
        for row in measured_df.itertuples():
            if row.Area <= 30:  # too small to be an isolated cell
                channel_count.append(0)
                continue

            cell_id = row.corresponding_cell
            i = idx_of.get(cell_id)
            if i is None:
                # Cell on a tile border is not isolated -> not in the archive.
                channel_count.append(0)
                continue

            channel_img = extra_arrs[i]
            mask = cell_arrs[i] if cell_id != 0 else dapi_arrs[i]

            filtered_dots = countSpotsOnExtraChannel(channel_img, mask)
            channel_count.append(len(filtered_dots))

            if cell_id in groups:
                groups[cell_id] = addBlobsToDetectedTranscripts(
                    filtered_dots, groups[cell_id], grid, tile_nr, measured_df, cell_id, ec.prefix
                )

        count_matrix[ec.prefix] = channel_count

    # Write the blob-augmented transcripts back (coordinate parsing reads these).
    if groups:
        pd.concat(list(groups.values()), ignore_index=True).to_parquet(
            p.single_cells_transcripts(tile_nr), index=False
        )
    count_matrix.to_csv(p.count_matrix_spots_tile(tile_nr))


# ===========================================================================
# Step 9 -- assemble the AnnData object from the count matrix
# ===========================================================================
class BuildAdata(PipelineStep):
    name = "build_adata"
    description = "Assemble AnnData (counts, obs metadata, gene labels)"

    def is_complete(self, ctx):
        return ctx.paths.adata_file.is_file()

    def run(self, ctx):
        cfg, p = ctx.config, ctx.paths

        concat_file = (
            p.count_matrix_spots_concat() if cfg.extra_channels else p.count_matrix_concat()
        )
        count_matrix = pd.read_csv(concat_file, index_col=0)

        adata = ad.AnnData(count_matrix)
        adata.obs_names = list(count_matrix.index)
        adata.var_names = list(count_matrix.columns)

        adata.obs["name"] = list(adata.obs_names)
        adata.obs["tile_nr"] = [int(re.findall(r"tile(\d+)", n)[0]) for n in adata.obs["name"]]
        adata.obs["local_row"] = [int(re.findall(r"row(\d+)", n)[0]) for n in adata.obs["name"]]
        adata.obs["local_col"] = [int(re.findall(r"col(\d+)", n)[0]) for n in adata.obs["name"]]
        adata.obs["cell_id"] = list(adata.obs_names)
        adata.var["gene_name"] = list(adata.var_names)

        adata.var["gene_label"] = self._gene_labels(adata, p)

        adata.uns["measured_df"] = {
            str(tile_nr): pd.read_csv(p.measured_csv(tile_nr))
            for tile_nr in adata.obs["tile_nr"].unique()
        }
        adata = addMeasuredDfsAsObsMetadata(adata)

        # With Xenium's own segmentation, name cells by their original Xenium
        # cell_id (the ids in cells.parquet / cell_by_gene) instead of the
        # synthesized tile/row/col label. Done last, so the synthesized name in
        # obs["cell_id"] (which finalize/measured-df logic still keys off) and
        # the tile/row/col columns are all already computed.
        if cfg.use_xenium_segmentation:
            self._relabel_with_xenium_ids(ctx, adata)

        adata.write_h5ad(p.adata_file, compression="gzip")
        ctx.adata = adata

    def _relabel_with_xenium_ids(self, ctx, adata):
        cfg, log = ctx.config, ctx.log

        label_to_cellid = self._load_xenium_cell_id_map(cfg)

        # The trailing integer of the synthesized name is the cellbound mask
        # value, i.e. the original Xenium integer cell label.
        labels = adata.obs["cell_id"].str.rsplit("_", n=1).str[-1].astype(int)
        xenium_ids = labels.map(label_to_cellid)

        missing = xenium_ids.isna()
        if missing.any():
            # Should not happen with keep_unmatched_nuclei=False (every kept cell
            # is matched to a real Xenium cell). Fall back to the synthesized name.
            log.warning(
                "%d cells have no Xenium cell_id (likely kept unmatched nuclei); "
                "keeping their synthesized name", int(missing.sum())
            )
            xenium_ids = xenium_ids.where(~missing, adata.obs["cell_id"])

        adata.obs["xenium_cell_id"] = xenium_ids.to_numpy()
        adata.obs_names = xenium_ids.astype(str).to_numpy()

        if not adata.obs_names.is_unique:
            n_dup = int(adata.obs_names.duplicated().sum())
            log.warning(
                "%d Xenium cell_ids duplicated across tile borders "
                "(nuclei split between tiles) -> made unique with a suffix",
                n_dup,
            )
            adata.obs_names_make_unique()

    @staticmethod
    def _load_xenium_cell_id_map(cfg):
        """Map each integer mask label to its Xenium cell_id string.

        ``cells.parquet`` holds the cell_id strings (row-aligned with the cell
        order used by cell_by_gene). The mask integer of the i-th cell comes from
        ``seg_mask_value[i]`` when that array exists (Xenium analysis v5); newer
        outputs (e.g. v6) drop it and simply use ``label = row index + 1``, which
        we fall back to (verified by centroid matching on IMU020).
        """
        z = zarr.open(str(cfg.cells_path), mode="r")
        cell_ids = pd.read_parquet(cfg.cells_parquet, columns=["cell_id"])["cell_id"].to_numpy()
        try:
            seg_mask_value = np.asarray(z["seg_mask_value"])
        except KeyError:
            seg_mask_value = np.arange(1, len(cell_ids) + 1)
        return dict(zip(seg_mask_value.tolist(), cell_ids.tolist()))

    def _gene_labels(self, adata, p):
        """Map each var to its integer gene label, minting labels for extra
        ('mosaic') channels and persisting any additions back to gene_map.json."""
        with open(p.gene_map_file) as f:
            gene_map = json.load(f)

        max_label = max(gene_map.values())
        original_max = max_label
        labels = []
        for gene_name in list(adata.var_names):
            if gene_name in gene_map:
                labels.append(gene_map[gene_name])
            elif "mosaic" in gene_name:
                max_label += 1
                gene_map[gene_name] = max_label
                labels.append(max_label)
            else:
                raise KeyError(f"{gene_name} not in gene map, and not a newly counted gene.")

        if max_label > original_max:  # spot counting added new genes -> persist them
            with open(p.gene_map_file, "w") as f:
                json.dump(gene_map, f)

        return labels


# ===========================================================================
# Step 10 -- coordinate-parse single cells into autoencoder input
# ===========================================================================
class CoordinateParse(PipelineStep):
    name = "coordinate_parse"
    description = "Turn per-cell single-gene images into model input arrays"

    def is_complete(self, ctx):
        # Complete only when every tile has its embedding checkpoint, so an
        # interrupted parse resumes tile-by-tile instead of being skipped because
        # the directory merely exists.
        d = ctx.paths.coordinate_parsed_dir
        if not d.is_dir() or not ctx.grid_available:
            return False
        return all(
            ctx.paths.subcell_npz(t).is_file()
            for t in range(1, ctx.grid.n_tiles + 1)
        )

    def run(self, ctx):
        p = ctx.paths
        grid = ctx.grid
        p.coordinate_parsed_dir.mkdir(exist_ok=True)
        # Sequential: shares the GPU-resident model. Each tile is guarded by its
        # own npz checkpoint, so this resumes where an interrupted run left off.
        for tile_nr in range(1, grid.n_tiles + 1):
            _coordinate_parse_tile(ctx, tile_nr)


def _coordinate_parse_tile(ctx: PipelineContext, tile_nr: int):
    cfg, p, log = ctx.config, ctx.paths, ctx.log

    npz_path = p.subcell_npz(tile_nr)
    if npz_path.is_file():  # per-tile checkpoint -> tile already fully parsed
        return

    def _checkpoint(result):
        # Write transcripts first, then the npz, so the npz guard implies both done.
        _save_zoomed_transcripts(p.subcell_transcripts_zoomed(tile_nr), result.get("transcripts", []))
        _save_subcell_npz(npz_path, result)

    if not p.single_cells_archive(tile_nr).is_file():
        log.info("Tile %d skipped, no single-cell archive", tile_nr)
        _checkpoint({"embeddings": [], "dapi_masks": [], "cell_masks": [], "transcripts": []})
        return

    # Rebuild the per-cell inputs from the compact archive (masks) + per-tile
    # transcripts parquet, entirely in memory -- no per-cell files involved.
    cells = _load_cells_from_archive(cfg, tile_nr)
    if not cells:
        log.info("Tile %d skipped, empty archive", tile_nr)
        _checkpoint({"embeddings": [], "dapi_masks": [], "cell_masks": [], "transcripts": []})
        return

    with open(p.gene_map_file) as f:
        gene_map = json.load(f)

    # Per-tile rng so the randomised control is reproducible (when a seed is set)
    # yet independent across tiles.
    rng = (np.random.default_rng(cfg.random_seed + tile_nr)
           if cfg.random_seed is not None else np.random.default_rng())

    out_dir = p.coordinate_parsed_tile_dir(tile_nr)
    with torch.no_grad():
        ctx.model["cvae"].eval()
        result = processWithoutZoomingOut(
            [], out_dir, cfg.coord_target_shape, cfg.coord_sigma, gene_map,
            lower_thresh=cfg.spot_count_lower, upper_thresh=cfg.spot_count_upper,
            model=ctx.model, device=ctx.device, return_arrays=True, cells=cells,
            randomize=cfg.compute_random_control, rng=rng,
            embed_batch_size=cfg.embed_batch_size,
        )

    _checkpoint(result)


def _load_cells_from_archive(cfg, tile_nr):
    """Build the in-memory cells list for processWithoutZoomingOut from the
    compact archive: (stem, transcripts_df, dapi_array, cell_array) per cell."""
    p = cfg.paths
    data = np.load(p.single_cells_archive(tile_nr), allow_pickle=True)
    cell_ids = data["cell_ids"]
    dapi_arrs, cell_arrs = data["dapi"], data["cell"]

    transcripts_df = pd.read_parquet(p.single_cells_transcripts(tile_nr))
    groups = ({int(k): v for k, v in transcripts_df.groupby("cell_label_assigned")}
              if not transcripts_df.empty else {})

    cells = []
    for idx, cid in enumerate(cell_ids):
        cid = int(cid)
        stem = f"tile{tile_nr}_{p.segmented_prefix}_GElabeled_cellID{cid}"
        cells.append((stem, groups.get(cid, pd.DataFrame()), dapi_arrs[idx], cell_arrs[idx]))
    return cells


def _save_zoomed_transcripts(path, transcripts):
    """Concatenate one tile's zoomed per-cell transcripts into a single parquet."""
    if transcripts:
        df = pd.concat([t for _, t in transcripts], ignore_index=True)
    else:
        df = pd.DataFrame()
    df.to_parquet(path, index=False)


def _cell_id_from_stem(stem):
    """Integer Xenium/cellpose cell label parsed from a ``..._cellID<N>`` stem."""
    return int(re.findall(r"cellID(\d+)", stem)[0])


def _save_subcell_npz(npz_path, result):
    """Persist one tile's subcellular outputs compactly.

    ``result`` is the dict returned by processWithoutZoomingOut. We store the
    sparse embeddings as (cell_id, gene_id, vector) and the per-cell dapi/cell
    masks as flat (n_cells, 10000) arrays, all keyed by integer cell label.
    Compressed because the masks are smooth/mostly-constant and pack well.
    """
    def _embeddings_to_arrays(embeddings):
        if embeddings:
            cell_ids = np.array([_cell_id_from_stem(s) for s, _, _ in embeddings], dtype=np.int64)
            gene_ids = np.array([int(g) for _, g, _ in embeddings], dtype=np.int64)
            vectors = np.stack([np.asarray(e).reshape(-1) for _, _, e in embeddings]).astype(np.float32)
        else:
            cell_ids = np.empty(0, dtype=np.int64)
            gene_ids = np.empty(0, dtype=np.int64)
            vectors = np.empty((0, 0), dtype=np.float32)
        return cell_ids, gene_ids, vectors

    dapi_masks = result["dapi_masks"]
    cell_masks = result["cell_masks"]

    emb_cell_ids, gene_ids, emb_vectors = _embeddings_to_arrays(result["embeddings"])
    rand_cell_ids, rand_gene_ids, rand_vectors = _embeddings_to_arrays(result.get("random_embeddings", []))

    mask_cell_ids = np.array([_cell_id_from_stem(s) for s, _ in dapi_masks], dtype=np.int64)
    dapi_arr = (np.stack([m.reshape(-1) for _, m in dapi_masks]).astype(np.float32)
                if dapi_masks else np.empty((0, 0), dtype=np.float32))
    cell_arr = (np.stack([m.reshape(-1) for _, m in cell_masks]).astype(np.float32)
                if cell_masks else np.empty((0, 0), dtype=np.float32))

    np.savez_compressed(
        npz_path,
        emb_cell_ids=emb_cell_ids, gene_ids=gene_ids, emb_vectors=emb_vectors,
        rand_cell_ids=rand_cell_ids, rand_gene_ids=rand_gene_ids, rand_vectors=rand_vectors,
        mask_cell_ids=mask_cell_ids, dapi_masks=dapi_arr, cell_masks=cell_arr,
    )


# ===========================================================================
# Step 11 -- fold single-cell + subcellular info into the final AnnData
# ===========================================================================
class FinalizeAdata(PipelineStep):
    name = "finalize"
    description = "Add single-cell and subcellular embeddings -> adata_subcell.h5ad"

    def is_complete(self, ctx):
        return ctx.paths.adata_subcell_file.is_file()

    def run(self, ctx):
        p = ctx.paths
        adata = ctx.adata

        adata = self._add_detected_transcripts(ctx, adata)
        adata = self._assemble_from_npz(ctx, adata)

        adata.write_h5ad(p.adata_subcell_file, compression="gzip")
        ctx.adata = adata

    @staticmethod
    def _add_detected_transcripts(ctx, adata):
        """Fill obsm['detected_transcripts'] (ragged) from the per-tile zoomed
        transcripts parquets, grouped by cell label. One read + groupby per tile
        rather than one file (or AnnData view) per cell.
        """
        p = ctx.paths

        tile_groups = {}  # tile_nr -> {cell label: per-cell df}
        for tile_nr in adata.obs["tile_nr"].unique():
            tile_nr = int(tile_nr)
            f = p.subcell_transcripts_zoomed(tile_nr)
            if f.is_file():
                df = pd.read_parquet(f)
                tile_groups[tile_nr] = (
                    {int(k): v for k, v in df.groupby("cell_label_assigned")}
                    if not df.empty else {}
                )
            else:
                tile_groups[tile_nr] = {}

        records = []
        for tile_nr, cell_id_str in zip(adata.obs["tile_nr"], adata.obs["cell_id"]):
            corr = int(cell_id_str.split("_")[-1])
            df = tile_groups[int(tile_nr)].get(corr, pd.DataFrame()).copy()
            df.index = df.index.map(str)
            records.append(df.to_dict(orient="list"))
        adata.obsm["detected_transcripts"] = ak.Array(records)
        return adata

    @staticmethod
    def _assemble_from_npz(ctx, adata, nr_latent_dims=15):
        """Fill the dense subcell_embedding layer and the dapi/cell mask obsm
        from the per-tile npz checkpoints (instead of millions of tifs).

        Output formats are unchanged from the original: a dense
        (n_obs, n_var, 15) layer and dense (n_obs, target_pixels) obsm arrays
        (now float32 rather than float64, halving their memory).
        """
        p, log = ctx.paths, ctx.log
        mask_dim = int(np.prod(ctx.config.coord_target_shape))

        # obs is keyed by (tile_nr, integer cell label = trailing token of the
        # synthesized cell_id); var by gene_label. Neither depends on obs_names
        # (which may have been renamed to Xenium ids).
        corr = adata.obs["cell_id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
        tiles = adata.obs["tile_nr"].astype(int).to_numpy()
        obs_lookup = {(int(t), int(c)): i for i, (t, c) in enumerate(zip(tiles, corr))}
        var_lookup = {int(g): j for j, g in enumerate(adata.var["gene_label"].astype(int).to_numpy())}

        layer = np.zeros((*adata.shape, nr_latent_dims), dtype=np.float32)
        dapi_mask = np.zeros((adata.n_obs, mask_dim), dtype=np.float32)
        cell_mask = np.zeros((adata.n_obs, mask_dim), dtype=np.float32)
        want_random = ctx.config.compute_random_control
        rand_layer = np.zeros((*adata.shape, nr_latent_dims), dtype=np.float32) if want_random else None

        def _fill(target, cell_ids, gene_ids, vectors, tile_nr):
            filled = missing = 0
            for cid, gid, vec in zip(cell_ids, gene_ids, vectors):
                oi = obs_lookup.get((tile_nr, int(cid)))
                vj = var_lookup.get(int(gid))
                if oi is None or vj is None:
                    missing += 1
                    continue
                target[oi, vj, :] = vec
                filled += 1
            return filled, missing

        n_emb = n_rand = n_mask = n_missing = 0
        for npz_path in sorted(glob(str(p.coordinate_parsed_dir / "tile*_subcell.npz"))):
            tile_nr = int(re.findall(r"tile(\d+)_subcell", npz_path)[0])
            d = np.load(npz_path)
            f, m = _fill(layer, d["emb_cell_ids"], d["gene_ids"], d["emb_vectors"], tile_nr)
            n_emb += f
            n_missing += m
            if want_random and "rand_vectors" in d:
                f, _ = _fill(rand_layer, d["rand_cell_ids"], d["rand_gene_ids"], d["rand_vectors"], tile_nr)
                n_rand += f
            for cid, dv, cv in zip(d["mask_cell_ids"], d["dapi_masks"], d["cell_masks"]):
                oi = obs_lookup.get((tile_nr, int(cid)))
                if oi is None:
                    continue
                dapi_mask[oi, :] = dv
                cell_mask[oi, :] = cv
                n_mask += 1

        log.info("Filled %d embeddings, %d randomised, %d cell masks (%d embeddings had no match)",
                 n_emb, n_rand, n_mask, n_missing)
        adata.layers["subcell_embedding"] = layer
        if want_random:
            adata.layers["random_subcell_embedding"] = rand_layer
        adata.obsm["dapi_mask"] = dapi_mask
        adata.obsm["cell_mask"] = cell_mask
        return adata


# ===========================================================================
# The pipeline, in order
# ===========================================================================
def build_steps():
    return [
        ParseDetectedTranscripts(),
        BuildGeneMap(),
        BuildTileGrid(),
        TileDetectedTranscripts(),
        SegmentImages(),
        MakeCountMatrix(),
        CutSingleCells(),
        CountSpots(),
        BuildAdata(),
        CoordinateParse(),
        FinalizeAdata(),
    ]
