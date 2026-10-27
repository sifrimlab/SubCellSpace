# Processing your own spatial transcriptomics data

This module takes raw Xenium or MERFISH output and produces an AnnData object that holds your
count matrix plus subcellular information: a **SubCellSpace** embedding per (cell, gene), the
per-cell nucleus and cell masks, and the single-molecule coordinates in `.obsm` for later
plotting.

It is a config-driven pipeline: you copy one YAML file per sample and run it. Nothing needs
editing inside the Python.

## Setup

```bash
conda env create -f ../yamls/pyxel_env_crossplatform.yml
conda activate pyxel_env
```

This file works on Linux, macOS (Intel and Apple Silicon) and Windows. See the
[top-level README](../README.md#option-1--conda-recommended) for why there are two versions of
each environment file. If you have no CUDA GPU everything still runs, just more slowly.

Alternatively skip the environment entirely and use the Docker image, which ships the
environment and mounts this repository:

```bash
docker run -it --rm -v $(pwd):/workspace -v /path/to/your/data:/data \
    woutdavid/subcellspace:latest \
    python -m raw_spatial_data_processing.run raw_spatial_data_processing/configs/mine.yaml
```

## Running it

```bash
# from the repository root
cp raw_spatial_data_processing/configs/example_xenium.yaml \
   raw_spatial_data_processing/configs/mine.yaml
$EDITOR raw_spatial_data_processing/configs/mine.yaml     # set basedir (and platform)

python -m raw_spatial_data_processing.run raw_spatial_data_processing/configs/mine.yaml
```

Only `basedir` and `model_path` are required; every input file location is derived from
`basedir` unless you override it. Use `configs/example_merfish.yaml` for MERFISH.

For a walkthrough that runs these steps one at a time and shows what each produces — including
how to cut a small example tile out of a full sample so a test run takes about a minute — see
[`tutorial_creating_adata_subcell.ipynb`](../tutorial_creating_adata_subcell.ipynb).

| flag | effect |
|------|--------|
| `--status` | print which steps are already done and exit; touches no data |
| `--from STEP` | force the run to restart at that step (e.g. `--from segment`) |
| `--force` | rerun everything from the beginning |

**Resuming is automatic.** Each of the eleven steps knows its own checkpoint, so an
interrupted run picks up where it stopped — and within the expensive steps each *tile* is
guarded separately, so you do not redo finished tiles either. A per-step timing summary is
logged at the end; set `profile_resources: true` for memory and storage figures as well.

Logs are written to the working directory as `<basedir-name>_log_<date>.log`, plus a separate
`<basedir-name>_error_<timestamp>.log` naming the failing step if a run dies.

## The steps

| # | step | what it does |
|---|------|--------------|
| 1 | `parse_transcripts` | raw transcripts → pixel coordinates |
| 2 | `build_gene_map` | `{gene_name: integer label}` map |
| 3 | `build_tile_grid` | tile the images, build the tile grid |
| 4 | `tile_transcripts` | split transcripts into per-tile parquet |
| 5 | `segment` | segment nuclei and cells per tile, measure them |
| 6 | `count_matrix` | per-tile count matrices, then concatenate |
| 7 | `cut_single_cells` | cut each cell out into a compact per-tile archive |
| 8 | `count_spots` | count blobs on extra channels (only if `extra_channels` is set) |
| 9 | `build_adata` | assemble `adata.h5ad` (counts + cell metadata) |
| 10 | `coordinate_parse` | per-(cell, gene) images → embeddings |
| 11 | `finalize` | write `adata_subcell.h5ad` |

## Configuration

The important keys, with defaults. See the example configs for the full annotated list.

| key | default | meaning |
|-----|---------|---------|
| `platform` | `xenium` | `xenium` or `merfish`; selects the transcript parser and input layout |
| `basedir` | *required* | the raw output directory for one sample |
| `model_path` | *required* | the trained SubCellSpace model (`./SubCellSpace` in this repo) |
| `use_xenium_segmentation` | `true` | reuse the instrument's own masks from `cells.zarr.zip`. **Xenium only** — MERFISH always segments with cellpose, and setting this with `platform: merfish` is rejected. When `false`, cells are re-segmented on DAPI + the intra-RNA stain, which in our experience gives the best results. |
| `keep_unmatched_nuclei` | `false` | keep nuclei that segmented successfully but have no matching cell boundary. We discard these, as they tend to indicate segmentation artefacts. |
| `expand_segmentation` | `0` | pixels to expand labels by; default segmentation sometimes underestimates cell boundaries |
| `seg_diameter` | `100` | cellpose diameter |
| `tile_size` | `[10000, 10000]` | tile size in pixels, `[rows, cols]` |
| `compute_random_control` | `true` | also compute `layers["random_subcell_embedding"]`, the matched randomised control |
| `spot_count_lower` / `_upper` | `10` / `101` | the per-(cell, gene) transcript-count window that gets an embedding |
| `write_single_cell_images` | `false` | additionally dump the per-cell tif/parquet layout for visual diagnostics |
| `keep_tiled_images` | `false` | keep the tiled images after the run (they are regenerable, so they are deleted by default) |
| `extra_channels` | `[]` | additional image channels to crop per cell, optionally with `count_spots: true` |

If you used the previous version of this pipeline, the three flags you edited in the script
map to `keep_unmatched_nuclei`, `expand_segmentation` and `use_xenium_segmentation`.

## Output

Everything lands under `<basedir>/parsed/`. The file you want is **`adata_subcell.h5ad`**:

```
obs    name, tile_nr, local_row, local_col, cell_id, global_row, global_col, cell_size
       (+ xenium_cell_id when use_xenium_segmentation is true)
var    gene_name, gene_label
X      transcript counts per (cell, gene)
layers subcell_embedding            (n_cells, n_genes, 15)
       random_subcell_embedding     (n_cells, n_genes, 15)
obsm   dapi_mask, cell_mask         flat 100x100 masks
       detected_transcripts         ragged, per cell; carries gene, subcell_row, subcell_col
uns    measured_df
```

Feed that to the [example notebook](../example_usage_of_SubCellSpace_on_your_data.ipynb) to
analyse your own data, or see the
[tutorial](../tutorial_creating_adata_subcell.ipynb) for a guided tour of this schema.

The intermediates are compact by design: single cells and their embeddings are stored as one
compressed `.npz` per tile rather than as millions of small files (the previous version wrote
~2.8 M tifs / 42 GB for a single sample). Set `write_single_cell_images: true` if you want the
per-cell files anyway for inspection.

### One behaviour change worth knowing

With `write_single_cell_images: false` (the default), unmatched nuclei produce no per-cell
output at all, whereas the previous version always wrote them. This only matters if you set
`keep_unmatched_nuclei: true` *and* want the per-cell files — in that case set
`write_single_cell_images: true` as well.
