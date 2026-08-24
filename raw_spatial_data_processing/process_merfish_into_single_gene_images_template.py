#!/usr/bin/env python
import re
import ast
import sys
import json
import torch
import pickle
import logging
import imagesize
import numpy as np
import pandas as pd
import anndata as ad
import awkward as ak
from glob import glob
from tqdm import tqdm
from skimage import io
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from skimage.feature import blob_log
from skimage.exposure import match_histograms
from skimage.segmentation import expand_labels
from pyxelperfect.tiling import tile, tileGrid, tileBorder
from src.make_gene_labeled_image import create_gene_map
from src.add_to_adata import add_measured_dfs_as_obs_metadata
from src.parse_detected_transcripts import parse_detected_transcripts
from src.get_count_matrix import create_gene_matrix, concat_count_matrices
from src.split_single_cell_into_single_gene import split_labeled_image_into_single_gene_images
from src.dual_segment import segment_dual_channel, segment_dapi, combine_cell_and_dapi_info
from src.split_large_images_into_single_cell_images import cut_bbox_cells_based_on_dual_channel_segmentation
from src.parse_single_cell_single_gene_coordinate_images import match_genes_with_dapis, process_without_zooming_out, match_dapis_with_cell

# Input building

basedir = Path("path_to_region_dir/region_0/")
out_dir = basedir / "parsed"

out_dir.mkdir(exist_ok=True)

detected_transcripts_file = basedir  / "detected_transcripts.csv"
transform_path = basedir  / "images" / "micron_to_mosaic_pixel_transform.csv"
dapi_path = basedir  / "images" / "mosaic_DAPI_z3.tif"
cellbound2_path = basedir  / "images" / "mosaic_Cellbound2_z3.tif"
cellbound3_path = basedir  / "images" / "mosaic_Cellbound3_z3.tif"

# Global Parameters 
KEEP_UNMATCHED_NUCLEI = False # Whether to keep nuclei that do not have a matching cell-boundary segmentation (Especially for subcell embedding this has to be False)
EXPAND_SEGMENTATION = 0 # How many pixels to expand the segmentation with

# Dimensions of the tile grid
tile_size_row = 10000
tile_size_col = 10000

# GPU parameters
gpu_id = 2
device = torch.device(gpu_id)

sys.path.append('../') # need to be able to load CVAE.py
model_path = "../SubCellSpace" # Which model to create the embeddings with

# Extra channels to be taken into accounts in case you want to use them later
extra_channels_to_extract = [
    # basedir / "images" / "mosaic_PolyT_z3.tif",
]

# End input, now we build the filepaths of the output
dapi_prefix = dapi_path.stem
cellbound2_prefix = cellbound2_path.stem
cellbound3_prefix = cellbound3_path.stem

extra_channels_prefixes = []
if extra_channels_to_extract:
    for extra_channel_file in extra_channels_to_extract:
        extra_channels_prefixes.append(extra_channel_file.stem)

# Just a name to give to the segmentation output in case multiple runs need to be compared
segmented_prefix = "dapi-cellbound2-cellbound3"

# Tiled output
tiled_dir =  out_dir / "tiled"
dapi_tiled_dir = tiled_dir / "dapi_tiles"
cellbound2_tiled_dir = tiled_dir / "cellbound2_tiles"
cellbound3_tiled_dir = tiled_dir / "cellbound3_tiles"
detected_transcripts_tiled_out_dir = tiled_dir / "detected_transcripts" 

extra_channels_tiled_dirs = []
if extra_channels_prefixes:
    for extra_channel_stem in extra_channels_prefixes:
        extra_channels_tiled_dirs.append(tiled_dir / f"{extra_channel_stem}_tiles")


# Segmentation output
labeled_dapi_tiled_dir =  tiled_dir / "labeled_dapi"  
labeled_cellbound_tiled_dir =  tiled_dir / "labeled_cellbound"  
measured_df_dir = tiled_dir / "measured_df"

# Count matrix
count_matrix_out_dir = tiled_dir / "count_matrices"

# single cells
single_cells_out_dir = out_dir / "single_cells" 


# Embeddings
coordinate_parsed_out_dir = out_dir / "coordinate_parsed"

def time_now():
    return datetime.now().strftime('%H:%M:%S')

logging.basicConfig(filename=f'{basedir.name}_log_{datetime.now().strftime("%d:%m:%y")}.log', level=logging.INFO, filemode='a')

# Parse detected transcripts to pixel space instead of mosaic (real) space
def get_detected_transcripts(tmp):
    if not (out_dir / "detected_transcripts_parsed.parquet").is_file():
        logging.info(f"{time_now()}: Parsing detected transcripts")

        parsed_detected_transcripts = parse_detected_transcripts(detected_transcripts_file, transform_path)
        parsed_detected_transcripts.to_parquet(out_dir / "detected_transcripts_parsed.parquet", index=False)

        logging.info(f"{time_now()}: Finished parsing detected transcripts")
    else:
        logging.info(f"{time_now()}: Loading parsed detected transcripts")

        parsed_detected_transcripts = pd.read_parquet(out_dir / "detected_transcripts_parsed.parquet")

        logging.info(f"{time_now()}: Parsed detected transcripts loaded")

    return parsed_detected_transcripts

# Create a mapping of gene-label for downstream analysis
def get_gene_map(parsed_detected_transcripts):
    if not (out_dir / "gene_map.json").is_file():

        logging.info(f"{time_now()}: Creating gene map")

        gene_map = create_gene_map(parsed_detected_transcripts)
        with open(out_dir / "gene_map.json", "w") as fp:
            json.dump(gene_map, fp, indent=4)

        logging.info(f"{time_now()}: Finished creating gene map")

# Tile the images into workable tiles
def get_tile_grid(tmp):
    if not (tiled_dir / "tile_grid.pickle").is_file():
        # First we build all output dirs
        tiled_dir.mkdir(exist_ok=True)
        dapi_tiled_dir.mkdir(exist_ok=True)
        cellbound2_tiled_dir.mkdir(exist_ok=True)
        cellbound3_tiled_dir.mkdir(exist_ok=True)

        logging.info(f"{time_now()}: Started tiling")

        rowdiv, coldiv, target_full_rows, target_full_columns = tile(str(dapi_path), tile_size_row , tile_size_col, out_dir = dapi_tiled_dir)

        logging.info(f"{time_now()}: Dapi tiled")

        try:
            rowdiv, coldiv, target_full_rows, target_full_columns = tile(str(cellbound2_path), tile_size_row , tile_size_col, out_dir = cellbound2_tiled_dir)
            logging.info(f"{time_now()}: cellbound2 tiled")
        except:
            logging.info(f"{time_now()}: No cellbound2 found")
        try:
            rowdiv, coldiv, target_full_rows, target_full_columns = tile(str(cellbound3_path), tile_size_row , tile_size_col, out_dir = cellbound3_tiled_dir)
            logging.info(f"{time_now()}: cellbound3 tiled")
        except:
            logging.info(f"{time_now()}: No cellbound3 found")

        if extra_channels_to_extract:
            logging.info(f"{time_now()}: Started tiling extra channels")
            for extra_channel_path, extra_channel_tile_dir in zip(extra_channels_to_extract, extra_channels_tiled_dirs):
                extra_channel_tile_dir.mkdir(exist_ok=True)

                rowdiv, coldiv, target_full_rows, target_full_columns = tile(str(extra_channel_path), tile_size_row , tile_size_col, out_dir = extra_channel_tile_dir)
            logging.info(f"{time_now()}: Finished extra channels tiled")

        width, height = imagesize.get(dapi_path)
        image_shape = (height, width)

        grid = tileGrid(rowdiv, coldiv, target_full_rows, target_full_columns , image_shape)

        with open(tiled_dir / "tile_grid.pickle", 'wb') as f:
            pickle.dump(grid, f)

        logging.info(f"{time_now()}: Finished tiling")
    else:

        logging.info(f"{time_now()}: Loading tile grid")

        with open(tiled_dir / "tile_grid.pickle", 'rb') as f:
            grid = pickle.load(f)

        logging.info(f"{time_now()}: Finished loading tile grid")

    return grid


def do_tile_coordinates(grid, tile_nr, out_dir):
    tile_df = grid.getTileDataCoordinates(tile_nr, rowname="row", colname="col")
    tile_df.to_parquet(out_dir / f"detected_transcripts_transformed_tile{tile_nr}.parquet", index=False)

def tile_detected_transcripts(grid):
    if not detected_transcripts_tiled_out_dir.is_dir():
        logging.info(f"{time_now()}: Loading parsed detected transcripts")

        parsed_detected_transcripts = pd.read_parquet(out_dir / "detected_transcripts_parsed.parquet")

        logging.info(f"{time_now()}: Tiling detected transcripts")

        grid.addDataCoordinates(parsed_detected_transcripts)
        detected_transcripts_tiled_out_dir.mkdir(exist_ok=True)

        for tile_nr in range(1,grid.n_tiles+1):
           do_tile_coordinates(grid, tile_nr, detected_transcripts_tiled_out_dir)

        logging.info(f"{time_now()}: Finished tiling detected transcripts")
        return grid
    else:
        logging.info(f"{time_now()}: Skipped: detected transcripts already tiled")
        return grid

# Segment the dapi-cellboundary stainings
def do_segment_everything(tile_nr, grid):
    dapi_tile_path = Path(dapi_tiled_dir  / f"{dapi_prefix}_tile{tile_nr}.tif")
    cellbound2_tile_path = Path(cellbound2_tiled_dir / f"{cellbound2_prefix}_tile{tile_nr}.tif")
    cellbound3_tile_path = Path(cellbound3_tiled_dir / f"{cellbound3_prefix}_tile{tile_nr}.tif")

    labeled_cellbound_out_path = labeled_cellbound_tiled_dir / f"{cellbound2_prefix}_labeled_tile{tile_nr}.tif"
    labeled_dapi_out_path = labeled_dapi_tiled_dir / f"{dapi_prefix}_labeled_tile{tile_nr}.tif"


    dapi_tile = io.imread(dapi_tile_path)
    if np.amax(dapi_tile) == 0:  # sometimes the corner tiles of a staining are just empty, no use in segmenting them
        return

    cellbound2_tile = io.imread(cellbound2_tile_path)

    if cellbound3_tile_path.is_file():
        cellbound3_tile = io.imread(cellbound3_tile_path) 
    else:
        cellbound3_tile = np.zeros(cellbound2_tile.shape)
        logging.info(f"{time_now()}: No cellbound3 found ({cellbound3_tile_path}), only using 2 for segmentation")


    if not labeled_cellbound_out_path.is_file(): # Segmentation often gets interupted so we add this to not redo stuff
        labeled_cellbound = segment_dual_channel(cellbound2_tile + cellbound3_tile, dapi_tile, diameter=100, gpu=True, device=device)

        if EXPAND_SEGMENTATION > 0:
            labeled_cellbound = expand_labels(labeled_cellbound, EXPAND_SEGMENTATION)

        io.imsave(labeled_cellbound_out_path, labeled_cellbound, check_contrast=False)
    else:
        labeled_cellbound = io.imread(labeled_cellbound_out_path)

    if not labeled_dapi_out_path.is_file():
        labeled_dapi = segment_dapi(dapi_tile, diameter=100, gpu=True, device=device)

        if EXPAND_SEGMENTATION > 0:
            labeled_dapi = expand_labels(labeled_dapi, EXPAND_SEGMENTATION)

        io.imsave(labeled_dapi_out_path, labeled_dapi, check_contrast=False)
    else:
        labeled_dapi = io.imread(labeled_dapi_out_path)

    if not (measured_df_dir / f"{dapi_prefix}_{segmented_prefix}_measured_tile{tile_nr}.csv").is_file():
        measured = combine_cell_and_dapi_info(labeled_cellbound, labeled_dapi, grid=grid, tile_nr=tile_nr, keep_unmatched_dapis = KEEP_UNMATCHED_NUCLEI)
        if not measured.empty:
            measured.to_csv(measured_df_dir / f"{dapi_prefix}_{segmented_prefix}_measured_tile{tile_nr}.csv", index=False)


def segment_images(grid):
    logging.info(f"{time_now()}: Segmenting images")

    labeled_dapi_tiled_dir.mkdir(exist_ok=True)
    labeled_cellbound_tiled_dir.mkdir(exist_ok=True)
    measured_df_dir.mkdir(exist_ok=True)

    # This used to be parallized but cuda really does not enjoy that and it's not worth the headache
    for tile_nr in tqdm(range(1, grid.n_tiles+1)):
        do_segment_everything(tile_nr, grid)
        torch.cuda.empty_cache()

    logging.info(f"{time_now()}: Finished segmenting images")

    return grid

def do_make_count_matrix(tile_nr):
    # Some tiles won't contain any segmentation, so no need to run those
    if not (measured_df_dir / f"{dapi_prefix}_{segmented_prefix}_measured_tile{tile_nr}.csv").is_file():
        return
    measured_df =  pd.read_csv(measured_df_dir / f"{dapi_prefix}_{segmented_prefix}_measured_tile{tile_nr}.csv")
    labeled = io.imread(labeled_cellbound_tiled_dir / f"{cellbound2_prefix}_labeled_tile{tile_nr}.tif")
    df = pd.read_parquet(detected_transcripts_tiled_out_dir / f"detected_transcripts_transformed_tile{tile_nr}.parquet")
    count_matrix_df, adapted_decoded_df = create_gene_matrix(labeled, df, rowname = "local_row", colname="local_col", measured_df = measured_df, key_to_match_in_labeled_image = "corresponding_cell")

    # We add the cell every transcripts was assigned to to later make the subcell expression list
    adapted_decoded_df["cell_label_assigned"] = adapted_decoded_df["cell_label_assigned"] 

    # Add tile number to the cell labels to make them unique
    count_matrix_df.index = f"tile{tile_nr}_" + count_matrix_df.index

    count_matrix_df.to_csv(count_matrix_out_dir / f"count_matrix_tile{tile_nr}.csv") 
    adapted_decoded_df.to_parquet(detected_transcripts_tiled_out_dir / f"detected_transcripts_transformed_tile{tile_nr}.parquet", index=False)

def make_count_matrix(grid):

    logging.info(f"{time_now()}: Making count matrices")

    count_matrix_out_dir.mkdir(exist_ok=True)
    Parallel(n_jobs=48, backend="multiprocessing")(delayed(do_make_count_matrix)(tile_nr) for tile_nr in range(1, grid.n_tiles + 1))

    logging.info(f"{time_now()}: Finished making count matrices")

    # We make all count matrices seperarate per tile, but after this we want them to be concatenated
    logging.info(f"{time_now()}: Concatenating count matrices")

    concat_df = concat_count_matrices(str(count_matrix_out_dir / "*.csv"))
    concat_df.to_csv(count_matrix_out_dir / "concat_count_matrix.csv")

    logging.info(f"{time_now()}: Done concatenating count matrices")

    return grid


def cut_single_cells(tmp):
    grid  = tmp

    if not single_cells_out_dir.is_dir():
        logging.info(f"{time_now()}: Splitting segmentation into single cells")

        single_cells_out_dir.mkdir(exist_ok=True)
        Parallel(n_jobs=12, backend="multiprocessing")(delayed(do_extract_single_cells)(tile_nr) for tile_nr in range(1, grid.n_tiles + 1))

        logging.info(f"{time_now()}: Finished splitting segmentation into single cells")
        grid = tmp
        return tmp
    else:
        logging.info(f"{time_now()}: Skipped: splitting segmentation into single cells")
        grid = tmp
        return tmp

def do_extract_single_cells(tile_nr):
     # First check if this image was segmented, cause if it was empty it was just skipped and the tile won't exist
    if not (labeled_dapi_tiled_dir / f"{dapi_prefix}_labeled_tile{tile_nr}.tif").is_file():
        return
    # It's also possible it was segmented but no match was found, so the measured df might be non-existand as well
    if not (measured_df_dir / f"{dapi_prefix}_{segmented_prefix}_measured_tile{tile_nr}.csv").is_file():
        return
    single_cells_tile_out_dir = single_cells_out_dir / f"tile{tile_nr}"
    single_cells_tile_out_dir.mkdir(exist_ok=True)
    
    dapi_labeled_tile= io.imread(labeled_dapi_tiled_dir / f"{dapi_prefix}_labeled_tile{tile_nr}.tif")
    cellbound_labeled_tile = io.imread(labeled_cellbound_tiled_dir / f"{cellbound2_prefix}_labeled_tile{tile_nr}.tif")
    measured_tile_df = pd.read_csv(measured_df_dir / f"{dapi_prefix}_{segmented_prefix}_measured_tile{tile_nr}.csv")
    detected_transcripts_tile_df = pd.read_parquet(detected_transcripts_tiled_out_dir / f"detected_transcripts_transformed_tile{tile_nr}.parquet")

    # GEImage_tile = io.imread(gene_labeled_dir / f"gene_labeled_image_tile{tile_nr}.tif")

    dapi_tile =  io.imread(dapi_tiled_dir / f"{dapi_prefix}_tile{tile_nr}.tif")
    cellbound2_tile =  io.imread(cellbound2_tiled_dir / f"{cellbound2_prefix}_tile{tile_nr}.tif")
    try:
        cellbound3_tile =  io.imread(cellbound3_tiled_dir / f"{cellbound3_prefix}_tile{tile_nr}.tif")
    except:
        cellbound3_tile = np.zeros(cellbound2_tile.shape)

    extra_tiles_to_extract = []
    if extra_channels_tiled_dirs:
        for extra_channel_tiled_dir, extra_channel_prefix in zip(extra_channels_tiled_dirs, extra_channels_prefixes):
            extra_tiles_to_extract.append((extra_channel_prefix, io.imread(extra_channel_tiled_dir / f"{extra_channel_prefix}_tile{tile_nr}.tif")))


    cellbound_tile = cellbound2_tile + cellbound3_tile

    viz_image = np.zeros((*dapi_tile.shape, 3))
    viz_image[:,:,0] = dapi_tile / np.amax(dapi_tile)
    viz_image[:,:,1] = match_histograms(cellbound_tile / np.amax(cellbound_tile), dapi_tile / np.amax(dapi_tile))

    # cells at the border of the tile are not taken into account
    border = tileBorder(cellbound_labeled_tile, tile_nr = tile_nr)
    
    image_prefix = f"tile{tile_nr}_{segmented_prefix}_"
    # expand = 0
    detected_transcripts_with_subcell_location = cut_bbox_cells_based_on_dual_channel_segmentation(measured_tile_df, 
                                                   dapi_labeled_image = dapi_labeled_tile, 
                                                   cell_labeled_image = cellbound_labeled_tile, 
                                                   detected_transcripts = detected_transcripts_tile_df,    
                                                   # GEimage = GEImage_tile, 
                                                   origin_image = viz_image, 
                                                   border = border, 
                                                   image_prefix = image_prefix,
                                                   expand=0, 
                                                   out_dir = single_cells_tile_out_dir,
                                                   extra_channels = extra_tiles_to_extract)

# Create an initial adata object with just the count matrix and cell metadata
def get_adata(grid):
    logging.info(f"{time_now()}: Getting Adata object")

    if not (out_dir / "adata.h5ad").is_file():

        # If extra channels were counted, the count matrices are in a different directory
        if extra_channels_to_extract:
            count_matrix = pd.read_csv(count_matrix_with_spots_counted_out_dir / "concat_count_matrix.csv", index_col=0)
        else:
            count_matrix = pd.read_csv(count_matrix_out_dir / "concat_count_matrix.csv", index_col=0)

        adata = ad.AnnData(count_matrix)
        adata.obs_names = list(count_matrix.index)
        adata.var_names = list(count_matrix.columns)
        adata.obs["name"] = list(adata.obs_names)
        adata.obs["tile_nr"] = [int(re.findall('tile(\d+)', label)[0]) for label in adata.obs["name"]] 
        adata.obs["local_row"] = [int(re.findall('row(\d+)', label)[0]) for label in adata.obs["name"]] 
        adata.obs["local_col"] = [int(re.findall('col(\d+)', label)[0]) for label in adata.obs["name"]] 

        adata.obs["cell_id"] = list(adata.obs_names)
        adata.var["gene_name"] = list(adata.var_names)

        with open(out_dir / "gene_map.json", 'r') as f:
            gene_map = json.load(f)

        # Now we want to add gene labels from the map to adata, but the newly counted channels (which are genes that are not present in the molecular data) don't have numbers yet, so we fix that first
        max_gene_label = max(gene_map.values())
        old_max_gene_label = max_gene_label # make duplicate since we'll be checking whether it changed later

        gene_labels = []
        for gene_name in list(adata.var_names):
            if gene_name in gene_map:
                gene_label = gene_map[gene_name]
                gene_labels.append(gene_label)
            else:
                raise KeyError(f"{gene_name} not in gene map")

        adata.var["gene_label"] =  gene_labels

        adata.uns["detected_transcripts"] = {}
        for tile_nr in adata.obs["tile_nr"].unique():
            df = pd.read_parquet(detected_transcripts_tiled_out_dir  / f"detected_transcripts_transformed_tile{tile_nr}.parquet")
            df.index = df.index.map(str)
            detected_transcripts = df.to_dict(orient="list")

            adata.uns["detected_transcripts"][str(tile_nr)] = ak.Array(detected_transcripts)

        adata.uns["measured_df"] = {}
        for tile_nr in adata.obs["tile_nr"].unique():
            adata.uns["measured_df"][str(tile_nr)] = pd.read_csv(measured_df_dir / f"{dapi_prefix}_{segmented_prefix}_measured_tile{tile_nr}.csv")

        adata = add_measured_dfs_as_obs_metadata(adata)

        adata.write_h5ad(out_dir / "adata.h5ad", compression="gzip")

        tmp = (grid, adata)
        return tmp
    else:
        adata = ad.read_h5ad(out_dir / "adata.h5ad")
        tmp = (grid, adata)
        return tmp

def do_coordinate_parsing(tile_nr, model):
    # same here, first check if this tile was actually processed:
    if not (single_cells_out_dir / f"tile{tile_nr}").is_dir():
        return

    with open(out_dir / "gene_map.json", 'r') as f:
        gene_map = json.load(f)

    coordinate_parsed_tile_out_dir = coordinate_parsed_out_dir / f"tile{tile_nr}"
    coordinate_parsed_tile_out_dir.mkdir(exist_ok=True)
    
    all_detected_transcripts_paths = glob(str(single_cells_out_dir / f"tile{tile_nr}" / "detected_transcripts" / "*.parquet"))
    
    if not all_detected_transcripts_paths:
        logging.info(f"{time_now()}: Tile{tile_nr} skipped, empty directory")
        return
    
    dapi_dir = single_cells_out_dir / f"tile{tile_nr}" / "single_dapi_masks"
    cell_mask_dir = single_cells_out_dir / f"tile{tile_nr}" / "single_cell_masks"

    target_shape = (100,100)
    sigma = 1.5

    corresponding_dapis = match_genes_with_dapis(all_detected_transcripts_paths, dapi_dir, dapi_prefix = "cellID")
    corresponding_cell_masks = match_dapis_with_cell(cell_mask_dir, corresponding_dapis, replace_dapi_to_cell={"dapi-mask": "cell-mask"})

    with torch.no_grad():
        model["cvae"].eval()
        process_without_zooming_out(all_detected_transcripts_paths, coordinate_parsed_tile_out_dir, target_shape, sigma, gene_map, corresponding_dapis = corresponding_dapis, corresponding_cell_masks = corresponding_cell_masks, model = model, device = device)


def process_to_input_cvae(tmp):
    grid, adata = tmp
    # Now we have single gene images, we coordinate parse to input for the model 
    if not coordinate_parsed_out_dir.is_dir(): 
        logging.info(f"{time_now()}: Preparing single-gene images for input to model")
        coordinate_parsed_out_dir.mkdir(exist_ok=True)

        model = torch.load(model_path)
        model["cvae"].to(device)
        # Parallel(n_jobs=24, backend="multiprocessing")(delayed(doCoordinateParsing)(tile_nr, model) for tile_nr in range(1, grid.n_tiles + 1))
        for tile_nr in range(1, grid.n_tiles + 1):
            do_coordinate_parsing(tile_nr, model)
        # Parallel(n_jobs=12, backend="multiprocessing")(delayed(doCoordinateParsing)(tile_nr) for tile_nr in range(1, grid.n_tiles + 1))
        logging.info(f"{time_now()}: Finished preparing single-gene images for input to model")
        tmp = (grid, adata)
        return tmp
    else:
        logging.info(f"{time_now()}: Skipped: preparing single-gene images for input to model")
        tmp = (grid, adata)
        return tmp


from src.add_to_adata import add_single_cell_info_to_adata, add_subcell_representation_to_adata
def finalize_anndata_object(tmp):
    grid, adata = tmp
    logging.info(f"{time_now()}: Adding single cell info to Anndata")

    adata = add_single_cell_info_to_adata(adata, coordinate_parsed_out_dir, segmented_prefix)
    # adata.write_h5ad(out_dir / "adata_finished.h5ad", compression="gzip")

    logging.info(f"{time_now()}: Finished adding single cell info to Anndata")

    logging.info(f"{time_now()}: Adding subcell info to Anndata")

    adata = add_subcell_representation_to_adata(adata, coordinate_parsed_out_dir, segmented_prefix)
    adata.write_h5ad(out_dir / "adata_subcell.h5ad", compression="gzip")

    logging.info(f"{time_now()}: Finished adding single cell info to Anndata")

    tmp = (grid, adata)
    return tmp

# Before we do anything, we check what the last step that was done is, so we only have to do those that are necessary
## This in case intermediate data is removed due to storage concerns, something I wanna implement to be automatic but not yet implemented

order_bool_list = [
    (out_dir / "detected_transcripts_parsed.parquet").is_file(),
    (out_dir / "gene_map.json").is_file(),
    (tiled_dir / "tile_grid.pickle").is_file(),
    detected_transcripts_tiled_out_dir.is_dir(),
    measured_df_dir.is_dir(),
    count_matrix_out_dir.is_dir(),
    single_cells_out_dir.is_dir(),
    count_matrix_with_spots_counted_out_dir.is_dir(),
    (out_dir / "adata.h5ad").is_file(),
    coordinate_parsed_out_dir.is_dir(),
    (out_dir / "adata_subcell.h5ad").is_file(),
]

function_list = [
    get_detected_transcripts, # returns detected transcripts
    get_gene_map, #
    get_tile_grid, # return grid
    tile_detected_transcripts, # for this we need to load detected transcripts
    segment_images,
    make_count_matrix,
    cut_single_cells,
    get_adata,
    process_to_input_cvae,
    finalize_anndata_object,
]

# Check which functions still need to get done, and then we take the last one
res = list(filter(lambda i: order_bool_list[i], range(len(order_bool_list))))
max_index = max(res) if res else 0
to_do_index = max_index + 1 if max_index > 0 else 0
to_do_functions = function_list[to_do_index:]

print(to_do_functions) # Diagnostics

# If we start after the tile grid is made, we load it
if to_do_index > 2:
    grid = get_tile_grid(None) # the tile grid we need anyway
    tmp_res = grid
else:
    tmp_res = None

# If we start after the adata is made, we load it
if to_do_index > 7:
    tmp_res = get_adata(grid) # adata we need anyway


for function in to_do_functions:
    tmp_res = function(tmp_res)

logging.info(f"{time_now()}: All done!")
