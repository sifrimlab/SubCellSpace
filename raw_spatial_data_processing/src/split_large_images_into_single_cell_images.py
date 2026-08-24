from glob import glob
from pathlib import Path
from pyxelperfect.tiling import tileBorder
from pyxelperfect.utils import getPatch
from skimage import io
from skimage.draw import rectangle_perimeter
from tqdm import tqdm
from typing import List, Tuple
import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

def cutBboxesFromCenter(measured_df: pd.DataFrame, dapi_image: np.ndarray, labeled_image, GEimage, border, image_prefix: str = "", expand=10):
    if image_prefix and not image_prefix.endswith("_"):
        image_prefix += "_"
                                                                                                                                                                   
    for i,row in enumerate(measured_df.itertuples()):                                                                                                                       
        if (border is not None) and (row.image_label in border.border_labels):
            print(row.image_label)                                                                                                                                 
            continue
                                                                                                                                                                   
        X = row.center_col
        Y = row.center_row                                                                                                                                           
        bb = getPatch(dapi_image, (Y,X), radius = (expand,expand), fill=0)                                                                                                      
        io.imsave(f"{image_prefix}dapi_cellID{row.image_label}.tif", bb)

        labeled_bb = getPatch(labeled_image, (Y,X), radius = (expand,expand), fill=0)
        bb[labeled_bb != row.image_label] = 0 
        io.imsave(f"{image_prefix}cell-mask_cellID{row.image_label}.tif", bb)

        gene_labeled_bb = getPatch(GEimage, (Y,X), radius = (expand,expand), fill=0)
        gene_labeled_bb[labeled_bb != row.image_label] = 0
        io.imsave(f"{image_prefix}GElabeled_cellID{row.image_label}.tif", gene_labeled_bb)

def cutBboxCellsBasedOnDualChannelSegmentation(measured_df: pd.DataFrame, dapi_labeled_image: np.ndarray, cell_labeled_image, detected_transcripts, border, origin_image = None, image_prefix: str = "", expand=10, out_dir = Path("./"), extra_channels = [], write_files=True, return_cells=False):
    '''
    It's important here that measured_df was created by the dual_segment function, because that output already deletes nuclei that don't have corresponding cellbounds

    write_files  -- write the per-cell tif/parquet layout (single_*_masks/,
                    detected_transcripts/, actual_images/, extra channels). Set
                    False to skip it (e.g. when only the compact archive is wanted).
    return_cells -- also collect and return one dict per kept cell:
                    {"cell_id", "dapi_mask", "cell_mask", "transcripts", "extra"}.
    '''
    cell_mask_out_dir = out_dir / "single_cell_masks/"
    dapi_mask_out_dir = out_dir / "single_dapi_masks/"
    single_cell_detected_transcripts_out_dir =  out_dir / "detected_transcripts"
    if write_files:
        cell_mask_out_dir.mkdir(parents=True, exist_ok=True)
        dapi_mask_out_dir.mkdir(parents=True, exist_ok=True)
        single_cell_detected_transcripts_out_dir.mkdir(parents=True, exist_ok=True)
        if origin_image is not None:
            origin_image_out_dir =  out_dir / "actual_images/"
            origin_image_out_dir.mkdir(parents=True, exist_ok=True)

    collected = [] if return_cells else None

    if image_prefix and not image_prefix.endswith("_"):
        image_prefix += "_"

    left_out_because_of_border = 0
    for i, row in enumerate(measured_df.itertuples()):

        if (border is not None) and (row.corresponding_cell in border.border_labels):
            left_out_because_of_border += 1
            continue

        dapi_bbox = ast.literal_eval(row.bbox)
        dapi_bbox_slc = np.s_[dapi_bbox[0]: dapi_bbox[2], dapi_bbox[1]:dapi_bbox[3]]

        cellbound_bbox = ast.literal_eval(row.cell_bbox)
        cellbound_bbox_slc = np.s_[cellbound_bbox[0]: cellbound_bbox[2], cellbound_bbox[1]:cellbound_bbox[3]]

        # cellbound_bbox might be an empty tuple if we decide to keep the unmatched dapis, in that case it will be falsy
        if cellbound_bbox:
            if row.Area > 50: # Super tiny Areas will be wrong segmentation
                cell_labeled_bb = cell_labeled_image[cellbound_bbox_slc].copy()
                cell_labeled_bb[cell_labeled_bb != row.corresponding_cell] = 0 # We want bbox the size of the cell, but only with cell as label

                dapi_labeled_bb =  dapi_labeled_image[cellbound_bbox_slc].copy()
                dapi_labeled_bb[dapi_labeled_bb != row.image_label] = 0 # We want bbox the size of the cell, but only with dapi as label

                # Adapt the coordinates of the detected transcripts to be subcell
                detected_transcripts_for_this_cell = detected_transcripts[detected_transcripts["cell_label_assigned"] == row.corresponding_cell].copy()
                detected_transcripts_for_this_cell["subcell_row"] = detected_transcripts_for_this_cell["local_row"].apply(lambda x: round(x) - cellbound_bbox[0])
                detected_transcripts_for_this_cell["subcell_col"] = detected_transcripts_for_this_cell["local_col"].apply(lambda x: round(x) - cellbound_bbox[1])

                extra_bbs = {}
                if extra_channels:
                    for extra_channel_name, extra_channel_img in extra_channels:
                        extra_bbs[extra_channel_name] = extra_channel_img[cellbound_bbox_slc].copy()

                if return_cells:
                    collected.append({
                        "cell_id": row.corresponding_cell,
                        "dapi_mask": dapi_labeled_bb,
                        "cell_mask": cell_labeled_bb,
                        "transcripts": detected_transcripts_for_this_cell,
                        "extra": extra_bbs,
                    })

                if write_files:
                    try:
                        io.imsave(cell_mask_out_dir / f"{image_prefix}cell-mask_cellID{row.corresponding_cell}.tif", cell_labeled_bb, check_contrast=False)
                    except ValueError:
                        logging.info(f"{datetime.now().strftime('%H:%M:%S')}: failed, Area: {row.Area}")
                    io.imsave(dapi_mask_out_dir / f"{image_prefix}dapi-mask_cellID{row.corresponding_cell}.tif", dapi_labeled_bb, check_contrast=False)

                    if origin_image is not None:
                        if len(origin_image.shape) == 2:
                            origin_image_bbox =  origin_image[cellbound_bbox_slc].copy()
                        elif len(origin_image.shape) == 3:
                            origin_image_bbox =  origin_image[cellbound_bbox[0]: cellbound_bbox[2], cellbound_bbox[1]:cellbound_bbox[3], :].copy()
                        io.imsave( origin_image_out_dir / f"{image_prefix}actualimage_cellID{row.corresponding_cell}.tif", origin_image_bbox, check_contrast=False)
                    for extra_channel_name, extra_channel_bb in extra_bbs.items():
                        extra_channel_out_dir =  out_dir / extra_channel_name
                        extra_channel_out_dir.mkdir(exist_ok=True)
                        io.imsave(extra_channel_out_dir / f"{image_prefix}{extra_channel_name}_cellID{row.corresponding_cell}.tif", extra_channel_bb, check_contrast=False)
                    detected_transcripts_for_this_cell.to_parquet(single_cell_detected_transcripts_out_dir / f"{image_prefix}GElabeled_cellID{row.corresponding_cell}.parquet", index=False)


        elif write_files: # If we just have dapi (kept unmatched nuclei); diagnostic layout only
            if row.Area > 30:
                dapi_labeled_bb =  dapi_labeled_image[dapi_bbox_slc].copy()
                dapi_labeled_bb[dapi_labeled_bb != row.image_label] = 0 # Now we take just the bbox of the dapi, cause we don't know anything about the cell.
                io.imsave(dapi_mask_out_dir / f"{image_prefix}dapi-mask_cellID{row.corresponding_cell}.tif", dapi_labeled_bb, check_contrast=False)


                # NOTE: this branch used to also write a gene-labeled tif from a
                # `GEimage` argument that this function never took, into an out_dir that
                # was never defined -- two NameErrors that made the branch unreachable.
                # Transcripts are stored as parquet by the matched branch instead, so the
                # write is simply dropped rather than reconstructed.

                if origin_image is not None:
                    if len(origin_image.shape) == 2:
                        origin_image_bbox =  origin_image[dapi_bbox_slc].copy()
                    elif len(origin_image.shape) == 3:
                        origin_image_bbox =  origin_image[dapi_bbox[0]: dapi_bbox[2], dapi_bbox[1]:dapi_bbox[3], :].copy()
                    io.imsave( origin_image_out_dir / f"{image_prefix}actualimage_cellID{row.corresponding_cell}.tif", origin_image_bbox, check_contrast=False)
                if extra_channels:
                    for extra_channel_name, extra_channel_img in extra_channels:
                        extra_channel_out_dir =  out_dir / extra_channel_name
                        extra_channel_out_dir.mkdir(exist_ok=True)

                        extra_channel_bb = extra_channel_img[dapi_bbox_slc].copy()
                        # extra_channel_bb[cell_labeled_bb != row.corresponding_cell] = 0 # set all genes that fall outside of the cell boundary to zero
                        io.imsave(extra_channel_out_dir / f"{image_prefix}{extra_channel_name}_cellID{row.corresponding_cell}.tif", extra_channel_bb, check_contrast=False)
    logger.info(f"Number of cells on the border left out: {left_out_because_of_border}")

    if return_cells:
        return collected






if __name__ == '__main__':
    from skimage.exposure import match_histograms
    for i in tqdm(range(1,97)):
        if i == 49:
            continue
        dapi = io.imread(f"/mnt/data/MERFISH/colonCancer1/tiled/DAPI/mosaic_DAPI_z3_tile{i}.tif")
        cellbound = io.imread(f"/mnt/data/MERFISH/colonCancer1/tiled/cellbound2/mosaic_Cellbound2_z3_tile{i}.tif") + io.imread(f"/mnt/data/MERFISH/colonCancer1/tiled/cellbound3/mosaic_Cellbound3_z3_tile{i}.tif") 
        viz_image = np.zeros((*dapi.shape, 3))
        viz_image[:,:,0] = dapi / np.amax(dapi)
        viz_image[:,:,1] = match_histograms(cellbound / np.amax(cellbound), dapi / np.amax(dapi))

        measured_df = pd.read_csv(f"/mnt/data/MERFISH/colonCancer1/dual_segmentation/measuredDfs/tile{i}_dapi-cellbound2-cellbound3_measured.csv")
        cellbound_labeled = io.imread(f"/mnt/data/MERFISH/colonCancer1/dual_segmentation/labeled_images/tile{i}_dapi-cellbound2-cellbound3_labeled.tif")
        image_prefix = f"tile{i}_dapi-cellbound2-cellbound3_"
        dapi_labeled = io.imread(f"/mnt/data/MERFISH/colonCancer1/dual_segmentation/labeled_images/tile{i}_dapi_labeled.tif")
        GEimage = io.imread(f"/mnt/data/MERFISH/colonCancer1/tiled/GE_labeled/mosaic_DAPI_z3_GElabeled_tile{i}.tif")
        out_dir = Path("/mnt/data/MERFISH/colonCancer1/dual_segmentation/single_cells/")
        out_dir.mkdir(exist_ok=True)

        expand = 0

        border = tileBorder(cellbound_labeled, tile_nr = i)
        cutBboxCellsBasedOnDualChannelSegmentation(measured_df, dapi_labeled_image = dapi_labeled, cell_labeled_image = cellbound_labeled, GEimage = GEimage, origin_image = viz_image, border = border, image_prefix = image_prefix, expand=expand, out_dir = out_dir)
