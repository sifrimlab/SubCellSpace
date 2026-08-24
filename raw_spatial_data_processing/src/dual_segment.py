import re
import time
import torch
from glob import glob
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import match_histograms
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from pyxelperfect.measure import measureLabeledImage
import numpy as np
from cellpose import models
import pandas as pd
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def get_gpu_utilization(device_id):
    torch.cuda.set_device(device_id)
    return torch.cuda.utilization()
 
def get_least_busy_gpu(gpu_ids):
    gpu_info = [(gpu_id, get_gpu_utilization(gpu_id)) for gpu_id in gpu_ids]
    return min(gpu_info, key=lambda x: x[1])[0]

def segment_dual_channel(cellbound, dapi, diameter = None, also_plot = False, gpu = False, device=None, available_gpu_ids = [0,1,2,3]):
    model = models.Cellpose(gpu=gpu, device=device, model_type='cyto2')
    channels = [[2,1]]
    merged =  np.zeros((*cellbound.shape, 2))
    merged[:,:,1] = cellbound
    merged[:,:,0] = dapi
    masks, flows, styles, diams = model.eval(merged, diameter=diameter, channels=channels[0])
    
    if also_plot:
        viz_image = np.zeros((*dapi.shape, 3))
        viz_image[:,:,0] = dapi / np.amax(dapi)
        viz_image[:,:,1] = match_histograms(cellbound / np.amax(cellbound), dapi / np.amax(dapi))
        marked = mark_boundaries(viz_image, masks)
        plt.imshow(marked)
    return masks

def segment_dapi(dapi, diameter = None, also_plot = False, gpu=False, device = None, available_gpu_ids=[0,1,2,3]):
    model = models.Cellpose(gpu=gpu, device = device, model_type='nuclei')
    channels = [[0,0]]

    masks, flows, styles, diams = model.eval(dapi, diameter=diameter, channels=channels[0])
    if also_plot:
        viz_image = np.zeros((*dapi.shape, 3))
        viz_image[:,:,0] = dapi / np.amax(dapi)
        marked = mark_boundaries(viz_image, masks)
        plt.imshow(marked)
    return masks

def combine_cell_and_dapi_info(cellbound_labeled, dapi_labeled, grid, tile_nr, keep_unmatched_dapis = False):
    cellbound_measured = measureLabeledImage(cellbound_labeled)
    dapi_measured = measureLabeledImage(dapi_labeled)
    original_nr_dapis = len(dapi_measured)
    original_nr_cells = len(cellbound_measured)

    if cellbound_measured.empty:
        return cellbound_measured
    if dapi_measured.empty:
        return dapi_measured

    corresponding_cells = []
    for row in dapi_measured.itertuples():
        corresponding_cell = cellbound_labeled[row.center_row, row.center_col]
        corresponding_cells.append(corresponding_cell)

    dapi_measured["corresponding_cell"] = corresponding_cells
    if not keep_unmatched_dapis:
        dapi_measured = dapi_measured[dapi_measured["corresponding_cell"] != 0 ] # delete every dapi that doesn't have a cell attached to it
        dapi_measured = dapi_measured.drop_duplicates(subset='corresponding_cell', keep=False) # remove every dapi that has a cell assigned to it that another dapi also targets, this is most likely due to segmentation problems
        new_size_dapi_measured = original_nr_dapis - len(dapi_measured)
        new_size_cell_measured = original_nr_cells - len(dapi_measured)
        logger.info(f"Number of dapis dropped after not keeping matched: {new_size_dapi_measured} ({new_size_dapi_measured / original_nr_dapis:.02f}%)")
        logger.info(f"Number of cells dropped after not keeping matched: {new_size_cell_measured} ({new_size_cell_measured / original_nr_cells:.02f}%)")


    # Now we add the cell_bbox to dapi measured so we can extract it later
    cell_bboxes = []
    for cell in dapi_measured["corresponding_cell"]:
        if cell != 0:
            cell_bboxes.append(cellbound_measured.loc[cellbound_measured["image_label"] == cell]["bbox"].iloc[0])
        else:
            cell_bboxes.append(tuple())

    dapi_measured["cell_bbox"] = cell_bboxes
    if keep_unmatched_dapis:
        # if we keep the unmatched dapis, we need to give them a corresponding cell, otherwise later cell-naming will fail, since it's based on cell identity.
        dapi_measured.loc[dapi_measured['corresponding_cell'] == 0, 'corresponding_cell'] = dapi_measured['image_label']

    # Also rename the cell label to include the corresponding cell label
    dapi_measured["cell_label"] = dapi_measured['cell_label'] + ("_corrID_" + dapi_measured['corresponding_cell'].astype(str))

    # Now also add global coordinates of all centers, cause now we're working on a tile basis
    dapi_measured = add_real_coordinates_to_measured_df(dapi_measured, grid, tile_nr)
    return dapi_measured

def concat_measured_dfs(glob_pattern):
    files = glob(glob_pattern)
    tile_nrs = [re.findall('tile(\d+)', file)[0] for file in files]
    dfs = [pd.read_csv(file) for file in files]
    for df, tile_nr in zip(dfs, tile_nrs):
        df["tile_nr"] = tile_nr
    concat_df = pd.concat(dfs)
    # concat_df = concat_df.fillna(0)
    return concat_df

def add_real_coordinates_to_measured_df(measured_df, grid, tile_nr, rowname="center_row", colname = "center_col"):
    row_slice, col_slice =  grid.tile_boundaries[int(tile_nr)][0], grid.tile_boundaries[int(tile_nr)][1]
    row_start = row_slice.start
    col_start = col_slice.start

    measured_df["global_row"] = measured_df[rowname].apply(lambda x: x + row_start)
    measured_df["global_col"] = measured_df[colname].apply(lambda x: x + col_start)
    return measured_df


if __name__ == '__main__':
    pass
    # measured_df = pd.read("/mnt/data/tool/merscope_helmsley/vizgen_postprocessing/segmentation_output/slide1/region_0/parsed/tiled/measured_df/mosaic_DAPI_z3_dapi-cellbound2-cellbound3_measured_tile4.csv")

