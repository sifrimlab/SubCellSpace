import os
import re
import ast
import math
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from skimage import io
from pathlib import Path
# import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.draw import disk, rectangle
from skimage.transform import rotate, resize, warp
from .subcellspace_utils import create_subcell_representation



def resize_point_with_transform(point, zoom_factors):
    """resizePointWithTransform.
        
    Parameters
    ----------
    point :
        2D point in form of a tuple
    zoom_factors :
        transform in the form of the zoom factors used in the ndi.zoom function by scipy
    """
    
    point = np.array([point]) # add a second artificial dimension to it
    new_point = np.array(zoom_factors) * point
    new_point = (float(new_point[0,0]), float(new_point[0,1]))
    return new_point

def calculate_resize_tform(original_shape, target_shape): 
    tform = [] 
    for i in range(len(original_shape)):
        tform.append(target_shape[i] / original_shape[i])
    tform = np.array(tform)
    return tform

def convert_image_to_coords(image):
    rr, cc = np.where(image != 0)
    coords = [(row, col) for row,col in zip(rr, cc)]
    return coords

def coords_to_img(coords, shape):
    empty = np.zeros(shape, dtype=bool)
    for coord in coords:
        coord = round(coord[0]), round(coord[1])
        try:
            empty[coord] = 1
        except:
            empty[coord[0]-1, coord[1]-1] = 1
    return empty
def get_coords_list(df, rowname = "pxl_row_in_fullres", colname="pxl_col_in_fullres"):
    rows = df[rowname]
    cols = df[colname]

    points = [(row, col) for row, col in zip(rows, cols)]
    return points

def convert_points_to_df(points, original_df, rowname = "pxl_row_in_fullres", colname="pxl_col_in_fullres"):
    """convertPointsToDf.
    Parameters
    ----------
    points :
        list of (row, col) tuples
    """
    return_df = original_df.copy()
    points = [(0,0) if v is None else v for v in points]
    rows, cols = zip(*points)
    return_df[rowname] = rows
    return_df[colname] = cols
#     original_df = original_df.drop(["pxl_row_in_fullres", "pxl_col_in_fullres"], axis=1)
    return return_df


def register_point(point, args_list):
    args_dict = {"flip": flip_point, "rotation": rotate_point_dict, "bbox": transform_points_with_bbox, "zoom_factors": resize_point_with_transform, "transform": sitk_transform_point, "translation": translate_point}
    for arg in args_list:
        for key, value in arg.items():
            # arg = dict and will only have one item
            point = args_dict[key](point, value)
            if isinstance(point, type(None)):
                return (0,0)
    return point

def find_center(x_shape):
    middle = x_shape[0] / 2, x_shape[1] / 2
    return middle

def rotate_point_dict(point, arg_dict: dict):
    # Duplicate of rotatePoint but with takes input as a dict. This just to facilitate the function dict used in registerPoint

    origin, angle= arg_dict["center"], arg_dict["angle"]
    try:
        translation = arg_dict["translation"]
    except:
        translation = (0,0)

    point = (point[0] - translation[0], point[1] - translation[1])

    if isinstance(angle, int):
        angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)

def rotate_point(point, origin, angle, translation):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    If angle is an integer, it is assumed to be degrees and will be converted to radians
    """

    if isinstance(angle, int):
        angle = math.radians(angle)
        
    point = (point[0] - translation[0], point[1] - translation[1])

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)

def cut_bbox_from_image(bbox, image: np.ndarray):
    min_row, min_col, max_row, max_col = bbox

    cut_image = image[min_row:max_row, min_col:max_col].copy()
    return cut_image

def get_shape_from_bbox(bbox):
    min_row, min_col, max_row, max_col = bbox
    return (max_row, max_col)

def point2int(point):
    # Round to closest int but also set negative values to zero
    return tuple([round(el) for el in point])
#     return tuple([max(round(el), 0) for el in point])

def transform_points_with_bbox(point, bbox, force = False):
    min_row, min_col, max_row, max_col = bbox
    if min_row <= point[0] <= max_row and min_col <= point[1] <= max_col:
        new_row = point[0] - min_row
        new_col = point[1] - min_col
        return (new_row, new_col)
    else:
        if not force:
            print(point, bbox)
            return None
        else:
            # TODO you might want to add another snippet here that returns the closes point in the new bbox image
            return None

def calculate_resize_zoom_factors(original_shape, target_shape):
    """
    Calculates the zoom factors necessary to go from original_shape to target shape
    """
    tform = [] 
    for i in range(len(original_shape)):
        tform.append(target_shape[i] / original_shape[i])
    zoom_factors = np.array(tform)
    return zoom_factors

def resize_point_with_transform(point, zoom_factors):
    """resizePointWithTransform.
    Parameters
    ----------
    point :
        2D point in form of a tuple
    zoom_factors :
        transform in the form of the zoom factors used in the ndi.zoom function by scipy
    """

    point = np.array([point]) # add a second artificial dimension to it
    new_point = np.array(zoom_factors) * point
    new_point = (float(new_point[0,0]), float(new_point[0,1]))
    return new_point


def flip_point(point, shape):
    """
    Flips a point along a specific axis. 
    shape = [n_rows, n_cols, axis],
    where if axis  == 0: no flip
    axis == 1: flip along the rows | vertically | mirror over a horizontal line 
    axis == 2: flip along the cols | horizonatlly | mirror over a vertical line 
    axis == 3: flip along both rows and cols | vertically  and horizontally | mirror over both a horizontal and vertical line 
    """
    img_shape, axis = shape[:2], shape[2]
    # img_shape = shape[:2]
    
    if axis == 0: 
        return point
    elif axis == 1:
        new_row = (img_shape[0] - point[0]) - 1
        new_col = point[1]
        new_point = (new_row, new_col)

    elif axis == 2:
        new_col = (img_shape[1] - point[1]) -1
        new_row = point[0]
        new_point = (new_row, new_col)

    elif axis == 3:
        new_row = (img_shape[0] - point[0]) - 1
        new_col = (img_shape[1] - point[1]) -1
        new_point = (new_row, new_col)
    return new_point

def sitkTransformPoint(point, sitk_transform):
    """sitkTransformPoint.
    -   sitk.transform.TransformPoint assumes (col, row) under the hood, so we invert them beforehand and invert them back after transform
    Parameters
    ----------
    point :
        tuple: (row, col)
    sitk_transform :
        sitk_transform
    """
    if isinstance(sitk_transform, str):
        sitk_transform = sitk.ReadTransform(sitk_transform)
    inverted_point = (point[1], point[0])
    transformed_point = sitk_transform.TransformPoint(inverted_point)
    return_point = (transformed_point[1], transformed_point[0])
    return return_point

def translate_point(point, offset):
    new_point =(point[0] + offset[0] , point[1] + offset[1])
    return new_point


def calc_kwargs(input_shape, target_shape, rotation_angle, first_resize_step = True):
    kwargs = []
    if first_resize_step:
        first_zoom_factors = calculate_resize_zoom_factors(input_shape, target_shape)
        kwargs.append({"zoom_factors": first_zoom_factors})
        
    if rotation_angle is not None:
        translation, shape_after_rotation = calculateRotate_translation(target_shape, rotation_angle)
        center = find_center(shape_after_rotation)
        kwargs.append({"rotation": {"angle" : rotation_angle, "center": center, "translation": translation}})

        zoom_factors_after_rotation = calculate_resize_zoom_factors(shape_after_rotation, target_shape)
        kwargs.append({"zoom_factors": zoom_factors_after_rotation})

    # If we don't rotate but the first resize step hasn't been done, we still need to rotate
    elif not first_resize_step:
        first_zoom_factors = calculate_resize_zoom_factors(input_shape, target_shape)
        kwargs.append({"zoom_factors": first_zoom_factors})
    

    if not kwargs:
        raise ValueError("kwargs is empty, something probably went wrong.")

    if "shape_after_rotation" in locals():
        return kwargs, shape_after_rotation
    else:
        return kwargs

def calculate_rotate_translation(original_shape, angle):
    zeros = np.zeros(original_shape)
    new_zeros, translation = rotate(zeros, angle, resize=True)
    return translation, new_zeros.shape

def cut_out_coords_after_rotation(coords, target_shape):
    # We can do this because rotate(resize=True) pads the image equally on each side
    coords_arr = np.array(coords)
    min_row, min_col, max_row, max_col = np.min(coords_arr[:,0]), np.min(coords_arr[:,1]),np.max(coords_arr[:,0]), np.max(coords_arr[:,1])
    row_diff = target_shape[0] - (max_row - min_row)
    col_diff = target_shape[1] - (max_col - min_col)

    if row_diff > 0:
        min_row -= row_diff // 2 
        max_row += row_diff // 2
    if col_diff > 0:
        min_col -= col_diff // 2
        max_col += col_diff // 2

    bbox = max(min_row,0), max(min_col,0), max(max_row,0), max(max_col,0)
    new_coords = []
    for coord in coords:
        new_coord = transform_points_with_bbox(coord, bbox)
        new_coords.append(new_coord)
    return new_coords, bbox

def process_without_zooming_out(detected_transcript_paths, out_dir, target_shape, sigma, gene_map, corresponding_dapis=None, corresponding_cell_masks = None, lower_thresh=10, upper_thresh=101, model=None, device = None):
    out_dir.mkdir(exist_ok=True)
    detected_transcripts_output_dir =  out_dir / "detected_transcripts_zoomed"
    detected_transcripts_output_dir.mkdir(exist_ok=True)

    if corresponding_dapis is not None:
        # dapi_output_dir = out_dir.parent / "all_rotated_dapi_zoomed_again"
        dapi_output_dir = out_dir / "dapi_zoomed"
        dapi_output_dir.mkdir(exist_ok=True)
    if corresponding_cell_masks is not None:
        cell_mask_output_dir = out_dir / "cell_mask_zoomed"
        cell_mask_output_dir.mkdir(exist_ok=True)

    for i, detected_transcript_path in enumerate(detected_transcript_paths):

        detected_transcript_path = Path(detected_transcript_path)
        detected_transcripts = pd.read_parquet(detected_transcript_path)
        if detected_transcripts.empty:
            continue

        if corresponding_dapis is not None:
            dapi_prefix = Path(corresponding_dapis[i]).stem
            dapi = io.imread(corresponding_dapis[i])

        if corresponding_cell_masks is not None:
            cell_prefix = Path(corresponding_cell_masks[i]).stem
            cell_mask = io.imread(corresponding_cell_masks[i])

        zoom_factors = calculate_resize_tform(dapi.shape, target_shape)   

        coords = [(row, col) for row, col in zip(detected_transcripts["subcell_row"], detected_transcripts["subcell_col"])]
        new_coords = []
        for coord in coords:
            new_coords.append(point2int(resize_point_with_transform(coord, zoom_factors)))
        # new_coords = [coord for coord in new_coords if all(ele >= 1 for ele in coord)]
        new_rows, new_cols = zip(*new_coords)
        detected_transcripts["subcell_row"] = new_rows
        detected_transcripts["subcell_col"] = new_cols
        detected_transcripts.to_parquet(detected_transcripts_output_dir / detected_transcript_path.name, index=False)

        if corresponding_dapis is not None:
            dapi_output_file = dapi_output_dir / f"{dapi_prefix}_DAPI.tif"
            if not dapi_output_file.is_file():
                ## transformation of dapi
                # dapi, _ = resize(dapi, target_shape)
                dapi = resize(dapi, target_shape)
                io.imsave(dapi_output_file, dapi, check_contrast=False)
            else:
                # If model is given, we're going to directly make the embedding, and for that we need the dapi image
                if model is not None:
                    dapi = io.imread(dapi_output_file)
        if corresponding_cell_masks is not None:
            cell_mask_output_file = cell_mask_output_dir / f"{cell_prefix}_cell-mask.tif"
            if not cell_mask_output_file.is_file():
                ## transformation of cell_mask
                # cell_mask, _ = resize(cell_mask, target_shape)
                cell_mask = resize(cell_mask, target_shape)
                io.imsave(cell_mask_output_file, cell_mask, check_contrast=False)

        # Now that we have cell specific files, we should actually make the input to the model per gene

        unique_genes = detected_transcripts["gene"].unique()

        for gene in unique_genes:
            gene_id = gene_map[gene]

            detected_transcripts_for_this_gene = detected_transcripts[detected_transcripts["gene"] == gene]

            occurences = len(detected_transcripts_for_this_gene)

            if (occurences > lower_thresh) and (occurences < upper_thresh):
                coords_for_this_gene =  list(zip(detected_transcripts_for_this_gene["subcell_row"],detected_transcripts_for_this_gene["subcell_col"]))

                new_img = coords_to_img(coords_for_this_gene, target_shape)
                blurred = gaussian(new_img, sigma)
                if model is None:
                    io.imsave(out_dir / f"{detected_transcript_path.stem}_genID{gene_id}genName{gene}_spots{occurences}_rotated0_blurred{sigma:0.1f}.tif", blurred, check_contrast=False)
                else:
                    # We check first that they're not empty arrays, otherwise we get zerodivision errors when we normalize
                    if (np.any(blurred)) and (np.any(dapi)):
                        try:
                            embedding = create_subcell_representation(blurred, dapi, model, device)
                        except ValueError:
                            # print(np.amax(blurred), np.amax(dapi))
                            continue

                            

                        io.imsave(out_dir / f"{detected_transcript_path.stem}_genID{gene_id}genName{gene}_spots{occurences}_rotated0_blurred{sigma:0.1f}_embedding.tif", embedding.cpu().detach().numpy(), check_contrast=False)



# Works for real data
def match_genes_with_dapis(genes_glob, dapi_dir, dapi_prefix = "cellID", dapi_glob_pattern = None):
    '''
    Dapi glob pattern is there to guide the finding of a correct first dapi. 
    For example in case you have a dapi dir, but only some of the dapis are the ones you're looking for (like a specific tile), then we can guide it
    '''
    def getDapiBase(dapi_file, dapi_prefix):
        prefix_value = re.findall('\d+',re.findall(rf"{dapi_prefix}\d+", str(Path(dapi_file).name))[0])[0]
        # cell = re.findall('\d+',re.findall("cell\d+", str(Path(dapi_file).name))[0])[0]
        # dapi_file = dapi_file.replace(f"tile{tile}", "tileNumber")
        dapi_file = dapi_file.replace(f"{dapi_prefix}{prefix_value}", f"{dapi_prefix}Number")
        return dapi_file
    
    if dapi_glob_pattern is None:
        dapi_base = Path(getDapiBase(glob(str(dapi_dir/ "*.tif"))[0], dapi_prefix = dapi_prefix)).name
    else:
        dapi_base = Path(getDapiBase(glob(str(dapi_dir/ f"{dapi_glob_pattern}*.tif"))[0], dapi_prefix = dapi_prefix)).name

    
    corresponding_dapis = []
    for path in genes_glob:
        cell_id = re.findall(rf"{dapi_prefix}(\d+)", path)[0]
        current_dapi = dapi_base.replace("Number", cell_id)
        target_dapi_path = dapi_dir / current_dapi 
        corresponding_dapis.append(target_dapi_path)
    return corresponding_dapis

def match_dapis_with_cell(cell_mask_dir, corresponding_dapis, replace_dapi_to_cell={"dapi", "cell-mask"}):
    corresponding_cell_masks = []
    for dapi_path in corresponding_dapis:
        tmp_adapted_cell_path = Path(dapi_path).name
        for key, value in replace_dapi_to_cell.items():
            tmp_adapted_cell_path = tmp_adapted_cell_path.replace(key, value)

        corresponding_cell_mask_path = cell_mask_dir / tmp_adapted_cell_path
        corresponding_cell_masks.append(corresponding_cell_mask_path)
    return corresponding_cell_masks
