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
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.draw import disk, rectangle
from skimage.transform import rotate, resize, warp
import SimpleITK as sitk

"""
This module assumes 'point': len(tuple) == 2 with row and col being the fir and second element respectively.
"""


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
    """convert_points_to_df.
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

def sitk_transform_point(point, sitk_transform):
    """sitk_transform_point.
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
        translation, shape_after_rotation = calculate_rotate_translation(target_shape, rotation_angle)
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

def process_with_zooming_out(image_paths, out_dir, target_shape, sigma, number_of_rotations, corresponding_dapis=None, corresponding_cell_masks = None, first_resize_step = False, save_new_coords=False):
    out_dir.mkdir(exist_ok=True)
    if corresponding_dapis is not None:
        # dapi_output_dir = out_dir.parent / "all_rotated_dapi_zoomed_again"
        dapi_output_dir = out_dir / "dapi_zoomed"
        dapi_output_dir.mkdir(exist_ok=True)

    if corresponding_cell_masks is not None:
        cell_mask_output_dir = out_dir / "cell_mask_zoomed"
        cell_mask_output_dir.mkdir(exist_ok=True)

    if save_new_coords:
        coords_output_dir =  out_dir / "gene_coords_zoomed"
        coords_output_dir.mkdir(exist_ok=True)

    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        img = io.imread(img_path)

        if corresponding_dapis is not None:
            dapi = io.imread(corresponding_dapis[i])
        if corresponding_cell_masks is not None:
            cell_mask = io.imread(corresponding_cell_masks[i])
        if number_of_rotations > 0:
            for j in range(number_of_rotations):
                rotation_angle = random.randint(1,360)

                kwargs, shape_after_rotation = calc_kwargs(img.shape, target_shape, rotation_angle, first_resize_step=first_resize_step)
                # don't do final resize step but we're first gonna cut out to fix the zooming problem
                kwargs = kwargs[:-1]

                coords = convert_image_to_coords(img)

                new_coords = []
                for coord in coords:
                    new_coords.append(point2int(register_point(coord, kwargs)))
                new_coords = [coord for coord in new_coords if all(ele >= 1 for ele in coord)]
                            
                new_img = coords_to_img(new_coords, shape_after_rotation)
                
                new_coords, bbox = cut_out_coords_after_rotation(new_coords, target_shape)
                shape_after_bbox = bbox[2] - bbox[0], bbox[3] - bbox[1]
                zoom_factors_after_bbox_cutting = calculate_resize_zoom_factors(shape_after_bbox, target_shape)

                final_coords = []
                for new_coord in new_coords:
                    final_coord = resize_point_with_transform(new_coord, zoom_factors_after_bbox_cutting)
                    final_coords.append(final_coord)

                if save_new_coords:
                    rotated_coords_df = pd.DataFrame(final_coords, columns=['row', 'col'])
                    rotated_coords_df.to_csv(coords_output_dir / f"{img_path.stem}_rotated{rotation_angle}_blurred{sigma:1}_gene-coords.csv", index=False)
                new_img = coords_to_img(final_coords, target_shape)
                
                
                blurred = gaussian(new_img, sigma)
                io.imsave(out_dir / f"{img_path.stem}_rotated{rotation_angle}_blurred{sigma:1.1f}.tif", blurred, check_contrast=False)

                if corresponding_dapis is not None:
                    dapi_output_file = dapi_output_dir / f"{img_path.stem}_rotated{rotation_angle}_blurred{sigma:1.1f}_DAPI.tif"
                    ## transformation of dapi
                    if first_resize_step:
                        dapi = resize(dapi, target_shape)
                    rotated, shape_after_rotation = rotate(dapi, rotation_angle, resize=True)
                    bbox_cut = cut_bbox_from_image(bbox, rotated)
                    resized= resize(bbox_cut, target_shape)
                    io.imsave(dapi_output_file, resized, check_contrast=False)
                if corresponding_cell_masks is not None:
                    cell_mask_output_file = cell_mask_output_dir / f"{img_path.stem}_rotated{rotation_angle}_blurred{sigma:1.1f}_cell-mask.tif"
                    ## transformation of cell_mask
                    if first_resize_step:
                        cell_mask = resize(cell_mask, target_shape)
                    rotated, shape_after_rotation = rotate(cell_mask, rotation_angle, resize=True)
                    bbox_cut = cut_bbox_from_image(bbox, rotated)
                    resized = resize(bbox_cut, target_shape)
                    io.imsave(cell_mask_output_file, resized, check_contrast=False)

        else:
            kwargs = calc_kwargs(img.shape, target_shape, None, first_resize_step=first_resize_step)
            coords = convert_image_to_coords(img)

            new_coords = []
            for coord in coords:
                new_coords.append(point2int(register_point(coord, kwargs)))
            new_coords = [coord for coord in new_coords if all(ele >= 1 for ele in coord)]

            new_img = coords_to_img(new_coords, target_shape)
            blurred = gaussian(new_img, sigma)
            io.imsave(out_dir / f"{img_path.stem}_rotated1_blurred{sigma:0.1f}.tif", blurred, check_contrast=False)

            if corresponding_dapis is not None:
                dapi_output_file = dapi_output_dir / f"{img_path.stem}_rotated1_blurred{sigma:0.1f}_DAPI.tif"
                ## transformation of dapi
                dapi = resize(dapi, target_shape)
                # resized, _ = resize(dapi, target_shape)
                io.imsave(dapi_output_file, dapi, check_contrast=False)
            if corresponding_cell_masks is not None:
                cell_mask_output_file = cell_mask_output_dir / f"{img_path.stem}_rotated1_blurred{sigma:0.1f}_cell-mask.tif"
                ## transformation of cell_mask
                cell_mask = resize(cell_mask, target_shape)
                # resized, _ = resize(cell_mask, target_shape)
                io.imsave(cell_mask_output_file, cell_mask, check_contrast=False)


def match_genes_with_dapis(genes_glob, dapi_dir, dapi_prefix = "icell", dapi_glob_pattern = None):
    corresponding_dapis = []
    for path in tqdm(genes_glob):
        cell_id = re.findall(f"{dapi_prefix}(\d+)", str(path))[0]
        target_dapi_path =glob(str(dapi_dir / f"*_{dapi_prefix}{cell_id}*.tif"))[0]
        corresponding_dapis.append(target_dapi_path)
    return corresponding_dapis



def match_dapis_with_cell(cell_mask_dir, corresponding_dapis, replace_dapi_to_cell={"dapi": "cell-mask"}):
    corresponding_cell_masks = []
    for dapi_path in corresponding_dapis:
        tmp_adapted_cell_path = Path(dapi_path).name
        for key, value in replace_dapi_to_cell.items():
            tmp_adapted_cell_path = tmp_adapted_cell_path.replace(key, value)

        corresponding_cell_mask_path = cell_mask_dir / tmp_adapted_cell_path
        corresponding_cell_masks.append(corresponding_cell_mask_path)
    return corresponding_cell_masks


if __name__ == '__main__':
    path_to_simulation_dir = Path("/path/to/simulation/dir/")
    all_imgs_paths = list((path_to_simulation_dir / "gene_expression").glob("*.tif"))
    dapi_dir = path_to_simulation_dir / "nuclei_filtered"
    cell_dir = path_to_simulation_dir / "cell_filtered"
    out_dir = path_to_simulation_dir / "coordinate_parsed_with_coords"
    out_dir.mkdir(exist_ok = True)

    target_shape = (100,100)
    sigma = 1.5
    number_of_rotations = 5
    corresponding_dapis = match_genes_with_dapis(all_imgs_paths, dapi_dir, dapi_prefix = "cellID")
    corresponding_cell_masks = match_dapis_with_cell(cell_dir, corresponding_dapis, replace_dapi_to_cell={"dapi": "cell"})
    print("started")

    process_with_zooming_out(all_imgs_paths, out_dir, target_shape, sigma, number_of_rotations, corresponding_dapis = corresponding_dapis, corresponding_cell_masks = corresponding_cell_masks, first_resize_step = False, save_new_coords=True)
