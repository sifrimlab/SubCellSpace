import re
from tqdm import tqdm
from pathlib import Path
from skimage import io
import pandas as pd
from skimage.draw import disk, rectangle
from skimage.transform import rotate, resize, warp
import SimpleITK as sitk
from skimage.filters import gaussian
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import ast
import random
from glob import glob
from .subcellspace_utils import createSubcellRepresentation, embedSubcellBatch

"""
This module assumes 'point': len(tuple) == 2 with row and col being the fir and second element respectively.
All functions are structured in a way that they take only 2 arguments, point and value, where value is everything the function needs to do is thing.
This is to make sure that registerPoint can handle an arbitrary amount of different instructions, at the expense of readability.
"""


def resizePointWithTransform(point, zoom_factors):
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

def calculateResizeTform(original_shape, target_shape): 
    tform = [] 
    for i in range(len(original_shape)):
        tform.append(target_shape[i] / original_shape[i])
    tform = np.array(tform)
    return tform

def convertImageToCoords(image):
    rr, cc = np.where(image != 0)
    coords = [(row, col) for row,col in zip(rr, cc)]
    return coords

def coordsToImg(coords, shape):
    empty = np.zeros(shape, dtype=bool)
    for coord in coords:
        coord = round(coord[0]), round(coord[1])
        try:
            empty[coord] = 1
        except:
            empty[coord[0]-1, coord[1]-1] = 1
    return empty


def _coords_to_img(rows, cols, shape):
    """Vectorised coordsToImg for integer row/col arrays: a float spot map with
    1.0 at each (row, col). Out-of-range coords are clipped to the edge."""
    img = np.zeros(shape, dtype=float)
    rr = np.clip(rows, 0, shape[0] - 1)
    cc = np.clip(cols, 0, shape[1] - 1)
    img[rr, cc] = 1.0
    return img
def getCoordsList(df, rowname = "pxl_row_in_fullres", colname="pxl_col_in_fullres"):
    rows = df[rowname]
    cols = df[colname]

    points = [(row, col) for row, col in zip(rows, cols)]
    return points

def convertPointsToDf(points, original_df, rowname = "pxl_row_in_fullres", colname="pxl_col_in_fullres"):
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

def loadRegisterKwargsFromCsv(path_to_csv):
    ## Ordered processing steps will be a list of single element dicts, where key = step and value = value. registerPoint will have to be adapted with a mapping that maps step to their respective functions
    def readFlip(value):
        return ast.literal_eval(value)
    def readBbox(value):
        return ast.literal_eval(value)
    # usually when transforming a point it's for transforming a point belonging to the target coordinate axis, so we assume we want an inverse transform
    def readTransform(value):
        if "mha" in value:
            return sitk.DisplacementFieldTransform(sitk.ReadImage(value))
        else:
            return sitk.ReadTransform(value).GetInverse()
    def readRotation(value):
        rotation = ast.literal_eval(value)
        center = rotation[0]
        angle = rotation[1]
        if len(rotation) > 2:
            translation = rotation[2]
        else:
            translation = (0,0)
        return {"center": center, "angle": angle, "translation": translation}
    def readZoomFactors(value):
        return ast.literal_eval(value)

    def readTranslation(value):
        return ast.literal_eval(value)

    functions = {"flip": readFlip, "bbox": readBbox, "rotation":readRotation, "zoom_factors": readZoomFactors, "transform": readTransform, "translation": readFlip}


    df = pd.read_csv(path_to_csv)
    ordered_processing_steps = []
    for row in df.itertuples():
        # The value of the single element dict is already the parsed argument, as to not load in transforms multiple times when registering multiple points
        ordered_processing_steps.append({row.step: functions[row.step](row.value)})

    return ordered_processing_steps
        

def registerPoint(point, args_list):
    args_dict = {"flip": flipPoint, "rotation": rotatePointDict, "bbox": transformPointsWithBbox, "zoom_factors": resizePointWithTransform, "transform": sitkTransformPoint, "translation": translatePoint}
    for arg in args_list:
        for key, value in arg.items():
            # arg = dict and will only have one item
            point = args_dict[key](point, value)
            if isinstance(point, type(None)):
                return (0,0)
    return point

def findCenter(x_shape):
    middle = x_shape[0] / 2, x_shape[1] / 2
    return middle

def rotatePointDict(point, arg_dict: dict):
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

def rotatePoint(point, origin, angle, translation):
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

def cutBboxFromImage(bbox, image: np.ndarray):
    min_row, min_col, max_row, max_col = bbox

    cut_image = image[min_row:max_row, min_col:max_col].copy()
    return cut_image

def getShapeFromBbox(bbox):
    min_row, min_col, max_row, max_col = bbox
    return (max_row, max_col)

def point2int(point):
    # Round to closest int but also set negative values to zero
    return tuple([round(el) for el in point])
#     return tuple([max(round(el), 0) for el in point])

def transformPointsWithBbox(point, bbox, force = False):
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

def calculateResizeZoomFactors(original_shape, target_shape):
    """
    Calculates the zoom factors necessary to go from original_shape to target shape
    """
    tform = [] 
    for i in range(len(original_shape)):
        tform.append(target_shape[i] / original_shape[i])
    zoom_factors = np.array(tform)
    return zoom_factors

def resizePointWithTransform(point, zoom_factors):
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

def reflectPoint180degrees(point, origin):
    new_point = 2*origin[0] - point[0], 2*origin[1] - point[1]
    return new_point

def flipPoint(point, shape):
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

def translatePoint(point, offset):
    new_point =(point[0] + offset[0] , point[1] + offset[1])
    return new_point


def calckwargs(input_shape, target_shape, rotation_angle, first_resize_step = True):
    kwargs = []
    if first_resize_step:
        first_zoom_factors = calculateResizeZoomFactors(input_shape, target_shape)
        kwargs.append({"zoom_factors": first_zoom_factors})
        
    if rotation_angle is not None:
        translation, shape_after_rotation = calculateRotateTranslation(target_shape, rotation_angle)
        center = findCenter(shape_after_rotation)
        kwargs.append({"rotation": {"angle" : rotation_angle, "center": center, "translation": translation}})

        zoom_factors_after_rotation = calculateResizeZoomFactors(shape_after_rotation, target_shape)
        kwargs.append({"zoom_factors": zoom_factors_after_rotation})

    # If we don't rotate but the first resize step hasn't been done, we still need to rotate
    elif not first_resize_step:
        first_zoom_factors = calculateResizeZoomFactors(input_shape, target_shape)
        kwargs.append({"zoom_factors": first_zoom_factors})
    

    if not kwargs:
        raise ValueError("kwargs is empty, something probably went wrong.")

    if "shape_after_rotation" in locals():
        return kwargs, shape_after_rotation
    else:
        return kwargs

def calculateRotateTranslation(original_shape, angle):
    zeros = np.zeros(original_shape)
    new_zeros, translation = rotate(zeros, angle, resize=True)
    return translation, new_zeros.shape

def cutOutCoordsAfterRotation(coords, target_shape):
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
        new_coord = transformPointsWithBbox(coord, bbox)
        new_coords.append(new_coord)
    return new_coords, bbox

def processWithoutZoomingOut(detected_transcript_paths, out_dir, target_shape, sigma, gene_map, corresponding_dapis=None, corresponding_cell_masks = None, lower_thresh=10, upper_thresh=101, model=None, device = None, return_arrays=False, cells=None, randomize=False, rng=None, embed_batch_size=256):
    # Two input modes:
    #   * file-based (default): read each cell's transcripts/dapi/cell-mask from
    #     detected_transcript_paths / corresponding_dapis / corresponding_cell_masks.
    #   * in-memory (cells != None): an iterable of
    #     (stem, transcripts_df, dapi_array, cell_mask_array); nothing is read from disk.
    #
    # When return_arrays is True the resized dapi/cell masks, per-gene embeddings
    # and zoomed transcripts are collected and returned instead of being written
    # as per-cell files, so the caller can checkpoint them compactly per tile:
    #   {"embeddings":[(stem,gene_id,arr)], "dapi_masks":[(stem,arr)],
    #    "cell_masks":[(stem,arr)], "transcripts":[(stem,df)],
    #    "random_embeddings":[(stem,gene_id,arr)]}
    #
    # When randomize is True, each embedded gene is ALSO embedded after scattering
    # its spots to uniformly-random pixels inside the cell mask -- the randomised
    # control. Done in the same loop (same model on the GPU), so it reuses the
    # already-resized dapi/cell mask and avoids a second pass over the data. rng
    # is an optional numpy Generator for reproducibility.
    out_dir.mkdir(exist_ok=True)
    in_memory = cells is not None
    if randomize and rng is None:
        rng = np.random.default_rng()
    collected = {"embeddings": [], "dapi_masks": [], "cell_masks": [], "transcripts": [], "random_embeddings": []}

    # Embeddings are batched on the GPU rather than computed one image at a time:
    # (gene_img, dapi) pairs are queued with their (target list, stem, gene_id) and
    # flushed in groups of embed_batch_size through embedSubcellBatch (a vectorised,
    # exact equivalent of createSubcellRepresentation). Used only in return_arrays
    # mode; the file-output path keeps the per-image behaviour.
    _embed_queue = []  # (target_key, stem, gene_id, gene_img, dapi_img)

    def _flush_embeddings():
        if not _embed_queue:
            return
        vectors = embedSubcellBatch([q[3] for q in _embed_queue],
                                    [q[4] for q in _embed_queue], model, device)
        for (target, stem, gene_id, _g, _d), vec in zip(_embed_queue, vectors):
            collected[target].append((stem, gene_id, vec))
        _embed_queue.clear()

    def _queue_embedding(target, stem, gene_id, gene_img, dapi_img):
        _embed_queue.append((target, stem, gene_id, gene_img, dapi_img))
        if len(_embed_queue) >= embed_batch_size:
            _flush_embeddings()

    if not in_memory:
        detected_transcripts_output_dir = out_dir / "detected_transcripts_zoomed"
        detected_transcripts_output_dir.mkdir(exist_ok=True)
        if corresponding_dapis is not None and not return_arrays:
            dapi_output_dir = out_dir / "dapi_zoomed"
            dapi_output_dir.mkdir(exist_ok=True)
        if corresponding_cell_masks is not None and not return_arrays:
            cell_mask_output_dir = out_dir / "cell_mask_zoomed"
            cell_mask_output_dir.mkdir(exist_ok=True)

    n = len(cells) if in_memory else len(detected_transcript_paths)
    for i in range(n):
        if in_memory:
            stem, detected_transcripts, dapi, cell_mask = cells[i]
            detected_transcripts = detected_transcripts.copy()
            has_dapi = dapi is not None
            has_cell = cell_mask is not None
        else:
            detected_transcript_path = Path(detected_transcript_paths[i])
            stem = detected_transcript_path.stem
            detected_transcripts = pd.read_parquet(detected_transcript_path)
            has_dapi = corresponding_dapis is not None
            has_cell = corresponding_cell_masks is not None
            if has_dapi:
                dapi_prefix = Path(corresponding_dapis[i]).stem
                dapi = io.imread(corresponding_dapis[i])
            if has_cell:
                cell_prefix = Path(corresponding_cell_masks[i]).stem
                cell_mask = io.imread(corresponding_cell_masks[i])

        if detected_transcripts.empty:
            continue

        zoom_factors = calculateResizeTform(dapi.shape, target_shape)

        # Vectorised coordinate resize (was a per-transcript Python loop allocating
        # a numpy array per point). np.rint matches point2int's round-half-to-even.
        new_rows = np.rint(np.asarray(detected_transcripts["subcell_row"], dtype=float) * zoom_factors[0]).astype(np.int64)
        new_cols = np.rint(np.asarray(detected_transcripts["subcell_col"], dtype=float) * zoom_factors[1]).astype(np.int64)

        detected_transcripts["subcell_row"] = new_rows
        detected_transcripts["subcell_col"] = new_cols

        if in_memory:
            collected["transcripts"].append((stem, detected_transcripts))
        else:
            detected_transcripts.to_parquet(detected_transcripts_output_dir / detected_transcript_path.name, index=False)

        if has_dapi:
            ## transformation of dapi
            dapi = resize(dapi, target_shape)
            if return_arrays:
                # keep float64 dapi for the embedding below; store a float32 copy
                collected["dapi_masks"].append((stem, dapi.astype(np.float32)))
            else:
                dapi_output_file = dapi_output_dir / f"{dapi_prefix}_DAPI.tif"
                if not dapi_output_file.is_file():
                    io.imsave(dapi_output_file, dapi, check_contrast=False)
                elif model is not None:
                    # If model is given, we make the embedding directly and need the dapi image
                    dapi = io.imread(dapi_output_file)
        if has_cell:
            ## transformation of cell_mask
            cell_mask = resize(cell_mask, target_shape)
            if return_arrays:
                collected["cell_masks"].append((stem, cell_mask.astype(np.float32)))
            elif not in_memory:
                cell_mask_output_file = cell_mask_output_dir / f"{cell_prefix}_cell-mask.tif"
                if not cell_mask_output_file.is_file():
                    io.imsave(cell_mask_output_file, cell_mask, check_contrast=False)

        # Valid pixels for the randomised control: anywhere inside the cell mask.
        valid_coords = np.argwhere(cell_mask) if (randomize and has_cell) else None
        # A constant image makes the (batched) normalisation divide by a zero std
        # -- the old per-image path raised ValueError and skipped; guard for that.
        dapi_ok = has_dapi and float(np.std(dapi)) > 0

        # Group spots by gene with numpy (sort once, split on the unique genes)
        # instead of a pandas boolean filter per gene -- ~100x faster per cell.
        genes_arr = detected_transcripts["gene"].to_numpy()
        order = np.argsort(genes_arr, kind="stable")
        genes_s, rows_s, cols_s = genes_arr[order], new_rows[order], new_cols[order]
        uniq, starts = np.unique(genes_s, return_index=True)
        ends = np.append(starts[1:], len(genes_s))

        for gene, s, e in zip(uniq, starts, ends):
            occurences = int(e - s)
            if not (lower_thresh < occurences < upper_thresh):
                continue
            gene_id = gene_map[gene]

            blurred = gaussian(_coords_to_img(rows_s[s:e], cols_s[s:e], target_shape), sigma)
            if model is None:
                io.imsave(out_dir / f"{stem}_genID{gene_id}genName{gene}_spots{occurences}_rotated0_blurred{sigma:0.1f}.tif", blurred, check_contrast=False)
            elif return_arrays:
                # Queue for batched embedding (see _queue_embedding above).
                if dapi_ok and float(np.std(blurred)) > 0:
                    _queue_embedding("embeddings", stem, gene_id, blurred, dapi)

                # Randomised control: same gene, same count, but spots
                # scattered to random pixels inside the cell mask.
                if randomize and valid_coords is not None and len(valid_coords) > 0:
                    pts = valid_coords[rng.integers(len(valid_coords), size=occurences)]
                    rand_blurred = gaussian(_coords_to_img(pts[:, 0], pts[:, 1], target_shape), sigma)
                    if dapi_ok and float(np.std(rand_blurred)) > 0:
                        _queue_embedding("random_embeddings", stem, gene_id, rand_blurred, dapi)
            else:
                # File-output mode: per-image embedding (unchanged behaviour).
                if (np.any(blurred)) and (np.any(dapi)):
                    try:
                        embedding = createSubcellRepresentation(blurred, dapi, model, device)
                    except ValueError:
                        continue
                    io.imsave(out_dir / f"{stem}_genID{gene_id}genName{gene}_spots{occurences}_rotated0_blurred{sigma:0.1f}_embedding.tif", embedding.cpu().detach().numpy(), check_contrast=False)

    _flush_embeddings()
    if return_arrays:
        return collected


def processWithZoomingOut(image_paths, out_dir, target_shape, sigma, number_of_rotations, corresponding_dapis=None, corresponding_cell_masks = None, first_resize_step = False, save_new_coords=False):
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

    # TMP PLEASE CHANGE BACK AFTER YOURE DONE
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

                kwargs, shape_after_rotation = calckwargs(img.shape, target_shape, rotation_angle, first_resize_step=first_resize_step)
                # don't do final resize step but we're first gonna cut out to fix the zooming problem
                kwargs = kwargs[:-1]

                coords = convertImageToCoords(img)

                new_coords = []
                for coord in coords:
                    new_coords.append(point2int(registerPoint(coord, kwargs)))
                new_coords = [coord for coord in new_coords if all(ele >= 1 for ele in coord)]
                            
                new_img = coordsToImg(new_coords, shape_after_rotation)
                
                new_coords, bbox = cutOutCoordsAfterRotation(new_coords, target_shape)
                shape_after_bbox = bbox[2] - bbox[0], bbox[3] - bbox[1]
                zoom_factors_after_bbox_cutting = calculateResizeZoomFactors(shape_after_bbox, target_shape)

                final_coords = []
                for new_coord in new_coords:
                    final_coord = resizePointWithTransform(new_coord, zoom_factors_after_bbox_cutting)
                    final_coords.append(final_coord)

                if save_new_coords:
                    rotated_coords_df = pd.DataFrame(final_coords, columns=['row', 'col'])
                    rotated_coords_df.to_csv(coords_output_dir / f"{img_path.stem}_rotated{rotation_angle}_blurred{sigma:1}_gene-coords.csv", index=False)
                new_img = coordsToImg(final_coords, target_shape)
                
                
                blurred = gaussian(new_img, sigma)
                io.imsave(out_dir / f"{img_path.stem}_rotated{rotation_angle}_blurred{sigma:1.1f}.tif", blurred, check_contrast=False)

                if corresponding_dapis is not None:
                    dapi_output_file = dapi_output_dir / f"{img_path.stem}_rotated{rotation_angle}_blurred{sigma:1.1f}_DAPI.tif"
                    ## transformation of dapi
                    if first_resize_step:
                        dapi = resize(dapi, target_shape)
                    rotated, shape_after_rotation = rotate(dapi, rotation_angle, resize=True)
                    bbox_cut = cutBboxFromImage(bbox, rotated)
                    resized= resize(bbox_cut, target_shape)
                    io.imsave(dapi_output_file, resized, check_contrast=False)
                if corresponding_cell_masks is not None:
                    cell_mask_output_file = cell_mask_output_dir / f"{img_path.stem}_rotated{rotation_angle}_blurred{sigma:1.1f}_cell-mask.tif"
                    ## transformation of cell_mask
                    if first_resize_step:
                        cell_mask = resize(cell_mask, target_shape)
                    rotated, shape_after_rotation = rotate(cell_mask, rotation_angle, resize=True)
                    bbox_cut = cutBboxFromImage(bbox, rotated)
                    resized = resize(bbox_cut, target_shape)
                    io.imsave(cell_mask_output_file, resized, check_contrast=False)

        else:
            kwargs = calckwargs(img.shape, target_shape, None, first_resize_step=first_resize_step)
            coords = convertImageToCoords(img)

            new_coords = []
            for coord in coords:
                new_coords.append(point2int(registerPoint(coord, kwargs)))
            new_coords = [coord for coord in new_coords if all(ele >= 1 for ele in coord)]

            new_img = coordsToImg(new_coords, target_shape)
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

# Works for real data
def matchGenesWithDapis(genes_glob, dapi_dir, dapi_prefix = "cellID", dapi_glob_pattern = None):
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

# This one only works for simulated data
# def matchGenesWithDapis(genes_glob, dapi_dir, dapi_prefix = "icell", dapi_glob_pattern = None):
#     corresponding_dapis = []
#     for path in tqdm(genes_glob):
#         cell_id = re.findall(f"{dapi_prefix}(\d+)", path)[0]
#         target_dapi_path =glob(str(dapi_dir / f"*_{dapi_prefix}{cell_id}*.tif"))[0]
#         corresponding_dapis.append(target_dapi_path)
#     return corresponding_dapis



def matchDapisWithCell(cell_mask_dir, corresponding_dapis, replace_dapi_to_cell={"dapi", "cell-mask"}):
    corresponding_cell_masks = []
    for dapi_path in corresponding_dapis:
        tmp_adapted_cell_path = Path(dapi_path).name
        for key, value in replace_dapi_to_cell.items():
            tmp_adapted_cell_path = tmp_adapted_cell_path.replace(key, value)

        corresponding_cell_mask_path = cell_mask_dir / tmp_adapted_cell_path
        corresponding_cell_masks.append(corresponding_cell_mask_path)
    return corresponding_cell_masks


if __name__ == '__main__':
    # all_imgs_paths = glob("/mnt/data/david/simulated_cells/with_dapi/damn_we_need_foci_again/gene_expression/*.tif")
    # dapi_dir = Path("/mnt/data/david/simulated_cells/with_dapi/damn_we_need_foci_again//nuclei_filtered/")
    # cell_dir = Path("/mnt/data/david/simulated_cells/with_dapi/damn_we_need_foci_again//cell_filtered/")
    # out_dir = Path("/mnt/data/david/simulated_cells/with_dapi/damn_we_need_foci_again/coordinate_parsed_with_coords/")
    # out_dir.mkdir(exist_ok = True)

    # target_shape = (100,100)
    # sigma = 1.5
    # number_of_rotations = 5
    # corresponding_dapis = matchGenesWithDapis(all_imgs_paths, dapi_dir, dapi_prefix = "cellID")
    # corresponding_cell_masks = matchDapisWithCell(cell_dir, corresponding_dapis, replace_dapi_to_cell={"dapi": "cell"})
    # print("started")

    # processWithZoomingOut(all_imgs_paths, out_dir, target_shape, sigma, number_of_rotations, corresponding_dapis = corresponding_dapis, corresponding_cell_masks = corresponding_cell_masks, first_resize_step = False, save_new_coords=True)

    all_imgs_paths = glob("/mnt/data/syno_voet_backup/Xenium/tejpar_16CRC_xenium/IMU029/parsed/single_cells/tile14/detected_transcripts/tile14_xenium_segmentation_GElabeled_cellID13274.parquet")
    dapi_dir = Path("/mnt/data/syno_voet_backup/Xenium/tejpar_16CRC_xenium/IMU029/parsed/single_cells/tile14/single_dapi_masks/")
    cell_dir = Path("/mnt/data/syno_voet_backup/Xenium/tejpar_16CRC_xenium/IMU029/parsed/single_cells/tile14/single_cell_masks/")
    
    out_dir = Path(' /mnt/data/syno_voet_backup/Xenium/tejpar_16CRC_xenium/IMU029/parsed/single_cells/tile14/tmp_out_dir_debugging/')

    target_shape = (100,100)
    sigma = 1.5
    corresponding_dapis = matchGenesWithDapis(all_imgs_paths, dapi_dir, dapi_prefix = "cellID")
    corresponding_cell_masks = matchDapisWithCell(cell_dir, corresponding_dapis, replace_dapi_to_cell={"dapi": "cell"})
    print("started")
    processWithoutZoomingOut(all_imgs_paths, out_dir, target_shape, sigma, gene_map = {1:'fake'}, corresponding_dapis = corresponding_dapis, corresponding_cell_masks = corresponding_cell_masks, save_new_coords=True)

