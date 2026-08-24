import numpy as np
import pandas as pd
from typing import Tuple
from pyxelperfect.measure import measureLabeledImage


def index_labeled_image_faster(labeled_image, cell_label, measured_df=None, copy=True) -> Tuple:
    """indexLabeledImageFaster.

    Parameters
    ----------
    labeled_image :
        labeled_image
    cell_label :
        cell_label
    measured_df :
        measured_df, default = None -> measure within the function

    Returns
    -------
    Tuple(np.ndarray, Tuple, slice)
    1: bbox of labeled_image,
    2: bbox (min_row, min_col, max_row, max_cor)
    3: slice object of that bbox


    """
    if measured_df is None:
        measured_df = measureLabeledImage(labeled_image)
    cell_property_row = measured_df.loc[measured_df["image_label"] == cell_label]
    cell_bbox = cell_property_row["bbox"].iloc[0]
    if copy:
        cell_bbox_for_this_cell = labeled_image[cell_bbox[0]: cell_bbox[2], cell_bbox[1]:cell_bbox[3]].copy()
    else:
        cell_bbox_for_this_cell = labeled_image[cell_bbox[0]: cell_bbox[2], cell_bbox[1]:cell_bbox[3]]

    bbox_slice = np.s_[cell_bbox[0]: cell_bbox[2], cell_bbox[1]:cell_bbox[3]]

    return cell_bbox_for_this_cell, cell_bbox, bbox_slice
