from skimage import io
from skimage.filters import gaussian
from skimage.feature import blob_log
from pathlib import Path
import pandas as pd
import numpy as np
import ast

def isNotZeroInMask(row, mask):
    '''
    row is one row of the return
    '''
    return (mask[round(row[0]),round(row[1])] != 0)

def filterDotsBasedOnMask(dots, mask):
    dots_df = pd.DataFrame(dots)
    masked_dots = dots_df.apply(lambda x: isNotZeroInMask(x, mask), axis=1)
    filtered_df = dots_df[masked_dots]
    return_array = np.array(filtered_df)
    return return_array

def countSpotsOnExtraChannel(img, mask):
    dots = blob_log(img, min_sigma = 1, max_sigma = 3, threshold=0.005)
    filtered_dots = filterDotsBasedOnMask(dots, mask)
    return filtered_dots

def addBlobsToDetectedTranscripts(dots, detected_transcripts, grid, tile_nr, measured_df, cell_id, gene):
    row_slc, col_slc = grid.tile_boundaries[tile_nr]
    row_start, col_start = row_slc.start, col_slc.start

    min_row, min_col, max_row, max_col = ast.literal_eval(measured_df.loc[measured_df["corresponding_cell"]  == cell_id]["cell_bbox"].iloc[0])


    rows_list = []
    for tmp_row in dots:
        attribute_dict = {}
        dot_row, dot_col = tmp_row[0], tmp_row[1]

        attribute_dict['global_x'] = 0
        attribute_dict['global_y'] = 0
        attribute_dict['global_z'] = 3.0
        attribute_dict['gene'] = gene
        attribute_dict['transcript_id'] = gene
        attribute_dict['cell_id'] = cell_id
        attribute_dict['row'] = row_start + min_row +  dot_row 
        attribute_dict['col'] = col_start + min_col +  dot_col 
        attribute_dict['local_row'] = min_row + dot_row 
        attribute_dict['local_col'] = min_col + dot_col 
        attribute_dict['subcell_row'] = dot_row 
        attribute_dict['subcell_col'] = dot_col 
        attribute_dict['cell_label_assigned'] = cell_id
        
        rows_list.append(attribute_dict)

    result_df = pd.DataFrame(rows_list)

    if (result_df.empty) and not (detected_transcripts.empty):
        return detected_transcripts
    if (detected_transcripts.empty) and not (result_df.empty):
        return result_df
    else:
        concatenated = pd.concat([detected_transcripts, result_df])
        concatenated = concatenated.drop_duplicates()
        return concatenated
