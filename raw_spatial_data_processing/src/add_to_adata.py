import pandas as pd
from glob import glob
import anndata as ad
from pathlib import Path
from skimage import io
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
from scipy.sparse import bsr_matrix
import awkward as ak


def add_measured_dfs_as_obs_metadata(adata):
    global_rows, global_cols, areas = [],[],[]
    for tmp_row in adata.obs.itertuples():
        tile_nr = tmp_row.tile_nr 
        corresponding_cell = int(tmp_row.cell_id.split("_")[-1])
        measured_df = adata.uns["measured_df"][str(tile_nr)]
        measured_df = measured_df[measured_df["corresponding_cell"] == corresponding_cell]

        assert len(measured_df) == 1

        global_rows.append(measured_df["global_row"].iloc[0])
        global_cols.append(measured_df["global_col"].iloc[0])
        areas.append(measured_df["Area"].iloc[0])

    adata.obs["global_row"] = global_rows
    adata.obs["global_col"] = global_cols
    adata.obs["cell_size"] = areas

    return adata




def add_single_cell_info_to_adata(adata, coordinate_parsed_dir: Path, segmented_prefix):
    # empty = np.zeros((adata.shape[0], 10000))
    detected_transcripts = []
    empty_dapi = np.zeros((adata.shape[0], 10000))
    empty_cell = np.zeros((adata.shape[0], 10000))

    # nonzero_rows, nonzero_cols = np.nonzero(adata.X)

    not_found_counter = 0
    for i, data_point in tqdm(enumerate(adata), total=len(adata), desc = "Adding single cell info: "):
        tile_nr = data_point.obs["tile_nr"].iloc[0]
        corresponding_cell = data_point.obs["cell_id"].iloc[0].split("_")[-1]

        # gen_id = data_point.var["gene_label"].iloc[0]
        # gene_name = data_point.var["name"].iloc[0]


        # subcell detected transcripts
        target_file_detected_transcripts = coordinate_parsed_dir / f"tile{tile_nr}" / "detected_transcripts_zoomed" / f"tile{tile_nr}_{segmented_prefix}_GElabeled_cellID{corresponding_cell}.parquet"
        if target_file_detected_transcripts.is_file():
            detected_transcripts.append(pd.read_parquet(target_file_detected_transcripts))
        else:
            detected_transcripts.append(pd.DataFrame())
        # dapi
        target_file_dapi = coordinate_parsed_dir / f"tile{tile_nr}" / "dapi_zoomed" / f"tile{tile_nr}_{segmented_prefix}_dapi-mask_cellID{corresponding_cell}_DAPI.tif"

        # Some cells might not be saved because it had not trnascripts
        if target_file_dapi.is_file():
            img = io.imread(target_file_dapi)
            empty_dapi[i, :] = img.flatten()
        else:
            not_found_counter += 1
            continue

        # cell
        target_file_cell =coordinate_parsed_dir / f"tile{tile_nr}" / "cell_mask_zoomed" / f"tile{tile_nr}_{segmented_prefix}_cell-mask_cellID{corresponding_cell}_cell-mask.tif"

        if target_file_cell.is_file():
            img = io.imread(target_file_cell)
            empty_cell[i, :] = img.flatten()

    print(not_found_counter)

    # First we parse the dataframes to dicts to be able to be put in an awkward array
    for df in detected_transcripts:
        df.index = df.index.map(str)
    detected_transcripts = [df.to_dict(orient="list") for df in detected_transcripts]
    
    adata.obsm["detected_transcripts"] = ak.Array(detected_transcripts)
    adata.obsm["dapi_mask"]  = empty_dapi
    adata.obsm["cell_mask"]  = empty_cell

    return adata

def add_subcell_representation_to_adata(adata, coordinate_parsed_dir, segment_prefix, nr_latent_dims = 15):
    '''
    Note to self: for now it only checks where everything is between 10 and 101
    '''

    coordinate_parsed_dir = Path(coordinate_parsed_dir)

    new_subcell_layer = np.zeros((*adata.shape, nr_latent_dims))

    thresholded_rows, thresholded_cols = np.where(np.logical_and(adata.X > 10, adata.X < 101))
    # device = model["device"]
    
    for row, col in tqdm(zip(thresholded_rows, thresholded_cols), total=len(thresholded_rows), desc = "Adding subcell info"):
        adata_for_this_element = adata[row, col]
        assert len(adata_for_this_element) == 1
        tile_nr = adata_for_this_element.obs["tile_nr"].iloc[0]
        gene_name = adata_for_this_element.var["gene_name"].iloc[0]

        # Right now, I don't isolate the extra counted genes yet
        if "mosaic" in gene_name:
            continue

        gene_id = adata_for_this_element.var["gene_label"].iloc[0]
        cell_id =  adata_for_this_element.obs["cell_id"].iloc[0].split("_")[-1]
        value = int(adata_for_this_element.X[0,0])
        
        dapi = adata_for_this_element.obsm["dapi_mask"][0].reshape(100,100)
        
        target_file = coordinate_parsed_dir / f"tile{tile_nr}" / f"tile{tile_nr}_{segment_prefix}_GElabeled_cellID{cell_id}_genID{gene_id}genName{gene_name}_spots{value}_rotated0_blurred1.5_embedding.tif"
        if not target_file.is_file():
            # this just means that the cell was not parsed cause it was incomplete
            if np.amax(dapi) == 0:
                continue

            # If it reaches here, that means that there was not enough signal on the input image and a nulldvision error was thrown, we will just skip this one.
            print(target_file)
            continue
        embedding = io.imread(target_file)
        
        new_subcell_layer[row, col, :] = embedding
        
    adata.layers["subcell_embedding"] = new_subcell_layer

    return adata
    

