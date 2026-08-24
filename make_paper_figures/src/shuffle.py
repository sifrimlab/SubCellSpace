if __package__:
    from .adata_utils import convertRecordToDf
else:  # when run as a script from inside src/
    from adata_utils import convertRecordToDf
from skimage.filters import gaussian
from tqdm import tqdm
import awkward as ak
import pandas as pd
import numpy as np
import torch

def shuffleGenesInCell(cell_adata_row, default_transcript_columns = []):
    transcripts = convertRecordToDf(cell_adata_row.obsm["detected_transcripts"][0])
    if transcripts is None:
        return pd.DataFrame(columns=default_transcript_columns)
    # if transcripts.empty:
    #     return transcripts
    cell_mask = cell_adata_row.obsm["cell_mask"][0].reshape((100,100))
    
    new_rows, new_cols = [], []
    
    valid_coords = np.argwhere(cell_mask)
    
    new_points = valid_coords[np.random.choice(len(valid_coords), size=len(transcripts))]
    new_rows = new_points[:, 0]
    new_cols = new_points[:, 1]

    transcripts["subcell_row_shuffled"] = new_rows
    transcripts["subcell_col_shuffled"] = new_cols
    return transcripts

def shuffle_all_gene_expression(single_cell_adata, default_transcript_columns):
    detected_transcripts_list = []

    for tmp_row in tqdm(single_cell_adata):
        shuffled_transcripts = shuffleGenesInCell(tmp_row, default_transcript_columns = default_transcript_columns)
        detected_transcripts_list.append(shuffled_transcripts)

    for df in detected_transcripts_list:
        df.index = df.index.map(str)

    detected_transcripts = [df.to_dict(orient="list") for df in detected_transcripts_list]
    
    single_cell_adata.obsm["detected_transcripts"] = ak.Array(detected_transcripts)

    return single_cell_adata

def add_shuffled_embedding(single_cell_adata, model):

    def create_new_input_row(single_cell_row, model, lower_thresh = 10, upper_thresh = 101):
        dapi = single_cell_row.obsm["dapi_mask"][0].reshape((100,100))
        cell = single_cell_row.obsm["cell_mask"][0].reshape((100,100))
        
        points = convertRecordToDf(single_cell_row.obsm["detected_transcripts"][0])
        new_subcell_embedding_row = np.zeros((1, len(single_cell_row.var_names), 15))
        
        points_by_gene = dict(tuple(points.groupby("gene")))
        for i, unique_gene in enumerate(list(single_cell_row.var["gene_name"])):
    #         points_for_this_gene = points[points["gene"] == unique_gene]
            points_for_this_gene = points_by_gene.get(unique_gene)
            if points_for_this_gene is not None:
                occurences = len(points_for_this_gene)
                if (occurences > lower_thresh) and (occurences < upper_thresh):
                    rows, cols = points_for_this_gene["subcell_row_shuffled"], points_for_this_gene["subcell_col_shuffled"]
                    gene_expression_img = np.zeros(dapi.shape)
                    for tmp_row, tmp_col in zip(rows, cols):
                        gene_expression_img[int(tmp_row), int(tmp_col)] = 1
                    gauss = gaussian(gene_expression_img, 1.5)

                    gene_expression = model["cvae"].prepare_new_input(gauss)
                    dapi_prepared = model["cvae"].prepare_new_input(dapi)

                    input_img = torch.cat((gene_expression, dapi_prepared), dim=1)

                    model["cvae"].to(model['device'])
                    input_img = input_img.to(model['device'])
                    embedding, _, _ = model['cvae'].embed(input_img)
                    embedding = embedding.cpu().detach().numpy()    
                    new_subcell_embedding_row[0, i, :] = embedding
                
        return new_subcell_embedding_row

    new_embedding_layer = np.zeros((len(single_cell_adata), len(single_cell_adata.var_names), 15))
    for i, tmp_row in tqdm(enumerate(single_cell_adata), total = len(single_cell_adata), desc = "Making new shuffled subcell embeddings"):
        new_subcell_embedding_row = create_new_input_row(tmp_row, model)
        new_embedding_layer[i, :, :] = new_subcell_embedding_row
    single_cell_adata.layers["random_subcell_embedding"] = new_embedding_layer
    return single_cell_adata

