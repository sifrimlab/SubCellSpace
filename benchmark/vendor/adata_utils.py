import awkward as ak
import numpy as np
import pandas as pd
import anndata as ad

def convertRecordToDf(record):
    awkward_list = ak.to_list(record)
    if all([value is None for value in awkward_list.values()]):
        # Empty cell: return an empty frame that still carries the expected columns,
        # which are the keys of the record itself.
        return pd.DataFrame(columns=list(awkward_list.keys()))
    else:
        return pd.DataFrame.from_dict(awkward_list)

def decodeSubcellEmbedding(embedding, model):
    import torch   # imported here so the benchmark environment does not need torch
    changed_tensor = torch.from_numpy(embedding).float()
    decoded_tensor = model["cvae"].decode(changed_tensor.to(model["device"]))
    reconstruction = decoded_tensor[0,0, :, :].cpu().detach().numpy()
    return reconstruction

def getAllDetectedTranscripts(adata):
    all_transcript_dfs = [convertRecordToDf(record) for record in adata.obsm["detected_transcripts"]]
    all_transcripts_df = pd.concat(all_transcript_dfs)
    return all_transcripts_df

def getXYOfSingleCell(adata, slc):
    row = adata[slc]
    assert len(row) == 1
    return int(row.obs["original_cell_index"][0]), int(row.obs["original_gene_index"][0])

def reshape_and_filter_anndata_with_index_and_mapping(adata, layer = "subcell_embedding"):
    # Get the shapes of the single_cell data
    n_cells, n_genes, latent_dim = adata.layers[layer].shape

    # Reshape the latent embedding layer
    latent_embedding_reshaped = adata.layers[layer].reshape((n_cells * n_genes, latent_dim))

    # Create indices referring to the original observations
    cell_indices = np.repeat(np.arange(n_cells), n_genes)
    gene_indices = np.tile(np.arange(n_genes), n_cells)
    
    original_indices = np.repeat(adata.obs.index.values, n_genes)
    original_gene_names = np.tile(adata.var.index.values, n_cells)
    
    n_spots = adata.X[cell_indices, gene_indices]

    # Filter the reshaped data to keep only nonzero rows
    nonzero_mask = np.any(latent_embedding_reshaped != 0, axis=1)
    filtered_latent_embedding = latent_embedding_reshaped[nonzero_mask]
    filtered_cell_indices = cell_indices[nonzero_mask]
    filtered_gene_indices = gene_indices[nonzero_mask]
    filtered_original_indices = original_indices[nonzero_mask]
    filtered_original_gene_names = original_gene_names[nonzero_mask]
    filtered_n_spots = n_spots[nonzero_mask]
    
    # Create a new AnnData object with the filtered latent embeddings
    filtered_adata = ad.AnnData(X=filtered_latent_embedding)

    # Create the new .obs DataFrame by combining the cell and gene metadata
    cell_obs = pd.DataFrame(np.repeat(adata.obs.values, n_genes, axis=0), columns=adata.obs.columns)
    gene_var = pd.DataFrame(np.tile(adata.var.values, (n_cells, 1)), columns=adata.var.columns)
    new_obs = pd.concat([cell_obs.reset_index(drop=True), gene_var.reset_index(drop=True)], axis=1)

    # Filter the new .obs DataFrame to keep only the rows corresponding to nonzero latent embeddings
    filtered_obs = new_obs[nonzero_mask].reset_index(drop=True)

    # Add original cell and gene indices to the new .obs DataFrame
    filtered_obs['original_cell_index'] = filtered_cell_indices
    filtered_obs['original_gene_index'] = filtered_gene_indices
    filtered_obs['original_obs_index'] = filtered_original_indices
    filtered_obs['original_gene_name'] = filtered_original_gene_names
    filtered_obs['n_spots'] = filtered_n_spots
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('Inf')]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "70-80","80-90","90-100","100+"]
    n_spots_interval = pd.cut(filtered_obs['n_spots'], bins=bins, labels=labels)
    filtered_obs["n_spots_interval"] = n_spots_interval
    
    # Assign the filtered .obs to the filtered AnnData object
    filtered_adata.obs = filtered_obs

    # Set the index to the filtered original indices
    filtered_adata.obs.index = filtered_original_indices + "_"  + filtered_original_gene_names

    return filtered_adata
