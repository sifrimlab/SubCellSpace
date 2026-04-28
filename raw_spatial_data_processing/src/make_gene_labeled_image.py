from skimage import io
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_gene_expression_image(detected_transcripts_df, img_shape):
    """
        Game plan:
        - Create image representing genes with label
        - for each bbox (= single nuclei), get bbox in labeled image
        - for each labeled bbox, get matching df entry to DataFrame --> not even necessary for initial concept
    """


    labeled_image = np.zeros(img_shape)
    gene_map ={k: v for v, k in enumerate(sorted(set(detected_transcripts_df["gene"])), 1)}

    for row in tqdm(detected_transcripts_df.itertuples()):
        labeled_image[round(row.row), round(row.col)] = gene_map[row.gene]

    return labeled_image, gene_map

def create_gene_map(detected_transcripts_df):
    gene_map ={k: v for v, k in enumerate(sorted(set(detected_transcripts_df["gene"])), 1)}

    return gene_map

