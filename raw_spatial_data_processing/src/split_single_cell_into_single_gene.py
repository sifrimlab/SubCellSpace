from tqdm import tqdm
import numpy as np
from pathlib import Path
from skimage import io
import json
from glob import glob

def split_labeled_image_into_single_gene_images(cell_labeled_image, detected_transcripts, cell_label, cell_bbox, gene_map, image_prefix, lower_thresh, upper_thresh, out_dir, cell_column_to_match = "cell_label_assigned"):
    import matplotlib.pyplot as plt

    gene_map = {k: int(v) for k,v in gene_map.items()} # Make sure values are ints, sometimes json is weird about integers

    min_row, min_col, max_row, max_col = cell_bbox

    detected_transcripts_for_this_cell = detected_transcripts[detected_transcripts[cell_column_to_match] == cell_label]

    unique_genes = detected_transcripts_for_this_cell["gene"].unique()
    
    for gene in unique_genes:
        gene_id = gene_map[gene]

        detected_transcripts_for_this_gene = detected_transcripts_for_this_cell[detected_transcripts_for_this_cell["gene"] == gene]

        occurences = len(detected_transcripts_for_this_gene)

        if (occurences > lower_thresh) and (occurences < upper_thresh):
            tmp_image = np.zeros(cell_labeled_image.shape)
            for tmp_row in detected_transcripts_for_this_gene.itertuples():
                try:
                    real_row = round(tmp_row.local_row) - min_row
                    real_col = round(tmp_row.local_col) - min_col
                    tmp_image[real_row, real_col] = 1
                except IndexError:
                    real_row = int(tmp_row.local_row) - min_row
                    real_col = int(tmp_row.local_col) - min_col
                    tmp_image[real_row, real_col] = 1

            io.imsave(out_dir / f"{image_prefix}_genID{gene_id}genName{gene}_spots{occurences}.tif", tmp_image,check_contrast=False)

