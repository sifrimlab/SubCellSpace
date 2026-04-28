from skimage import io
from skimage.measure import regionprops_table
from skimage.draw import disk
from pyxelperfect.measure import measureLabeledImage
from pyxelperfect.decorators import measureTime
from pyxelperfect.display import plotSpatial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

# @measureTime
def create_gene_matrix(labeled_image, decoded_df, rowname = "row", colname="col", genename="gene", measured_df = None, key_to_match_in_labeled_image="image_label"):
    # We add option to give measured df such that this is compatible with splitting segmentation into single cells of the SEP workflow
    if measured_df is None:
        measured_df = measureLabeledImage(labeled_image)

    def assign_genes_to_spots(labeled_image, decoded_df, cell_properties):
        gene_dict = {} # keys = label integers, values = dict{gene: count}

        ### Archived
        # n_labels = np.unique(labeled_image)
        # for label in n_labels:
        #     gene_dict[label] = {}
        ###

        # We've pivoted to shipping the measured cell properties instead of taking just all unique elements in the labeled image, for filtering purposes. However, this makes it so that some genes will want to be assigned to cells that aren't in the measured df, so we need to make a caveat for that
        for label in list(cell_properties[key_to_match_in_labeled_image]):
            gene_dict[label] = {}

        # if filter_unrecognized:
        #     decoded_df = decoded_df[decoded_df['gene'].isnull()!=True]

        assigned_to_label = []
        for row in decoded_df.itertuples():
            row = row._asdict()
            try:
                label = labeled_image[round(row[rowname]), round(row[colname])]
            except IndexError:
                try:
                    label = labeled_image[int(row[rowname]), int(row[colname])]
                except IndexError:
                    label = labeled_image[int(row[rowname]) -1, int(row[colname]) -1]

            if label != 0:
                if label in gene_dict.keys():
                    gene_dict[label][row[genename]] = gene_dict[label].get(row[genename], 0) + 1 

            assigned_to_label.append(label) # At the moment, genes still get assigned to cells even if that cell isn't in matched cell/dapi

        # then rename the label to me a bit more clear
        for label in list(cell_properties[key_to_match_in_labeled_image]):
            if label == 0:
                continue
            gene_dict[f"{cell_properties[cell_properties[key_to_match_in_labeled_image] == label].iloc[0]['cell_label']}"] = gene_dict.pop(label)

        # Then add cell assignment to decoded df to extract into single-gene images later on
        decoded_df["cell_label_assigned"] = assigned_to_label
        return gene_dict


    gene_dict = assignGenesToSpots(labeled_image, decoded_df, measured_df)
    try:
        gene_dict.pop(0)
    except KeyError:
        pass
        # print("No label 0 found.")


    sorted_gene_keys = sorted(list(gene_dict.keys()), key=lambda x: int(x.split('_')[-1]))




    gene_dict_list = [gene_dict[i] for i in sorted(list(gene_dict.keys()), key=lambda x: int(x.split('_')[-1]))]

    keys = set().union(*gene_dict_list)
    final = {k: [d.get(k, 0) for d in gene_dict_list] for k in keys}

    count_matrix = pd.DataFrame(final)
    count_matrix["cell_label"] = sorted_gene_keys
    count_matrix.set_index("cell_label", inplace=True)

    return count_matrix, decoded_df


def parse_cell_label(cell_label: str):
    str_split = cell_label.split("_")
    x = int(str_split[0][1:])
    y = int(str_split[1][1:])
    label = int(str_split[2])
    return (x,y,label)

def concat_count_matrices(glob_pattern):
    dfs = [pd.read_csv(file, index_col = 0) for file in glob(glob_pattern)]
    concat_df = pd.concat(dfs)
    concat_df = concat_df.fillna(0)
    return concat_df

