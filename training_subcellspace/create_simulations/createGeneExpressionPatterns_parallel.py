import itertools
import simfish
import pandas as pd
import random
import numpy as np
from skimage.transform import resize
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import dill

from scipy.stats import kde

absolute_lower = 10
absolute_upper = 100

def create_spot_image():
    imgs = simfish.simulate_images(6, 2, n_spots = 50, random_n_spots=True, image_shape=(100,100), sigma=15, noise_level=0) 

    fig, axs = plt.subplots(3,2)
    axs = axs.flatten()
    for i, (img, ground_truth) in enumerate(imgs):
        print(ground_truth.shape)
        axs[i].imshow(img)
    plt.show()

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


def create_gene_expression_img(rows, cols, shape=(100,100)):
    zeros = np.zeros(shape, dtype=np.float)
    for row,col in zip(rows, cols):
        zeros[int( row ),int( col )] = 1.0
    return zeros

def do_simulate_pattern(cell_id, pattern, it, proportion_pattern, kde_distribution):
    # n_spots = random.randint(5, 100)
    n_spots = round(kde_distribution.resample(1)[0][0])

    try:
        instance_coord = simfish.simulate_localization_pattern("/mnt/data/david/simulated_cells/templates/", i_cell = cell_id, n_spots = n_spots, pattern = pattern, proportion_pattern = proportion_pattern)
    except ValueError:
        return
    new_rows, new_cols, new_nuc, new_cell = parse_simulation(instance_coord)

    df = pd.DataFrame(list(zip(new_rows, new_cols)), columns=["row", "col"])

    ge_image = create_gene_expression_img(new_rows, new_cols, (100,100))

    tmp_pattern = pattern.replace("_", "-")

    # ## save:
    io.imsave(gene_expression_dir / f"{tmp_pattern}_{it}it_{n_spots}spots_cellID{cell_id}_prop{proportion_pattern:.2f}_gene_expression.tif", ge_image, check_contrast=False)
    io.imsave(nuc_dir / f"{tmp_pattern}_cellID{cell_id}_dapi.tif", new_nuc, check_contrast=False)
    io.imsave(cell_dir / f"{tmp_pattern}_cellID{cell_id}_cell.tif", new_cell, check_contrast=False)
    df.to_csv(gene_coords_dir  / f"{tmp_pattern}_{it}it_{n_spots}spots_cellID{cell_id}_prop{proportion_pattern:.2f}_coords.csv", index=False)


def parse_simulation(instance_coord):
    rows = instance_coord["rna_coord"][:,1]
    cols = instance_coord["rna_coord"][:,2]
    zoom_factors = 100 / instance_coord["nuc_mask"].shape[0], 100 / instance_coord["nuc_mask"].shape[1]
    new_nuc = resize(instance_coord["nuc_mask"], (100,100))
    new_cell = resize(instance_coord["cell_mask"], (100,100))
    new_rows = []
    new_cols = []

    for point in zip(rows, cols):
        new_point = resize_point_with_transform(point, zoom_factors)
        new_rows.append(new_point[0])
        new_cols.append(new_point[1])

    return new_rows, new_cols, new_nuc, new_cell


    # ## save:
    io.imsave(gene_expression_dir / f"{tmp_pattern1}x{tmp_pattern2}_{it}it_{n_spots_for_n1}spots1_{n_spots_for_n2}spots2_cellID{cell_id}_prop1-{proportion_pattern1:.2f}_prop2-{proportion_pattern2:.2f}_gene_expression.tif", ge_image, check_contrast=False)
    io.imsave(nuc_dir / f"{tmp_pattern1}x{tmp_pattern2}_{it}it_{n_spots_for_n1}spots1_{n_spots_for_n2}spots2_cellID{cell_id}_prop1-{proportion_pattern1:.2f}_prop2-{proportion_pattern2:.2f}_dapi.tif", new_nuc, check_contrast=False)
    # io.imsave(cell_dir / f"{tmp_pattern}_cellID{i}_cell.tif", new_cell, check_contrast=False)
    total_df.to_csv(gene_coords_dir  / f"{tmp_pattern1}x{tmp_pattern2}_{it}it_{n_spots_for_n1}spots1_{n_spots_for_n2}spots2_cellID{cell_id}_prop1-{proportion_pattern1:.2f}_prop2-{proportion_pattern2:.2f}_coords.csv", index=False)


if __name__ == '__main__':
    out_dir =  Path("/path/to/out/dir/")
    out_dir.mkdir(exist_ok=True)

    # # Only needed if wanting to sample from realistic n_spots
    from scipy.stats import gaussian_kde
    # Load the saved input data
    hist_data = np.load("./validation_n_spots_distribution.npy")
    # Recreate the KDE object
    kde = gaussian_kde(hist_data)

    # Downloads simulated cells template
    simfish.load_extract_template(out_dir)

    gene_expression_dir =out_dir / "gene_expression"
    gene_coords_dir =out_dir / "gene_coords"
    nuc_dir =out_dir / "nuclei_filtered"
    cell_dir =out_dir / "cell_filtered"

    gene_expression_dir.mkdir(exist_ok=True)
    gene_coords_dir.mkdir(exist_ok=True)
    nuc_dir.mkdir(exist_ok=True)
    cell_dir.mkdir(exist_ok=True)
    pattern_strings = ["intranuclear", "extranuclear", "perinuclear", "cell_edge", "pericellular", "protrusion", "random", "nuclear_edge"]

    prop_patterns = np.arange(0.10, 1.00, 0.01) # For the EMD calculation data
    prop_patterns = [0.9] # for the CVAE training data

    for pattern in (pattern_strings):
        for cell_id in tqdm(range(0,318)):
            for prop in prop_patterns:
                Parallel(n_jobs=32, backend="multiprocessing")(delayed(do_simulate_pattern)(cell_id, pattern, it, prop, kde_distribution = kde) for it in range(40))
