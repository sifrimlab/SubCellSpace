import pickle
import random
import logging
import numpy as np
import scanpy as sc
from math import comb
from scipy.stats import ks_2samp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def compute_power_rf(params, rf_singlejob, adata_test, adata_control):
    """
    Function to compute the power of the RF classifier for a given parameter combination.
    
    inputs:
    params: tuple of pattern strength, RNA count, cell count
    rf_singlejob: RF classifier object. Should be created as a single job object.
    adata_test: AnnData object containing the test dataset.
    
    output:
    result, bonferroni_result: tuple of parameter combination and power, tuple of parameter combination and Bonferroni corrected power
    """
    logging.info('Starting compute_power_rf with params: %s', params)
    try:
        strength, rna_count, cell_count = params
        significant_count = 0
        bonferroni_count = 0

        for i in range(1000):
            # sample new gene. No random seed so that every time a different "gene" is sampled.
            pattern = create_simulated_gene(adata_test, mixed_patterns = True, pattern_strength= strength, rna_count = rna_count, sample_size = cell_count)
            control = adata_test_shuffled[pattern.obs.index]
            # control = create_simulated_gene(adata_test, pattern = 'random', mixed_patterns = False, rna_count = rna_count, sample_size = cell_count)

            # Calculate RF classification probabilities
            pattern_score = rf_singlejob.predict_proba(pattern.obsm["latent"])[:,1]
            control_score = rf_singlejob.predict_proba(control.obsm["latent"])[:,1]

            statistic, pvalue = ks_2samp(pattern_score, control_score)
            critical_value = 0.05
            adjusted_critical_value = critical_value / 5000
            if pvalue < critical_value:
                significant_count += 1
            if pvalue < adjusted_critical_value:
                bonferroni_count += 1
        
        with open('/mnt/data/david/power_analysis/power_rf.txt', 'a') as f:
            f.write(f'{strength}\t{rna_count}\t{cell_count}\t{significant_count / 1000}\t{bonferroni_count / 1000}\n')

        result = (f'{strength}_{rna_count}_{cell_count}', significant_count/1000)
        bonferroni_result = (f'{strength}_{rna_count}_{cell_count}', bonferroni_count/1000)

        logging.info('Finished compute_power_rf with params: %s', params)

        return result, bonferroni_result
    
    except ValueError as e:
        if str(e) == "Cannot take a larger sample than population when 'replace=False'":
            print(f"Error for parameters: {params}")
            return None
        else:
            raise e

if __name__ == '__main__':
    from dataprep import initialize_adata, create_simulated_gene
    import pandas as pd
    import pickle
    """ Main function to run the power analysis for the permutation test, currently set up for the latent space method (i.e. Chamfer distance).
    If you want to run the RF method, load the RF classifier object here, which should be trained as a single job object as we multiprocess the power analysis.
    
    Creates a dictionary with the power results for each parameter combination and saves it to a pickle file. Within the functions, the results are also saved to a text file. For further visualizations, see the notebook 'Power Analysis'.
    """
    # Set up logging
    logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    path = '/home/luna.kuleuven.be/u0147891/models/classifiers/rf_pattern_nonpattern_no_foci_strong_patterns_homemade_random_10-80spots_maxdepth32_maxfeatures5.pkl'
    with open(path, 'rb') as p:
        classifier = pickle.load(p)

    adata_split_cellID = sc.read_h5ad("/mnt/data/david/simulated_cells/with_dapi/custom_nynke_panel/custom_nynke_panel/custom_nynke_panel.h5ad")
    adata_shuffled = sc.read_h5ad("/mnt/data/david/simulated_cells/with_dapi/custom_nynke_panel/custom_nynke_panel/custom_nynke_panel_shuffled.h5ad")
    adata_split_cellID = initialize_adata(adata_split_cellID)
    adata_shuffled = initialize_adata(adata_shuffled)
    adata_split_cellID.obs["cell_id"] = adata_split_cellID.obs["cell_id"].astype(int) 
    adata_shuffled.obs["cell_id"] = adata_shuffled.obs["cell_id"].astype(int) 
    test_cell_ids =list(pd.read_csv("/home/luna.kuleuven.be/u0147891/sim_cells/test_cell_ids.csv", header=None)[0])
    adata_test = adata_split_cellID[adata_split_cellID.obs['cell_id'].isin(test_cell_ids)]
    adata_test_shuffled = adata_shuffled[adata_shuffled.obs['cell_id'].isin(test_cell_ids)]

    assert len(adata_test) == len(adata_test_shuffled)

    logging.info('Loaded the dataset')

    strengths = ['strong'] 
    # rna_counts =list( adata_test.obs['n_spots_interval'].unique())
    rna_counts = ['0-10']#.remove('0-10')

    # Generated 14 points between 5 and 7000 on a log scale:
    # points = np.logspace(np.log10(5), np.log10(7000), num=14)
    # cell_counts = np.round(points).astype(int) # output is separated by /t iso commas, so can't use directly in the code

    cell_counts = [5, 9, 15, 27, 46, 81, 142, 247, 432, 753, 1315, 2297, 4009, 7000]

    # Create a list of all combinations of strength, count, and cell_count
    combinations = [(strength, rna_count, cell_count) for strength in strengths for rna_count in rna_counts for cell_count in cell_counts]
    print(combinations)

    logging.info('Starting each param combination...')

    results = [compute_power_rf(params, classifier, adata_test, adata_test_shuffled) for params in combinations]

    # Convert the results to a dictionary
    power_results = {result[0]: result[1] for result, _ in results if result}
    bonferroni_power_results = {bonferroni_result[0]: bonferroni_result[1] for _, bonferroni_result in results if bonferroni_result}

    path = "/mnt/data/david/power_analysis/latent_space_logscale_uncorrected.pkl"
    bonferroni_path = "/mnt/data/david/power_analysis/latent_space_logscale_bonferroni.pkl"

    # Open the file in write-binary mode and dump the object
    with open(path, 'wb') as f:
        pickle.dump(power_results, f)

    # Open the file in write-binary mode and dump the object
    with open(bonferroni_path, 'wb') as f:
        pickle.dump(bonferroni_power_results, f)
