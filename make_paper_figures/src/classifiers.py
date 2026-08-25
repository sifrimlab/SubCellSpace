import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import ks_2samp, epps_singleton_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.linalg import sqrtm

import scanpy as sc
from tqdm import tqdm
import seaborn as sns


seed = 101

distance_threshold_1000obs_new_classifier_custom_random = 0.06

colorblind_palette = sns.color_palette("colorblind")

def loadPatternNoPatternClassifier(model_dir):
    model_dir = Path(model_dir)
    # path = model_dir / "rf_pattern_balanced_singlejob.pkl"
    path = model_dir / "rf_pattern_nonpattern_wo_foci.pkl"
    with open(path, 'rb') as p:
        rf_pattern_balanced_singlejob = pickle.load(p)
    return rf_pattern_balanced_singlejob

def loadPatternClassifier(pattern, model_dir):
    model_dir = Path(model_dir)
    path = model_dir / f"rf_{pattern}_OtherPatterns.pkl"
    with open(path, 'rb') as p:
        classifier = pickle.load(p)
    return classifier

def quantify_gene_patterness(adata, gene_of_interest, classifier, random_test_data, max_number_of_observations_to_consider = None, random_seed = 101):
    pattern = adata[adata.obs["gene_name"] == gene_of_interest].copy()
    if max_number_of_observations_to_consider is not None:
        if len(pattern) > max_number_of_observations_to_consider:
            pattern = sc.pp.subsample(pattern, n_obs = max_number_of_observations_to_consider, random_state=random_seed, copy=True)
    try:
        control = random_test_data[pattern.obs_names].copy()
    except KeyError as e:
        # If wwwe try to index but it fails, it might be cause we already matched before, so we check if lengths are same
        if len(pattern) == len(random_test_data):
            control = random_test_data.copy()
        # if it's not the same, we raise the error anyway
        else:
            raise KeyError(e)

    assert len(control) == len(pattern)

    pattern_index = list(classifier.classes_).index('pattern')
    
    pattern_score = classifier.predict_proba(pattern.X)[:,pattern_index]
    pattern.obs["classifier_proba"] = list(pattern_score)

    control_score = classifier.predict_proba(control.X)[:,pattern_index]
    control.obs["classifier_proba"] = list(control_score)

    distance = wasserstein_distance(control_score, pattern_score)

    statistic, pvalue = ks_2samp(pattern_score, control_score)
    
    pattern_bool = True if pvalue < 0.05 else False

    return distance, pvalue, pattern, control

def classify_all_genes(adata, classifier, random_test_data, earth_dist_threshold = distance_threshold_1000obs_new_classifier_custom_random, max_number_of_observations_to_consider=None, random_seed = seed,store_adata=True):
    """classifyAllGenes.

    Parameters
    ----------
    adata :
        adata
    classifier :
        classifier
    random_test_data :
        random_test_data
    max_number_of_observations_to_consider :
        max_number_of_observations_to_consider
    random_seed :
        random_seed
    store_adata :
        bool that determines wheter to store the data used for the statistical test in the results_df
    """
    # For the final df
    results_list = []

    # To add results to the adata itself
    pattern_col = np.array([None] * len(adata))
    dist_col = np.array([0.0] * len(adata))

    genes_to_test = adata.obs["gene_name"].unique()

    # This needs to be true for the correct indexing of test vs control
    assert all(adata.obs.index ==  random_test_data.obs.index)

    for gene in tqdm(genes_to_test):
        attribute_dict = {}
        dist, pvalue, pattern, control = quantify_gene_patterness(
                adata, gene, classifier, random_test_data,
                max_number_of_observations_to_consider=max_number_of_observations_to_consider,
                random_seed=seed
        )
        attribute_dict["gene_name"] = gene
        attribute_dict["earth_dist"] = dist
        attribute_dict["pval"] = pvalue
        attribute_dict["n_obs"] = len(pattern)
        attribute_dict["total_obs_avail"] = adata.obs["gene_name"].value_counts().get(gene, 0)
        attribute_dict["bonferonni_pval"] = pvalue * len(genes_to_test)
        attribute_dict["passes"] = attribute_dict["bonferonni_pval"] < 0.05
        attribute_dict["passes_earth_thresh"] =  dist > earth_dist_threshold

        if store_adata:
            attribute_dict["pattern"] = pattern
            attribute_dict["control"] = control

        pattern_bool = attribute_dict["passes"] and attribute_dict["passes_earth_thresh"]

        attribute_dict["gene_pattern"] =  pattern_bool

        results_list.append(attribute_dict)

        pattern_col[np.s_[np.where(adata.obs["gene_name"] == gene)]] = pattern_bool
        dist_col[np.s_[np.where(adata.obs["gene_name"] == gene)]] = dist
        
    adata.obs["earth_dist"] = dist_col
    adata.obs["gene_pattern"] = pattern_col

    results_df = pd.DataFrame(results_list)
    
    return results_df

def classify_all_genes_for_pattern_subtypes(result_df, multi_classifier):
    subtype_pattern_list = []
    final_class_list = []
    for tmp_row in result_df.itertuples():
        tmp_adata = tmp_row.pattern

        tmp_random_adata = tmp_row.control
        
        assert len(tmp_adata) == tmp_row.n_obs
        assert len(tmp_adata) == len(tmp_random_adata)

        class_scores = multi_classifier.predict_proba(tmp_adata.X)
        random_scores = multi_classifier.predict_proba(tmp_random_adata.X)

        adata_df = pd.DataFrame(class_scores, columns=multi_classifier.classes_)
        random_df = pd.DataFrame(random_scores, columns=multi_classifier.classes_)
        assert all(adata_df.columns == random_df.columns)
        
        max_patt = adata_df.drop("random",axis=1).median().idxmax()
        subtype_pattern_list.append(max_patt)
        
        if not tmp_row.passes:
            final_class_list.append("random")
            continue
        if not tmp_row.passes_earth_thresh:
            final_class_list.append("random")
        else:
            final_class_list.append(max_patt)
    
    result_df["subtype_pattern"] = subtype_pattern_list
    result_df["final_class"] = final_class_list


    return result_df
