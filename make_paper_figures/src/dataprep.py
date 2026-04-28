# Author: nynkekatinka
import pandas as pd
import scanpy as sc
import numpy as np
import sklearn.model_selection as skm
import faiss
from sklearn.neighbors import NearestNeighbors
import anndata as ad

seed = 101

def map_interval(interval):
    if interval == '0-10':
        return '0-10'
    elif interval in ['10-20', '20-30']:
        return '10-30'
    elif interval in ['30-40', '40-50', '50-60']:
        return '30-60'
    elif interval in ['70-80', '80-90', '90-100']:
        return '70-100'
    elif interval == '100+':
        return '100+'
    else:
        return None
    
def initialize_adata(adata):
    """
    Modify the AnnData object to include columns necessary for downstream analysis, namely:
        * pattern_strength: percentage of RNA in localization pattern. The percentages for low, intermediate and strong pattern strengths are used from SimFISH.
        * rna_count: Include modified RNA count intervals for easier interpretation of visualization. Intervals listed in map_interval(interval). 
        * pattern: Include a column for each pattern, whether the observation is that pattern or another one. Used for RF classification of specific patterns. 
        """
    
    # Include pattern strength in the adata object
    choices = ['strong', 'intermediate', 'low']
    conditions = [
        (adata.obs['prop'] == 0.9) | ((adata.obs['prop'] == 0.4) & (adata.obs['pattern'] == 'protrusion')),
        (adata.obs['prop'] == 0.5) | ((adata.obs['prop'] == 0.2) & (adata.obs['pattern'] == 'protrusion')),
        (adata.obs['prop'] == 0.1) | ((adata.obs['prop'] == 0.0) & (adata.obs['pattern'] == 'protrusion'))
    ]
    adata.obs['pattern_strength'] = np.select(conditions, choices, default='unknown')

    # Include modified RNA count intervals in the adata object
    adata.obs['rna_count'] = adata.obs['n_spots_interval'].apply(map_interval)

    # Include a column for each pattern in the adata object.
    for pattern in adata.obs["pattern"].unique():
        adata.obs[pattern] = np.where(adata.obs["pattern"] == pattern, pattern, "other")

    return adata


def create_simulated_gene(adata, pattern: str = 'pericellular', mixed_patterns: bool = True, pattern_strength: str = "strong", rna_count: str = '10-30', sample_size: int = 5, random_seed: bool = False):
    """
    Subset the anndata object into a `1 gene multiple cells` object. Can filter the cells based on the number of spots, the pattern and the pattern strength.

    Parameters
    ----------
    adata : ad.AnnData object
        complete anndata object.
    pattern : str
        Type of subcellular expression pattern you want to filter on. Default is 'pericellular', which has the highest f1 RF score.
    mixed_patterns: bool
        True: all patterns can be included, False: only the pattern type specified in `pattern` is included. Default is False. 
    pattern_strength : str
        strength of the pattern, which is labeled as pattern_strength in the anndata object. Strong, which comes down to 45% of points fall in the pattern for protrusion, 90% of points are in pattern for all the other patterns. 
        If the pattern is random, then pattern_strength is not used, given that irrelevant for random when using the simFISH v2 definition of patterness (90% of points assigned to the pattern 'random' will still amount to 100% randomness).
    high_or_low : str
        Whether you want to filter genes with a higher or lower count than the given threshold. Default is lower.
        If none, then no threshold is chosen and mixed counts are included. 
    count_threshold : int
        Count threshold to filter on. Default is 11, so that genes with count 0-10 are selected. For high, the threshold is 140.
    mixed_counts: bool
        True: all counts are included, False: only counts above or below the threshold are included. Default is False.
        
    Returns
    -------
    ad.AnnData

    """
    # print("Note you changed the indexing of rna_count to n_spots_interval")
    adata_filtered = adata[(
                                (
                                    adata.obs['pattern'] == pattern if mixed_patterns == False
                                    else adata.obs['pattern'] != 'random'
                                )
                            ) & 
                           # (
                           #      adata.obs['n_spots_interval'] == rna_count
                           #  ) & 
                           (
                               adata.obs['pattern_strength'] == pattern_strength if pattern != 'random' 
                               else True
                            )
                        ].copy()

    if random_seed == True:
        subset = adata_filtered.obs.sample(n=sample_size, random_state = seed) 
    else:
        subset = adata_filtered.obs.sample(n=sample_size)
    
    adata_subset = adata_filtered[adata_filtered.obs.index.isin(subset.index)]


    return adata_subset


def balanceTrainingData_pattern_noPattern(adata, random_seed: int = 101):
    """
    Downsample the anndata object so that the test and control group for RF are equal size, while the control group is balanced for patterns and spot count. 
    Assumes there are less test observations than the other subcellular expression patterns combined. 

    Parameters
    ----------
    adata : ad.AnnData object
        complete anndata object.
    random_seed: int
        Seed for reproducability purposes. 
        
    Returns
    -------
    balanced_adata : ad.AnnData
        test and control group are equal size, while the control group is balanced for patterns and spot count
    """
        
    adata_test = adata[adata.obs['random_or_pattern']=='pattern']
    adata_control = adata[adata.obs['random_or_pattern']=='random']
    
    sample_size = len(adata_control)
    
    subset = adata_test.obs['n_spots_interval'].sample(n=sample_size, random_state=random_seed)
    
    adata_test_subset = adata_test[adata_test.obs.index.isin(subset.index)]
    
    # Concatenate adata_test and adata_control_subset
    adata = ad.concat([adata_test_subset, adata_control])

    return adata


def balanceTrainingData(adata, testPattern: str = 'pericellular', include_random: bool = False, random_seed: int = 101, equalize_all = False):
    """
    Downsample the anndata object so that the test and control group for RF are equal size, while the control group is balanced for patterns and spot count. 
    Assumes there are less test observations than the other subcellular expression patterns combined. 

    Parameters
    ----------
    adata : ad.AnnData object
        complete anndata object.
    testPattern : str
        Type of subcellular expression pattern you want to use as the test case. All other patterns will be used as control. 
    pattern_no_pattern: bool
        True: test pattern or no pattern, False: test patterns versus other patterns. Default is False.
    include_random: bool
        Whether the `no_pattern` pattern is included in the data or not. Default is False. 
    random_seed: int
        Seed for reproducability purposes. 
        
    Returns
    -------
    balanced_adata : ad.AnnData
        test and control group are equal size, while the control group is balanced for patterns and spot count
    """

    if include_random == False:
        adata = adata[adata.obs['pattern']!='random'].copy()

    adata_test = adata[adata.obs['pattern']==testPattern]
    adata_control = adata[adata.obs['pattern']!=testPattern]

    if not equalize_all:
        sample_size = int(len(adata_test)/len(adata_control.obs['pattern'].unique()))
    else:
        sample_size = min(adata.obs["pattern"].value_counts())
    adata_test = sc.pp.subsample(adata_test, n_obs=sample_size, random_state = random_seed, copy=True)

    pattern_groups = adata_control.obs.groupby('pattern')
    subset_obs_list = []

    # loop over group patterns, for each pattern, group by n_spots_interval. In those interval groups, sample `target_samplesize`. 
    for pattern, group in pattern_groups:
        subset = group['n_spots_interval'].sample(n=sample_size, random_state=random_seed)
        subset_obs_list.append(subset)
    
    # Concatenate all subset observations and use these indices to subset the adata_control AnnData object
    all_subset_obs = pd.concat(subset_obs_list)
    adata_control_subset = adata_control[adata_control.obs.index.isin(all_subset_obs.index)]
    
    # Concatenate adata_test and adata_control_subset
    adata = ad.concat([adata_test, adata_control_subset])
    adata.obs[testPattern] = np.where(adata.obs["pattern"] == testPattern, testPattern, "other")

    return adata

def balanceControlAndPattern(adata, rna_count_sample_size_dict = {'10-20': 5}, random_seed= 101):

    list_of_filtered_adatas = []

    for rna_count, sample_size in rna_count_sample_size_dict.items():
        adata_filtered = adata[
                                    adata.obs['n_spots_interval'] == rna_count
                            ].copy()
                
        subset = adata_filtered.obs.sample(n=sample_size, random_state = random_seed) 
        
        adata_subset = adata_filtered[adata_filtered.obs.index.isin(subset.index)]
        list_of_filtered_adatas.append(adata_subset)
    
    return_adata = ad.concat(list_of_filtered_adatas)


    return return_adata




def match_unique_control_using_faiss(control_adata, pattern_adata):
    """
    Matches each row in pattern_adata to the closest available n_spots in control_adata
    without reusing any control rows. Uses FAISS for fast search 

    Parameters:
    - control_adata: AnnData object containing unique 'n_spots' values.
    - pattern_adata: AnnData object with 'n_spots' values to match.

    Returns:
    - Subset of control_adata with unique matches for each row in pattern_adata.
    """

    # Convert n_spots to float32 (FAISS requires this)
    control_n_spots = control_adata.obs[['n_spots']].values.astype('float32')
    pattern_n_spots = pattern_adata.obs[['n_spots']].values.astype('float32')

    # Build FAISS index for fast nearest neighbor search
    index = faiss.IndexFlatL2(1)  # L2 (Euclidean) search for 1D data
    index.add(control_n_spots)  # Add control points to the index

    # Get k nearest neighbors for each pattern sample
    k_neighbors = min(len(pattern_adata), len(control_n_spots))  # Check up to 5000 nearest
    distances, indices = index.search(pattern_n_spots, k_neighbors)

    # Convert indices to actual control_adata indices
    control_obs_indices = np.array(control_adata.obs.index)  # Convert to NumPy array for fast lookup

    # Create a boolean mask to track used indices
    used_mask = np.zeros(len(control_n_spots), dtype=bool)

    # Vectorized matching
    matched_indices = np.full(len(pattern_n_spots), -1)  # Placeholder for matched indices

    for i in range(len(indices)):  # Loop over pattern_adata rows
        for j in range(k_neighbors):  # Loop over nearest neighbors
            idx = indices[i, j]  # Get candidate index
            if not used_mask[idx]:  # Check if it's available
                matched_indices[i] = idx  # Assign match
                used_mask[idx] = True  # Mark as used
                break  # Move to the next pattern row

    # Remove unmatched cases (if any)
    matched_indices = matched_indices[matched_indices != -1]

    # Subset control_adata based on matched indices
    matched_obs_indices = control_obs_indices[matched_indices]
    matched_adata = control_adata[control_adata.obs.index.isin(matched_obs_indices)]

    return matched_adata.copy()

def match_unique_control(pattern_adata, control_adata):
    """
    Matches each row in pattern_adata to the closest available n_spots in control_adata 
    without reusing any control rows.

    Parameters:
    - pattern_adata: AnnData object with 'n_spots' values to match.
    - control_adata: AnnData object containing unique 'n_spots' values.

    Returns:
    - Subset of control_adata with unique matches for each row in pattern_adata.
    """

    # Extract n_spots values
    pattern_n_spots = pattern_adata.obs[['n_spots']].values
    control_n_spots = control_adata.obs[['n_spots']].values

    # Fit Nearest Neighbors model on control
    k_neighbors = min(5000, len(control_n_spots))  
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(control_n_spots)
    distances, indices = nbrs.kneighbors(pattern_n_spots)

    # Convert indices to actual control_adata indices
    control_obs_indices = np.array(control_adata.obs.index)  # Convert to NumPy array for fast lookup

    # Create a boolean mask to track used indices
    used_mask = np.zeros(len(control_n_spots), dtype=bool)

    # Vectorized matching
    matched_indices = np.full(len(pattern_n_spots), -1)  # Placeholder for matched indices

    for i in range(len(indices)):  # Loop over pattern_adata rows
        for j in range(nbrs.n_neighbors):  # Loop over nearest neighbors
            idx = indices[i, j]  # Get candidate index
            if not used_mask[idx]:  # Check if it's available
                matched_indices[i] = idx  # Assign match
                used_mask[idx] = True  # Mark as used
                break  # Move to the next pattern row

    # Remove unmatched cases (if any)
    matched_indices = matched_indices[matched_indices != -1]

    # Subset control_adata based on matched indices
    matched_obs_indices = control_obs_indices[matched_indices]
    matched_adata = control_adata[control_adata.obs.index.isin(matched_obs_indices)]

    return matched_adata

def match_continuous_distribution(ref_adata, target_adata, random_seed: int = None):
    """
    Matches each row in ref_adata to the closest n_spots value in target_adata.

    Parameters:
    - ref_adata: Reference AnnData object (source of n_spots values).
    - target_adata: AnnData object to be resampled.
    - random_seed: Seed for reproducibility.

    Returns:
    - Matched target_adata with the same number of observations as ref_adata.
    """

    ref_n_spots = ref_adata.obs[['n_spots']].values  # Reference values
    target_n_spots = target_adata.obs[['n_spots']].values  # Target values

    # Fit nearest neighbor search on target
    nbrs = NearestNeighbors(n_neighbors=1).fit(target_n_spots)
    _, indices = nbrs.kneighbors(ref_n_spots)

    # Get the matched indices from target_adata
    matched_indices = target_adata.obs.iloc[indices.flatten()].index

    # Subset target_adata to match the reference
    matched_adata = target_adata[target_adata.obs.index.isin(matched_indices)]

    return matched_adata
