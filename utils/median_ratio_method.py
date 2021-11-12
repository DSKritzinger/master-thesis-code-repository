'''
The following code implements the median ratio method standardization technique from
Anders and Huber, 2010, for the standardization of RNA-seq count data.
'''
# Imports

import pandas as pd
import numpy as np

# geometric mean function

def geo_mean(x):
    '''
    Calculates geometric mean of input array.
    Input:
        x -  numpy array (shape - 1, n_features)
             of gene counts
    Output:
          - geometric mean of input array
    '''
    val = np.log(x)
    return np.exp(val.sum()/len(val))

# median ratio method standardization function

def median_ratio_standardization_(counts):
    '''
    Applies median ration standardization
    to gene counts dataset.
    Input:
        x - numpy array (shape - n_samples, n_features) containing gene 
            counts for each sample
    Output:
          - standardized gene counts in a numpy array
    '''
    counts_df = pd.DataFrame(counts)
    counts_np = np.array(counts)
    # calculate the geometric mean of all the gene counts to create the pseudo-reference sample
    pseudo_ref_samples = np.apply_along_axis(geo_mean, 0, counts_np)
    pseudo_ref_samples = pd.Series(pseudo_ref_samples)
    if np.all(pseudo_ref_samples == 0):
        return counts_np
    else:
        # remove genes with zero pseudo-reference sample counts
        pseudo_ref_samples = pseudo_ref_samples[pseudo_ref_samples != 0]
        _counts = counts_df.loc[:, pseudo_ref_samples.index.values]
        # calculate ratio of each sample to the reference
        count_ratios = np.divide(_counts, pseudo_ref_samples)
        # Calculate each samples normalization/standardization factor
        norm_factor = np.median(count_ratios, axis=1)
        # standardize counts
        standardized_counts = (counts_df.values.T/norm_factor).T
        return np.round(standardized_counts, 0)


def median_ratio_standardization_log(counts):
    '''
    Applies median ration standardization
    with post log normalization to gene 
    counts dataset.
    Input:
        x - numpy array (shape - n_samples, n_features) containing gene 
            counts for each sample
    Output:
          - standardized and log normalized gene counts in a numpy array
    '''
    counts_df = pd.DataFrame(counts)
    counts_np = np.array(counts)
    # calculate the geometric mean of all the gene counts to create the pseudo-reference sample
    pseudo_ref_samples = np.apply_along_axis(geo_mean, 0, counts_np)
    pseudo_ref_samples = pd.Series(pseudo_ref_samples)
    if np.all(pseudo_ref_samples == 0):
        return counts_np
    else:
        # remove genes with zero pseudo-reference sample counts
        pseudo_ref_samples = pseudo_ref_samples[pseudo_ref_samples != 0]
        _counts = counts_df.loc[:, pseudo_ref_samples.index.values]
        # calculate ratio of each sample to the reference
        count_ratios = np.divide(_counts, pseudo_ref_samples)
        # Calculate each samples normalization/standardization factor
        norm_factor = np.median(count_ratios, axis=1)
        # standardize counts
        standardized_counts = (counts_df.values.T/norm_factor).T
        standardized_counts_round = np.round(standardized_counts, 0)
        return np.log2(standardized_counts_round+1)
