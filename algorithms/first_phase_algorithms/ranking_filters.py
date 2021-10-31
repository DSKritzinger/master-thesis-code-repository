'''
The following code implements the ranker feature selection methods, namely
    - ReliefF
    - Chi-squared
    - mRMR
    - gini index
    - Fischer score
These algorithms are implemented through multiprocessing to reduce computation time.

This specific code is setup for the preprocessig of the real gc6-74 matched datasets.
'''
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import concurrent.futures
import time
import pickle
# Feature Selection methods
# Scikit feature
from skfeature.function.similarity_based import reliefF
from skfeature.function.statistical_based import chi_square
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import gini_index
# Scikit-learning
from sklearn.feature_selection import mutual_info_classif
# Standardization
from median_ratio_method import geo_mean, median_ratio_standardization

from skfeature.utility.mutual_information import su_calculation
################################################################################################
# %%
directory_pull = "C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Method Development/Developmental sets/"
filename = 'ge_raw_6'
# Import dataset
_data = pd.read_csv(directory_pull+filename+'.csv', sep=',')
_data
# Extract labels, sample id's and count data from imported data
labels = _data.loc[:, 'label']
# For GC6-74
sample_info = _data.loc[:, :"before_diagnosis_group"]  # First 8 columns are sample information
count_data = _data.loc[:, "7SK":]
############################################Import Data#########################################
# %%
# Initialize data for input into feature selection and classification
X_train = count_data.to_numpy()  # count matrix numpy array
y_categorical = labels.to_numpy().reshape(len(labels),)  # labels numpy array
# Change categorical labels to binary (controls - 0 and cases - 1)
Label_Encoder = LabelEncoder()
y_train = Label_Encoder.fit_transform(y_categorical)
############################################Split Data##########################################
# %%
# Thereafter for Validation: apply stratified K-fold data splits
num_splits = 10
num_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=0)

'''
Important to note the random_state of the train_test_split function as well as the random_state and splitting criteria of the RepeatedStratifiedKFold
function for future use.

These criteria are essentially the data splitting criteria.
'''
# initialize lists
kf_train_idxcs = []
kf_test_idxcs = []

for kf_train_index, kf_test_index in rskf.split(X_train, y_train):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)
kf_test_idxcs
# %%
################################################################################################
#                                  Feature Selection Main function
################################################################################################


def feature_selection(train_idx):
    # create train and test data folds
    X_train_f = X_train[train_idx]
    y_train_f = y_train[train_idx]
    # If DESeq Evaluation
    #X_train_f = np.round(median_ratio_standardization(X_train_f), 0)
    # If normalization is required
    #X_train_f = np.log2(X_train_f+1)
    # Ranker methods
    # Fisher-score
    score_fisher = fisher_score.fisher_score(X_train_f, y_train_f)
    # Chi-squared
    score_chi = chi_square.chi_square(X_train_f, y_train_f)
    # ReliefF
    score_reliefF = reliefF.reliefF(X_train_f, y_train_f)
    # mRMR 200
    idx_mrmr = MRMR.mrmr(X_train_f, y_train_f, n_selected_features=100)[0]
    # mim sklearn
    score_mim = mutual_info_classif(X_train_f, y_train_f)
    # Gini index
    score_gini = gini_index.gini_index(X_train_f, y_train_f)

    return score_fisher, score_chi, score_reliefF, score_mim, score_gini, idx_mrmr, train_idx


# %%
################################################################################################
#                                  Un-parallelization Main function
################################################################################################
'''
idx_reliefF_score_list = []
idx_chi_score_list = []
idx_fisher_score_list = []
idx_mim_score_list = []
idx_gini_score_list = []
idx_mrmr_list = []

start = time.perf_counter()
for train_dx in kf_train_idxcs:
    score_fisher, score_chi, score_reliefF, score_mim, score_gini = feature_selection(train_dx)
    # idx_mrmr_list.append(idx_mrmr)

    # score_fisher, score_chi, score_reliefF, score_mim, score_gini
    #idx_fisher_score_list.append(score_fisher)
    # idx_chi_score_list.append(score_chi)
    # idx_reliefF_score_list.append(score_reliefF)
    # idx_mim_score_list.append(score_mim)
    # idx_gini_score_list.append(score_gini)

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
'''
# %%
################################################################################################
#                                  Parallelization Main function
################################################################################################


def main():
    directory_push = "C:/Users/Daniel/Documents/Thesis/Python Code/xfilter outputsx/"
    # initialize
    i = 0

    # initializing empty score lists
    idx_reliefF_score_list = []
    idx_chi_score_list = []
    idx_fisher_score_list = []
    idx_mrmr_list = []
    idx_mim_score_list = []
    idx_gini_score_list = []

    train_idx_list = []

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        print(filename)
        results = executor.map(feature_selection, kf_train_idxcs)
        for result in results:
            score_fisher, score_chi, score_reliefF, score_mim, score_gini, idx_mrmr, train_idx = result

            i += 1
            print("This is fold: ", i, "of", (num_splits*num_repeats))
            # FS selected features lists
            # ranker scores
            idx_fisher_score_list.append(score_fisher)
            idx_chi_score_list.append(score_chi)
            idx_reliefF_score_list.append(score_reliefF)
            idx_mim_score_list.append(score_mim)
            idx_gini_score_list.append(score_gini)
            # ranker indexes
            idx_mrmr_list.append(idx_mrmr)

            train_idx_list.append(train_idx)

        finish = time.perf_counter()

        print(f'Finished in {round(finish-start, 2)} second(s)')

        # Pickle dump feature subset score and index lists
        with open(filename + '_mrm_rank_' + str(num_splits) + str(num_repeats), 'wb') as f:
            pickle.dump([
                idx_fisher_score_list,
                idx_chi_score_list,
                idx_reliefF_score_list,
                idx_mim_score_list,
                idx_gini_score_list,
                idx_mrmr_list,
                train_idx_list], f)


if __name__ == '__main__':
    main()
