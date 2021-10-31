'''
The following code implements the subset feature selection methods, namely
    - CFS
    - FCBF
These algorithms are implemented through multiprocessing to reduce computation time.

This specific code is setup for the preprocessig of the real gc6-74 matched datasets.
'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import concurrent.futures
import time
import pickle
# Feature Selection methods
from skfeature.function.statistical_based import CFS
from fcbf_func import fcbf
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
    # Subset methods
    # CFS
    idx_cfs = CFS.cfs(X_train_f, y_train_f)
    # FCBF
    idx_fcbf = fcbf(X_train_f, y_train_f)[0]

    return idx_cfs, idx_fcbf, train_idx

################################################################################################
#                                  Parallelization Main function
################################################################################################


def main():
    directory_push = "C:/Users/Daniel/Documents/Thesis/Python Code/xfilter outputsx/"

    # initialize
    i = 0

    # initializing empty score lists
    idx_cfs_list = []
    idx_fcbf_list = []

    train_idx_list = []

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        print(filename)
        results = executor.map(feature_selection, kf_train_idxcs)
        for result in results:
            idx_cfs, idx_fcbf, train_idx = result

            i += 1
            print("This is fold: ", i, "of", num_splits)
            # FS selected features lists
            # subset indexes
            idx_cfs_list.append(idx_cfs)
            idx_fcbf_list.append(idx_fcbf)

            train_idx_list.append(train_idx)

        # print(filename)
        # results = [executor.submit(feature_selection, kf_train_idx)
        #            for kf_train_idx in kf_train_idxcs]
        #
        # for f in concurrent.futures.as_completed(results):
        #     idx_cfs, idx_fcbf = f.result()
        #
        #     i += 1
        #     print("This is fold: ", i, "of", num_splits)
        #     # FS selected features lists
        #     # subset indexes
        #     idx_cfs_list.append(idx_cfs)
        #     idx_fcbf_list.append(idx_fcbf)

        finish = time.perf_counter()

        print(f'Finished in {round(finish-start, 2)} second(s)')

        # Pickle dump feature subset score and index lists
        with open(filename+'_mrm_subset_' + str(num_splits) + str(num_repeats), 'wb') as f:
            pickle.dump([idx_cfs_list, idx_fcbf_list, train_idx_list], f)


if __name__ == '__main__':
    main()
