'''
The following script implements the all-relevant feature selection approach, Boruta.
The implementation is specifically focussed on the generation of feature sets for
the hybrid method developmental procedure (10 fold x 5 cross-validation).

As the cross-validation procedure is computationally intensive, a multiprocessing 
approach was implemented for use on a high performance compute cluster (many core 
system for ideal performance).
'''
# imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold

import concurrent.futures
import time
import pickle

# Boruta
from boruta_py import BorutaPy  # forked master boruta_py
from sklearn.ensemble import RandomForestClassifier
# Standardization
from median_ratio_method import  median_ratio_standardization_
############################################Import Data#########################################
# %%
directory = "C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Method Development/Developmental sets/"
filename = 'ge_raw_6'
# Import dataset
_data = pd.read_csv(directory+filename+'.csv', sep=',')
_data
# Extract labels, sample id's and count data from imported data
labels = _data.loc[:, 'label']
# For GC6-74
sample_info = _data.loc[:, :"before_diagnosis_group"]  # First 8 columns are sample information
count_data = _data.loc[:, "7SK":]
sum(labels == "case")
################################################################################################
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
kf_test_idxcs[49]
# %%
################################################################################################
#                                  Feature Selection Main function
################################################################################################


def boruta_all_relevance(train_idx):
    # create train and test data folds
    X_train_f = X_train[train_idx]
    y_train_f = y_train[train_idx]
    # DESeq
    X_train_f = np.round(median_ratio_standardization_(X_train_f), 0)
    # X_train_f = np.log2(X_train_f+1)
    # Define Boruta
    # Random Forest
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    # Boruta
    boruta = BorutaPy(rf, n_estimators=5000, verbose=2, random_state=1, max_iter=250)
    # Fit boruta
    boruta.fit(X_train_f, y_train_f)

    return boruta, train_idx

# %%
################################################################################################
#                                  Parallelization Main function
################################################################################################


def main():
    # initialize
    i = 0

    # initializing empty score lists
    boruta_list = []

    train_idx_list = []

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        print(filename)
        results = executor.map(boruta_all_relevance, kf_train_idxcs)
        for result in results:
            #  score_fisher, score_chi, score_reliefF, score_mim, score_gini, train_idx
            boruta_fitted, train_idx = result

            i += 1
            print("This is fold: ", i, "of", (num_splits*num_repeats))
            # FS selected features lists
            boruta_list.append(boruta_fitted)

            train_idx_list.append(train_idx)

        finish = time.perf_counter()

        print(f'Finished in {round(finish-start, 2)} second(s)')

        # Pickle dump feature subset score and index lists
        with open(filename + '_boruta_filter_stage_mrm_log_' + str(num_splits) + str(num_repeats), 'wb') as f:
            pickle.dump([
                boruta_list,
                train_idx_list], f)


if __name__ == '__main__':
    main()

