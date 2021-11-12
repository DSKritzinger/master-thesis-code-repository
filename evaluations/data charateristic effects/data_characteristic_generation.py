'''
The following script implements the Boruta-RFE feature selection approach.
The implementation is specifically focussed on the generation of feature
sets for the different temporal sample groupings.

As the cross-validation procedure is computationally intensive, a multiprocessing 
approach was implemented for use on a high performance compute cluster (many core 
system for ideal performance).
'''
# imports
############################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
import time
import concurrent.futures
import pickle

# local imports
from boruta_rfe_fs import BorutaRFE

# %%
# import data
############################################################
filename = 'ge_raw_24'
_data = pd.read_csv(filename+'.csv', sep=',')

# %%
# prepare data
############################################################
labels = _data.loc[:, 'label']
sample_info = _data.loc[:, :"before_diagnosis_group"]  # First 8 columns are sample information
count_data = _data.loc[:, "7SK":]

X_train = count_data.to_numpy()

Label_Encoder = LabelEncoder()
y_categorical = labels.to_numpy().reshape(len(labels),)
y_train = np.abs(Label_Encoder.fit_transform(y_categorical) - 1)

# split data
num_splits = 10
num_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=0)

kf_train_idxcs = []
kf_test_idxcs = []
for kf_train_index, kf_test_index in rskf.split(X_train, y_train):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)


# split temporal training groups
# get temporal case
temp_case_indices = np.where(sample_info['before_diagnosis_group'].str.contains(
    '(0-6|6-12|12-18)'))[0]  # |6-12|12-18|18-24

# get controls
control_indices = np.where(sample_info['label'] == 'control')[0]
temp_control_indices = np.random.choice(control_indices, len(temp_case_indices)*3, replace=False)

# temporal train indices
temp_train_indices = np.concatenate([temp_case_indices, temp_control_indices])
# %%
# generation function
############################################################

def data_characteristic_generation(train_idx):
    # split temporal training group
    temp_idx = np.intersect1d(train_idx, temp_train_indices, return_indices=True)[0]
    # split train and test data
    X_train_f = X_train[temp_idx]
    y_train_f = y_train[temp_idx]
    # Boruta-RFE
    selector_output, final_feat, selected_feat, X_train_f = BorutaRFE(X_train_f, y_train_f)

    return selector_output, final_feat, selected_feat, X_train_f

# %%
# main loop
############################################################

def main():
    # initialize
    i = 0

    # initializing empty score lists
    selectors_list = []
    idx_list_1 = []
    idx_list_2 = []

    train_idx_list = []

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        print(filename)
        results = executor.map(data_characteristic_generation, kf_train_idxcs)
        for result in results:
            output, output_features_2, output_features_1, train_idx = result  # extract output

            i += 1
            print("This is fold: ", i, "of", (num_splits*num_repeats))
            # Stage 2 output and selected features

            selectors_list.append(output)
            idx_list_2.append(output_features_2)
            idx_list_1.append(output_features_1)
            train_idx_list.append(train_idx)

        finish = time.perf_counter()

        print(f'Finished in {round(finish-start, 2)} second(s)')

        # Pickle dump feature subset score and index lists
        with open(filename + '_BorutaRFE_data_characteristic_results_18', 'wb') as f:
            pickle.dump([
                selectors_list,
                idx_list_2,
                idx_list_1,
                train_idx_list], f)


if __name__ == '__main__':
    main()
