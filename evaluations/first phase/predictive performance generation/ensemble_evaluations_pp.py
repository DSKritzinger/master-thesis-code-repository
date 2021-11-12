'''
The following script evaluates the predictive performance of the ensembled
filter algorithm's selected features.
'''
# %%
# Imports
# Basics
import pandas as pd
import numpy as np
import pickle
import time
import sys
# Evaluation functions
from eval_functions import predictive_ability
from sklearn.metrics import auc
# Data Prep functions
from sklearn.preprocessing import LabelEncoder
# Feature Selection Methods
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.similarity_based import reliefF
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.statistical_based import gini_index
# Machine Learning Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# %%
################################################################################################
# Functions
# rank filter method output score indices
'''
input:  ranker_score_lists = ranker filter scores (order: fisher, chi, reliefF, mim, gini) and fold sample indices
ouptput: ordered by score, ranker filter method indices (order: fisher, chi, reliefF, mim, gini)
'''
def rank_rank_dict(ranker_score_dict):
    # extract features from rank_rank() output
    fisher_score_list = ranker_score_dict['Fisher-Score']
    chi_score_list = ranker_score_dict['Chi-Square']
    reliefF_score_list = ranker_score_dict['ReliefF']
    mim_score_list = ranker_score_dict['Info Gain']
    gini_score_list = ranker_score_dict['Gini Index']
    # initialize feature output variables
    idx_fisher_list = []
    idx_reliefF_list = []
    idx_chi_list = []
    idx_mim_list = []
    idx_gini_list = []

    # ranker scores -> sorted feature indexes
    for i in range(0, len(fisher_score_list)):
        # Fisher-score
        idx_fisher = fisher_score.feature_ranking(fisher_score_list[i])
        idx_fisher_list.append(idx_fisher)
        # Chi-square
        idx_chi = chi_square.feature_ranking(chi_score_list[i])
        idx_chi_list.append(idx_chi)
        # ReliefF
        idx_reliefF = reliefF.feature_ranking(reliefF_score_list[i])
        idx_reliefF_list.append(idx_reliefF)
        # Gini
        idx_gini = gini_index.feature_ranking(gini_score_list[i])
        idx_gini_list.append(idx_gini)
        # MIM
        idx_mim = np.argsort(mim_score_list[i])[::-1]
        idx_mim_list.append(idx_mim)

    return idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_mim_list, idx_gini_list


# apply threshold to ranker method outputs ('top-k')
'''
input:  ranked_ranker_lists = ranked ranker filter indices (order: fisher, chi, reliefF, mim, gini, mrmr)
        treshold = # of genes to select
ouptput: 'top-k' ranker filter method indices (order: fisher, chi, reliefF, mim, gini, mrmr)
'''


def rank_thres(ranked_ranker_lists, threshold):
    list_th_out = []
    for list in ranked_ranker_lists:
        list_th = [item[0:threshold] for item in list]
        list_th_out.append(list_th)
    return list_th_out


# %%
################################################################################################
# Import dataset
'''##############################################Choose############################################'''
filename = 'ge_raw_6'
'''################################################################################################'''
directory = "C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Method Development/Developmental sets/"
# Import dataset
_data = pd.read_csv(directory+filename+'.csv', sep=',')

# Extract labels, sample id's and count data from imported data
labels = _data.loc[:, 'label']
labels
# For GC6-74
sample_info = _data.loc[:, :"before_diagnosis_group"]  # First 8 columns are sample information
count_data = _data.loc[:, "7SK":]
################################################################################################
# %%
# Initialize data for input into feature selection and classification
X = count_data.to_numpy()  # count matrix numpy array
y_categorical = labels.to_numpy().reshape(len(labels),)  # labels numpy array
# Change categorical labels to binary (controls - 0 and cases - 1)
Label_Encoder = LabelEncoder()
y = np.abs(Label_Encoder.fit_transform(y_categorical) - 1)
sum(y == 1)
# %%
################################################################################################
# CV procedure variables
################################################################################################
num_splits = 10  # number of folds
num_repeats = 5  # number of repeats
# %%
################################################################################################
# Ensemble Predictive Performance
################################################################################################
# ----------------- Variables -----------------
# Initialize variables
X_train = X
y_train = y
repeats_string = str(num_repeats)
splits_string = str(num_splits)
# Initialize classifiers to be used
classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=1),
    'SVM_linear': LinearSVC(dual=False),
    'SVM_rbf': SVC(kernel="rbf"),
    'GaussianNB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=1),
    'XGBoost': XGBClassifier(n_jobs=1)
}
# %%
# ----------------- Select ensemble components-----------------
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xfilter outputsx/'
# select dataset to be evaluated

if filename == "ge_raw_12":
    print("12")
    with open(filter_pickle_directory+filename+'_filter_stage_105', 'rb') as f:
        filter_set_105 = pickle.load(
            f)
elif filename == "ge_raw_18":
    print("18")
    with open(filter_pickle_directory+filename+'_filter_stage_105', 'rb') as f:
        filter_set_105 = pickle.load(
            f)
elif filename == "ge_raw_24":
    print("24")
    with open(filter_pickle_directory+filename+'_filter_stage_105', 'rb') as f:
        filter_set_105 = pickle.load(
            f)
elif filename == "ge_raw_6":
    print("6")
    with open(filter_pickle_directory+filename+'_filter_stage_105', 'rb') as f:
        filter_set_105 = pickle.load(
            f)

# %%
# extract scores and save to dict
fisher_score_list, chi_score_list, reliefF_score_list, mim_score_list, gini_score_list, _ = filter_set_105
fs_filter_set_scores = {
    'ReliefF': reliefF_score_list,
    'Chi-Square': chi_score_list,
    'Fisher-Score': fisher_score_list,
    'Info Gain': mim_score_list,
    'Gini Index': gini_score_list,
}

# %%
# ----------------- Rank all filters -----------------
ranked_filter_set = rank_rank_dict(fs_filter_set_scores)

# %%
# ----------------- Create Ensemble -----------------
# preprocessing procedure
preproc = "ens"  # mrm_log_log, ens
# -----------------
# %%
ensemble_thresholds = [5, 10, 25, 50, 125, 250]
for threshold in ensemble_thresholds:
    print("Now making: " + str(threshold))
    ensemble_threshold = threshold  # caps the number of features of each algorithm to put into ensemble

    # Apply thresholding for ensemble
    idx_fisher_list_th_e, idx_chi_list_th_e, idx_reliefF_list_th_e, idx_mim_list_th_e, idx_gini_list_th_e = rank_thres(
        ranked_filter_set, ensemble_threshold)

    # Initialize feature list
    idx_ensemble_list = []
    # append features from different methods together
    for i in range(0, (num_repeats*num_splits)):
        ensembled_features = np.append(idx_fisher_list_th_e[i], [
                                       idx_chi_list_th_e[i], idx_reliefF_list_th_e[i], idx_mim_list_th_e[i], idx_gini_list_th_e[i]])
        ensembled_features
        # remove features which are duplicated in the list
        ensembled_features = np.array(list(dict.fromkeys(ensembled_features)))
        # make list of every folds selected features
        idx_ensemble_list.append(ensembled_features)

    if ensemble_threshold == 5:
        idx_ensemble_list_5 = idx_ensemble_list
    elif ensemble_threshold == 10:
        idx_ensemble_list_10 = idx_ensemble_list
    elif ensemble_threshold == 25:
        idx_ensemble_list_25 = idx_ensemble_list
    elif ensemble_threshold == 50:
        idx_ensemble_list_50 = idx_ensemble_list
    elif ensemble_threshold == 125:
        idx_ensemble_list_125 = idx_ensemble_list
    elif ensemble_threshold == 250:
        idx_ensemble_list_250 = idx_ensemble_list

# %%
start = time.perf_counter()
# ----------------- Evaluate Ensemble -----------------
print(filename)
print('ensemble predictive ability with 5 features')
ensemble_effectiveness_5 = predictive_ability(
    classifiers, idx_ensemble_list_5, X_train, y_train, num_repeats, num_splits, preproc)
print('ensemble predictive ability with 10 features')
ensemble_effectiveness_10 = predictive_ability(
    classifiers, idx_ensemble_list_10, X_train, y_train, num_repeats, num_splits, preproc)
print('ensemble predictive ability with 25 features')
ensemble_effectiveness_25 = predictive_ability(
    classifiers, idx_ensemble_list_25, X_train, y_train, num_repeats, num_splits, preproc)
print('ensemble predictive ability with 50 features')
ensemble_effectiveness_50 = predictive_ability(
    classifiers, idx_ensemble_list_50, X_train, y_train, num_repeats, num_splits, preproc)
print('ensemble predictive ability with 125 features')
ensemble_effectiveness_125 = predictive_ability(
    classifiers, idx_ensemble_list_125, X_train, y_train, num_repeats, num_splits, preproc)
print('ensemble predictive ability with 250 features')
ensemble_effectiveness_250 = predictive_ability(
    classifiers, idx_ensemble_list_250, X_train, y_train, num_repeats, num_splits, preproc)
# %%
# Compile filter predictive performance results
ensemble_pa = {
    'ens_5': ensemble_effectiveness_5,
    'ens_10': ensemble_effectiveness_10,
    'ens_25': ensemble_effectiveness_25,
    'ens_50': ensemble_effectiveness_50,
    'ens_125': ensemble_effectiveness_125,
    'ens_250': ensemble_effectiveness_250
}

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
# %%
# Save predictive performance results
repeats_string = str(num_repeats)
splits_string = str(num_splits)
with open(filename + '_' + preproc + '_' + splits_string + repeats_string + '_predperf_ensemble', 'wb') as f:
    pickle.dump(ensemble_pa, f)
