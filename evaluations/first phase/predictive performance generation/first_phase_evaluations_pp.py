'''
The following script evaluates the predictive performance of the individual
first phase algorithm's selected features.
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
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Boruta
from boruta_py import BorutaPy  # forked master boruta_py
# %%
################################################################################################
# Functions
# rank filter method output score indices
'''
input:  ranker_score_lists = ranker filter scores (order: fisher, chi, reliefF, mim, gini, mrmr, train indices)
ouptput: featuer indices ordered by score, ranker filter method indices (order: fisher, chi, reliefF, mim, gini, mrmr)
'''


def rank_rank(ranker_score_lists):
    # extract features from rank_rank() output
    #idx_fisher_score_list, idx_chi_score_list, idx_reliefF_score_list, idx_mim_score_list, idx_gini_score_list, idx_mrmr_list
    fisher_score_list, chi_score_list, reliefF_score_list, mim_score_list, gini_score_list, idx_mrmr_list, _ = ranker_score_lists
    # idx_fisher_score_list,idx_chi_score_list,idx_reliefF_score_list,idx_mim_score_list,idx_gini_score_list,idx_mrmr_list
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

    return idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_mim_list, idx_gini_list, idx_mrmr_list


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
y
# %%
################################################################################################
# CV procedure variables
################################################################################################
num_splits = 10  # number of folds
num_repeats = 5  # number of repeats
# %%
################################################################################################
#   Load filter method outputs
################################################################################################
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xfilter outputsx/'
# Pickle load feature subset lists RANKERS
with open(filter_pickle_directory+filename+'_rank_105', 'rb') as f:
    rank_105_625 = pickle.load(
        f)
with open(filter_pickle_directory+filename+'_mrm_rank_105', 'rb') as f:
    mrm_rank_105_625 = pickle.load(
        f)
with open(filter_pickle_directory+filename+'_mrm_log_rank_105', 'rb') as f:
    mrm_log_rank_105_625 = pickle.load(
        f)
# %%
# Pickle load feature subset lists SUBSET
with open(filter_pickle_directory+filename+'_subset_105', 'rb') as f:
    subset_105 = pickle.load(f)
with open(filter_pickle_directory+filename+'_mrm_subset_105', 'rb') as f:
    mrm_subset_105 = pickle.load(f)
with open(filter_pickle_directory+filename+'_mrm_log_subset_105', 'rb') as f:
    mrm_log_subset_105 = pickle.load(f)
# %%
# Pickle load feature subset lists BORUTA
with open(filename+'_boruta_filter_stage_105_1', 'rb') as f:
    boruta_1 = pickle.load(f)
# %%
################################################################################################
# Select preprocessing procedure to evaluate
'''##############################################Choose############################################'''
preproc = "mrm"  # "raw", "mrm", "mrm_log"
'''################################################################################################'''
if preproc == "raw":
    print("raw")
    pp_proc_rank = rank_105_625
    pp_proc_subset = subset_105
elif preproc == "mrm":
    print("mrm")
    pp_proc_rank = mrm_rank_105_625
    pp_proc_subset = mrm_subset_105
elif preproc == "mrm_log":
    print("mrm_log")
    pp_proc_rank = mrm_log_rank_105_625
    pp_proc_subset = mrm_log_subset_105
elif preproc == "mrm_log_log":
    print("mrm_log_log")
    pp_proc_rank = mrm_log_rank_105_625
    pp_proc_subset = mrm_log_subset_105


# idx_fisher_score_list,idx_chi_score_list,idx_reliefF_score_list,idx_mim_score_list,idx_gini_score_list,idx_mrmr_list, train_idx
idx_fisher_score_list, idx_chi_score_list, idx_reliefF_score_list, idx_mim_score_list, idx_gini_score_list, idx_mrmr_list, _ = pp_proc_rank
idx_cfs_list, idx_fcbf_list, _ = pp_proc_subset
len(pp_proc_rank)
len(pp_proc_subset)

# %%
################################################################################################
# Rank ranker methods selected features
################################################################################################
# rank feature index lists

# rank-rank output: idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_mim_list, idx_gini_list, idx_mrmr_list
ranked_filters = rank_rank(pp_proc_rank)

# %%
################################################################################################
# Threshold ranker methods selected features
'''##############################################Choose############################################'''
threshold_feats = 4000
'''################################################################################################'''
# selecting x number of features from ranked methods
# rank_thres output: idx_fisher_list_th, idx_chi_list_th, idx_reliefF_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th
idx_fisher_list_th, idx_chi_list_th, idx_reliefF_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    ranked_filters, threshold_feats)
len(idx_fisher_list_th[0])
# %%
################################################################################################
# -------------------------------------------Evaluate-------------------------------------------
################################################################################################
#
######################################Predictive Performance######################################
# Initialize variables
X_train = X
y_train = y
repeats_string = str(num_repeats)
splits_string = str(num_splits)
# Evaluate dataset
len(X_train)
len(y_train)
sum(y_train == 0)
sum(y_train == 1)
# %%
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

start = time.perf_counter()
# Evaluate Predictive Performance
# All features predictive performance evaluation

# Extract already computed all selectd feature results
if preproc == "raw":
    print("raw")
    with open(filename+"_raw_105_predperf_th125", "rb") as f:  # if no smote + _nsm
        pa = pickle.load(f)
    all_effectiveness = pa["All"]
elif preproc == "mrm":
    print("mrm")
    with open(filename+"_mrm_105_predperf_th125", "rb") as f:  # if no smote + _nsm
        pa = pickle.load(f)
    all_effectiveness = pa["All"]
elif preproc == "mrm_log":
    print("mrm_log")
    with open(filename+"_mrm_log_105_predperf_th125", "rb") as f:
        pa = pickle.load(f)
    all_effectiveness = pa["All"]
elif preproc == "mrm_log_log":
    print("mrm_log_log")
    with open(filename+"_mrm_log_log_105_predperf_th125", "rb") as f:
        pa = pickle.load(f)
    all_effectiveness = pa["All"]
else:
    sys.exit()

# %%
# Ranker selected feature
print('mRMR predictive ability')
mrmr_effectiveness = predictive_ability(
    classifiers, idx_mrmr_list_th, X_train, y_train, num_repeats, num_splits, preproc)
print('ReliefF predictive ability')
reliefF_effectiveness = predictive_ability(
    classifiers, idx_reliefF_list_th, X_train, y_train, num_repeats, num_splits, preproc)
print('Chi-square predictive ability')
chi_effectiveness = predictive_ability(
    classifiers, idx_chi_list_th, X_train, y_train, num_repeats, num_splits, preproc)
print('Fisher score predictive ability')
fisher_effectiveness = predictive_ability(
    classifiers, idx_fisher_list_th, X_train, y_train, num_repeats, num_splits, preproc)
print('Information Gain predictive ability')
mim_effectiveness = predictive_ability(
    classifiers, idx_mim_list_th, X_train, y_train, num_repeats, num_splits, preproc)
print('Gini Index predictive ability')
gini_effectiveness = predictive_ability(
    classifiers, idx_gini_list_th, X_train, y_train, num_repeats, num_splits, preproc)
################################################################################################
# %%
# Subset selected features
print('CFS predictive ability')
cfs_effectiveness = predictive_ability(
    classifiers, idx_cfs_list, X_train, y_train, num_repeats, num_splits, preproc)
print('FCBF predictive ability')
fcbf_effectiveness = predictive_ability(
    classifiers, idx_fcbf_list, X_train, y_train, num_repeats, num_splits, preproc)
print("Effectiveness Evaluation Completed")
# %%
################################################################################################
# Compile filter predictive performance results
fs_pa = {
    'mRMR': mrmr_effectiveness,
    'ReliefF': reliefF_effectiveness,
    'Chi-squared': chi_effectiveness,
    'Fisher-Score': fisher_effectiveness,
    'Information Gain': mim_effectiveness,
    'Gini Index': gini_effectiveness,
    'CFS': cfs_effectiveness,
    'FCBF': fcbf_effectiveness,
    'All': all_effectiveness
}
finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
# %%
# Save filter predictive performance results
repeats_string = str(num_repeats)
splits_string = str(num_splits)
threshold_string = str(threshold_feats)
with open(filename + '_' + preproc + '_' + splits_string + repeats_string + '_predperf_th' + threshold_string, 'wb') as f:
    pickle.dump(fs_pa, f)
