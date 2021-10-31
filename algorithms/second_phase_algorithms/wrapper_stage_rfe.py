'''
The following script creates a RFE wrapper feature selection algorithm and embeddeds it
into a cross-validation loop capable of being implemented with multiprocessing


This specific code is setup for the preprocessig of the real gc6-74 datasets.
'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold

import concurrent.futures
import time
import pickle
# Feature Selection methods
# Scikit-learning
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.similarity_based import reliefF
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.statistical_based import gini_index
from sklearn.feature_selection import RFECV, RFE
# Estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# Standardization
from utils.median_ratio_method import geo_mean, median_ratio_standardization, median_ratio_standardization_, median_ratio_standardization_log
# Scaling
from sklearn.preprocessing import StandardScaler
# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_curve, precision_recall_curve

from sklearn.preprocessing import FunctionTransformer

from skfeature.utility.mutual_information import su_calculation
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

############################################Import Data#########################################


# %%
filename = 'ge_raw_12'
# Import dataset
_data = pd.read_csv(filename+'.csv', sep=',')
_data
# Extract labels, sample id's and count data from imported data
labels = _data.loc[:, 'label']
# For GC6-74
sample_info = _data.loc[:, :"before_diagnosis_group"]  # First 8 columns are sample information
count_data = _data.loc[:, "7SK":]
sum(labels == "case")
################################################################################################
# Initialize data for evaluation
# %%
# Initialize data for input into feature selection and classification
X_train = count_data.to_numpy()  # count matrix numpy array
y_categorical = labels.to_numpy().reshape(len(labels),)  # labels numpy array
# Change categorical labels to binary (controls - 0 and cases - 1)
Label_Encoder = LabelEncoder()
y_train = np.abs(Label_Encoder.fit_transform(y_categorical) - 1)
y_train
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
#                                 Import Ensemble Selected Features
################################################################################################

# import filter ensemble output
with open(filename+'_filter_stage_105', 'rb') as f:
    filter_set_105 = pickle.load(f)
# extract scores and save to dict
fisher_score_list, chi_score_list, reliefF_score_list, mim_score_list, gini_score_list, _ = filter_set_105
fs_filter_set_scores = {
    'ReliefF': reliefF_score_list,
    'Chi-Square': chi_score_list,
    'Fisher-Score': fisher_score_list,
    'Info Gain': mim_score_list,
    'Gini Index': gini_score_list,
}
# Rank feature indices based on scores
ranked_filter_set = rank_rank_dict(fs_filter_set_scores)
# %%
# Select filter threshold
# -----------------
threshold_feats = 50
# -----------------
# %%
# Apply thresholding for ensemble
idx_fisher_list_th_e, idx_chi_list_th_e, idx_reliefF_list_th_e, idx_mim_list_th_e, idx_gini_list_th_e = rank_thres(
    ranked_filter_set, threshold_feats)

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

# %%
################################################################################################
#                                 Import Boruta Selected Features
################################################################################################
# boruta
# boruta_pickle_directory = 'D:/Thesis_to_big_file/xboruta outputsx/'
boruta_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/boruta_rfe_data_effects/'

# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 7, 0.01
with open(boruta_pickle_directory+'ge_raw_12_boruta_filter_stage_105_auto_7_001_250', 'rb') as f:
    boruta_out = pickle.load(f)


def extract_boruta_list(boruta_output):
    confirmed_list = []
    tentative_list = []
    selected_list = []
    for fold in range(0, 50):
        X_train_f = X_train[kf_train_idxcs[fold]]
        confirmed = pd.DataFrame(X_train_f).columns[boruta_output[0][fold].support_].to_list()
        confirmed_list.append(np.array(confirmed))
        tentative = pd.DataFrame(X_train_f).columns[boruta_output[0][fold].support_weak_].to_list()
        tentative_list.append(np.array(tentative))
        selected = confirmed.copy()
        selected.extend(tentative)
        selected_list.append(np.array(selected))
    return confirmed_list, tentative_list, selected_list


print('# of selected estimators: "auto"')
confirmed_list, tentative_list, selected_list = extract_boruta_list(boruta_out)


# %%
# Set estimators to evaluate
estimators = {
    'SVM_linear': LinearSVC(dual=False),
    'RF': RandomForestClassifier(n_jobs=1, class_weight='balanced', n_estimators=500)
}
# for analysis variables
# RFE number of selected features
num_features = 1
# estimator
estimator_name = 'RF'
# preprocessig
preprocessing = "mrm"  # mrm_log_log
# SMOTE
smote_var = 0  # 1 - yes, 0 - no
# Evaluation Measure
# Gemean


def gmean(y_true, y_predicted):
    sensitivity = recall_score(y_true, y_predicted)
    specificity = recall_score(y_true, y_predicted, pos_label=0)
    error = np.sqrt(sensitivity*specificity)
    return error


geometric_mean = make_scorer(gmean, greater_is_better=True)


eval_measure = geometric_mean  # "recall" (sensitivity), specificity, roc_auc

# %%
################################################################################################
#                                  Feature Selection Main function
################################################################################################
# Create Main Function Pipeline Options
mrstand = FunctionTransformer(median_ratio_standardization_)
mrstand_log = FunctionTransformer(median_ratio_standardization_log)
# build rfecv function with pipelines and test results in evaluations pp
# first test here
pipe_sse_input = [('standardizer', mrstand),
                  ('scaler', StandardScaler()),
                  ('estimator', estimators[estimator_name])]
pipe_slse_input = [('standardizer_log', mrstand_log),
                   ('scaler', StandardScaler()),
                   ('estimator', estimators[estimator_name])]
pipe_se_input = [('standardizer', mrstand),
                 ('estimator', estimators[estimator_name])]
pipe_sle_input = [('standardizer_log', mrstand_log),
                  ('estimator', estimators[estimator_name])]
# %%


class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_


# standardization, scaling, estimator
pipeline_sse = Mypipeline(pipe_sse_input)
# standardization, normalization (log), scaling, estimator
pipeline_slse = Mypipeline(pipe_slse_input)
# standardization, estimator
pipeline_se = Mypipeline(pipe_se_input)
# standardization, normalization (log), estimator
pipeline_sle = Mypipeline(pipe_sle_input)


def rfe_wrapper_stage(train_idx, selected_features):
    # create train and test data folds
    X_train_f = X_train[train_idx]
    y_train_f = y_train[train_idx]

    # Preprocessing (mrm, mrm_log_log)
    if preprocessing == "mrm" and estimator_name == 'SVM_linear':
        print("Applying: " + preprocessing + " and Scaling")

        #selector = RFECV(pipeline_sse, step=1, cv=5, scoring=eval_measure)
        selector = RFE(pipeline_sse, step=1, n_features_to_select=num_features)
    elif preprocessing == "mrm_log_log" and estimator_name == 'SVM_linear':
        print("Applying: " + preprocessing + " and Scaling")

        #selector = RFECV(pipeline_slse, step=1, cv=5, scoring=eval_measure)
        selector = RFE(pipeline_slse, step=1, n_features_to_select=num_features)
    elif preprocessing == "mrm" and estimator_name != 'SVM_linear':
        print("Applying: " + preprocessing + " only")

        #selector = RFECV(pipeline_se, step=1, cv=5, scoring=eval_measure)
        selector = RFE(pipeline_se, step=1, n_features_to_select=num_features)
    elif preprocessing == "mrm_log_log" and estimator_name != 'SVM_linear':
        print("Applying: " + preprocessing + " only")

        #selector = RFECV(pipeline_sle, step=1, cv=5, scoring=eval_measure)
        selector = RFE(pipeline_sle, step=1, n_features_to_select=num_features)

    # extract selected features from phase 1
    X_train_f_sel = X_train_f[:, selected_features]
    ''' note: you can replace this with the filter
    algorithm to turn this into the full hybrid model'''

    # fit RFECV
    selector_output = selector.fit(X_train_f_sel, y_train_f)

    # selected features
    selected_features_2 = selected_features[selector_output.support_]

    return selector_output, selected_features_2, train_idx


# %%
# previous stage selected features
# ---------------------------------
input_set = selected_list  # idx_ensemble_list
# ---------------------------------
################################################################################################
#                                 Unparallelization Main function
################################################################################################
# initialize lists
# selectors_list = []
# idx_rfe_list = []
#
# start = time.perf_counter()
#
# print(filename)
#
# for i in range(0, len(kf_train_idxcs)):
#
#     output, output_features = rfe_wrapper_stage(kf_train_idxcs[i])
#     idx_rfe = output_features
#
#     # save fold results in list
#     selectors_list.append(output)
#     idx_rfe_list.append(idx_rfe)
#
# finish = time.perf_counter()
#
# print(f'Finished in {round(finish-start, 2)} second(s)')
# %%
################################################################################################
#                                  Parallelization Main function
################################################################################################


def main():
    # initialize
    i = 0

    # initializing empty score lists
    selectors_list = []
    idx_rfe_list = []

    train_idx_list = []

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(filename)
        results = executor.map(rfe_wrapper_stage, kf_train_idxcs, input_set)
        for result in results:
            output, output_features, train_idx = result  # extract output

            i += 1
            print("This is fold: ", i, "of", (num_splits*num_repeats))
            # Stage 2 output and selected features

            selectors_list.append(output)
            idx_rfe_list.append(output_features)
            train_idx_list.append(train_idx)

        finish = time.perf_counter()

        print(f'Finished in {round(finish-start, 2)} second(s)')

        # Pickle dump feature subset score and index lists
        with open(filename + '_rfe_wrapper_stage_bor' + '_' + estimator_name + '_' + preprocessing + '_' + str(smote_var) + '_' + str(eval_measure) + '_' + str(num_features), 'wb') as f:
            pickle.dump([
                selectors_list,
                idx_rfe_list,
                train_idx_list], f)


if __name__ == '__main__':
    main()
