'''
The following code aims to provide the evaluation results of the second phase (wrapper algorithms) of the feature selection process.

Specifically for the wrapper stage development process.

This code evaluates the wrapper algorithms in terms of their:
- predictive performance: in terms of sensitivity and specificity
- stability: ATI
- feature set cardinality
- computation time
'''
# %%
# Imports
# Basics
from eval_functions import intersystem_ATI, average_tanimoto_index, tanimoto_index
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle
import time
import sys
# Evaluation functions
from eval_functions import predictive_ability
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
from kneed import KneeLocator
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
# Graphing
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
# %%
################################################################################################
# classes


class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_
# Functions


'''
- boxplot_filter
input:  fs_pa =  feature selection methods predictive ability outputs
        classifiers = dictionary of classifiers to be used
        data_name = results to be graphed (accuracy, sensitivity, specificity, auc)
        figure_size = size of figure to be output
ouptput: boxplot of relevant filters
'''


def boxplot_filter(fs_pa, classifiers_, data_name, sel_classifiers, axis, ordering=None):
    # fs_pa = results
    # classifiers_ = classifiers
    # data_name = "Geomean"
    # sel_classifiers = selected_classifiers
    # axis = ax2
    # initialize empty dataframes for input to graph
    classifier_names = list(classifiers_.keys())
    all_filter = pd.DataFrame(columns=classifier_names)
    all_filter_sens = pd.DataFrame(columns=classifier_names)
    all_filter_spec = pd.DataFrame(columns=classifier_names)
    # extract
    for pa_keys, pa_values in fs_pa.items():
        # accuracy
        if data_name == "Accuracy":
            ind_filter = pd.DataFrame(
                pa_values[1][:-2], columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)
        elif data_name == "Sensitivity":
            # sensitivity
            ind_filter = pd.DataFrame(
                pa_values[2][:-2], columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)
        elif data_name == "Specificity":
            # specificity
            ind_filter = pd.DataFrame(
                pa_values[3][:-2], columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)
        elif data_name == "Geomean":
            # sensitivity
            ind_filter_sens = pd.DataFrame(
                pa_values[2][:-2], columns=classifier_names).assign(Filters=pa_keys)
            all_filter_sens = all_filter_sens.append(ind_filter_sens, ignore_index=True)
            # specificity
            ind_filter_spec = pd.DataFrame(
                pa_values[3][:-2], columns=classifier_names).assign(Filters=pa_keys)
            all_filter_spec = all_filter_spec.append(ind_filter_spec, ignore_index=True)

        elif data_name == "AUC":
            # fpr
            fpr_list = pa_values[-2]
            # tpr
            tpr_list = pa_values[-1]
            # extract auc for all Classifiers
            auc_clf_list = auc_clf_compiler(classifiers_, fpr_list, tpr_list)
            ind_filter = pd.DataFrame(
                auc_clf_list, columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)

    if data_name == "Geomean":
        all_filter = pd.DataFrame(np.sqrt(
            all_filter_sens.iloc[:, 0:all_filter_sens.shape[1]-1]*all_filter_spec.iloc[:, 0:all_filter_spec.shape[1]-1]))
        all_filter["Filters"] = all_filter_spec.iloc[:, all_filter_spec.shape[1]-1]
        all_filter

    # select classifiers to be displayed
    if "Filters" not in sel_classifiers:
        sel_classifiers.append("Filters")
    sel_filter = all_filter[sel_classifiers]
    sel_classifiers.pop()
    # melt all_filter dataframe for input to graph
    all_filter_m = sel_filter.melt(
        id_vars=['Filters'], var_name='Classifiers', value_name=data_name)
    all_filter_m
    # order of output x-axis variables
    # if any(all_filter_m['Filters'].str.contains('Boruta')):
    #     if any(all_filter_m['Filters'].str.contains('ens_')):
    #     ordering = ['mRMR', 'ReliefF', 'Chi-Square', 'Fisher-Score',
    #                 'Info Gain', 'Gini Index', 'CFS', 'FCBF', 'Boruta', 'All']
    # else:
    #     ordering = None
    ax = sns.boxplot(ax=axis, x=all_filter_m['Filters'], y=all_filter_m[data_name],
                     hue=all_filter_m['Classifiers'], order=ordering, fliersize=2, linewidth=0.8, showfliers=False)
    sns.despine()
    ax.set(ylim=(-0.05, 1.04))
    ax.grid(axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    # Put the legend out of the figure
    # ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    return ax


def auc_clf_compiler(classifiers_, fpr_list, tpr_list):
    # initialize graph variables
    auc_clf_list = []
    for clf_key, clf in classifiers_.items():

        fpr_list_df = pd.DataFrame(fpr_list, columns=classifiers_.keys())[clf_key]
        tpr_list_df = pd.DataFrame(tpr_list, columns=classifiers_.keys())[clf_key]
        # initialize classifier graph input variables
        auc_list = []

        for fold in range(0, (5*10)):
            # extract fold specific false positive rate and true positive rate
            fpr = fpr_list_df[fold]
            tpr = tpr_list_df[fold]

            # AUC
            roc_auc = auc(fpr, tpr)
            # compile auc of each fold
            auc_list.append(roc_auc)
        # compile auc_list of each classifier
        auc_clf_list.append(auc_list)
        auc_clf_list_t = np.array(auc_clf_list).T.tolist()

    return auc_clf_list_t

# Gemean


def gmean(y_true, y_predicted):
    sensitivity = recall_score(y_true, y_predicted)
    specificity = recall_score(y_true, y_predicted, pos_label=0)
    error = np.sqrt(sensitivity*specificity)
    return error


geometric_mean = make_scorer(gmean, greater_is_better=True)

'''
- plot_classifier_nfold_rocs

input:  classifiers = dictionary of classifiers to be used
        fpr_list = list of fpr's from predictive performance function of a specific filter method
        tprs_list = list of tpr's from predictive performance function of a specific filter method
ouptput: ROC Curve for said filter run on all classifiers
'''


def plot_classifier_nfold_rocs(classifiers_, fpr_list, tprs_list, axis):
    # initialize graph variables
    mean_tprs_list = []
    mean_fprs_list = []
    mean_auc_list = []
    mean_fpr = np.linspace(0, 1, 100)

    ax = axis

    for clf_key, clf in classifiers_.items():

        fpr_list_df = pd.DataFrame(fpr_list, columns=classifiers_.keys())[clf_key]
        tpr_list_df = pd.DataFrame(tprs_list, columns=classifiers_.keys())[clf_key]
        # initialize classifier graph input variables
        auc_clfs_list = []
        tprs_clfs_list = []

        for fold in range(0, (num_repeats*num_splits)):
            # extract fold specific false positive rate and true positive rate
            fpr = fpr_list_df[fold]
            tpr = tpr_list_df[fold]

            tprs_clfs_list.append(np.interp(mean_fpr, fpr, tpr))
            tprs_clfs_list[-1][0] = 0.0
            # AUC
            roc_auc = auc(fpr, tpr)
            auc_clfs_list.append(roc_auc)

        # mean precision
        mean_tpr = np.mean(tprs_clfs_list, axis=0)
        mean_tpr[-1] = 1
        # mean auc
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(auc_clfs_list)
        # classifiers output lists
        mean_tprs_list.append(mean_tpr)
        mean_fprs_list.append(mean_fpr)
        mean_auc_list.append(mean_auc)

        # plot ROC curves
        ax.plot(mean_fpr, mean_tpr, lw=2, alpha=0.8,
                label=clf_key + ' (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
        ax.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
        ax.set_xlabel('1-Specificity')
        ax.set_ylabel('Sensitivity')
        ax.legend(loc="lower right")
        sns.despine()
    return ax


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
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsx/'
# Pickle load wrapper outputs

with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_SVM_linear_mrm_0_f1', 'rb') as f:
    wrap_mrm_svm_050_f1 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_SVM_linear_mrm_0_roc_auc', 'rb') as f:
    wrap_mrm_svm_050_auc = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_SVM_linear_mrm_0_gmean', 'rb') as f:
    wrap_mrm_svm_050_gmean = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_f1', 'rb') as f:
    wrap_mrm_RF_050_f1 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_roc_auc', 'rb') as f:
    wrap_mrm_RF_050_auc = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_gmean', 'rb') as f:
    wrap_mrm_RF_050_gmean = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_SVM_linear_mrm_log_log_0_gmean', 'rb') as f:
    wrap_mrm_ll_svm_050_gmean = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_log_log_0_gmean', 'rb') as f:
    wrap_mrm_ll_RF_050_gmean = pickle.load(
        f)
# %%
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsx/New/'
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_gmean', 'rb') as f:
    wrap_mrm_RF_050bal_gmean = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_ens_25_RF_mrm_0_gmean', 'rb') as f:
    wrap_mrm_RF_025bal_gmean = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_ens_25_SVM_linear_mrm_0_gmean', 'rb') as f:
    wrap_mrm_svm_025_gmean = pickle.load(
        f)

with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_boruta_RF_mrm_0_gmean', 'rb') as f:
    wrap_boruta_mrm_RF_050bal_gmean = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_boruta_SVM_linear_mrm_0_gmean', 'rb') as f:
    wrap_boruta_mrm_svm_050_gmean = pickle.load(
        f)
# %%
################################################################################################
# Select preprocessing procedure to evaluate
'''##############################################Choose############################################'''
preproc = "mrm"  # "mrm", "mrm_log"
'''################################################################################################'''
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
# ---------------
'''Effect of evaluation measure'''
# ---------------
#       SVM
# # Features
# ------------------
input1 = wrap_mrm_svm_050_f1
input2 = wrap_mrm_svm_050_auc
input3 = wrap_mrm_svm_050_gmean
print("Number of selected features:")
print("F1 Score: " + str(np.mean([n.n_features_ for n in input1[0]])
                         ) + " + " + str(np.std([n.n_features_ for n in input1[0]])))
print("AUC Score: " + str(np.mean([n.n_features_ for n in input2[0]])
                          ) + " + " + str(np.std([n.n_features_ for n in input2[0]])))
print("Geometric Mean Score: " +
      str(np.mean([n.n_features_ for n in input3[0]])) + " + " + str(np.std([n.n_features_ for n in input3[0]])))
# %%
# RFECV GridScore Outputs
# ------------------
input = input1[0]  # f1 score
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C1")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
# %%
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
np.mean(top_score_list_1)
# %%
np.std(top_score_list_1)
# %%
input = input2[0]  # auc score
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C1")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
# %%
top_score_list_2 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_2.append(top_score)
np.mean(top_score_list_2)
# %%
np.std(top_score_list_2)
# %%
input = input3[0]  # gmean score
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C1")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
# %%
top_score_list_3 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_3.append(top_score)
np.mean(top_score_list_3)
# %%
np.std(top_score_list_3)
# %%
print("Top score direct comparison")
print("Mean F1:" + str(np.mean(top_score_list_1)) + " @ " + str(np.std(top_score_list_1)))
print("Mean AUC:" + str(np.mean(top_score_list_2)) + " @ " + str(np.std(top_score_list_2)))
print("Mean GMean:" + str(np.mean(top_score_list_3)) + " @ " + str(np.std(top_score_list_3)))
# %%
# Outer CV loop Predictive Performance
# ------------------
# SVM
# ------------------
svm_mrm_nsm_f1 = predictive_ability(
    classifiers, input1[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_mrm_nsm_auc = predictive_ability(
    classifiers, input2[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_mrm_nsm_gmean = predictive_ability(
    classifiers, input3[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = svm_mrm_nsm_auc
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = svm_mrm_nsm_f1
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean F1", "Std F1", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = svm_mrm_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean GMean", "Std GMean", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
results = {
    "svm_f1": svm_mrm_nsm_f1,
    "svm_auc": svm_mrm_nsm_auc,
    "svm_gmean": svm_mrm_nsm_gmean
}
# %%
'''
From these results it is evident that the AUC generated results are less reliable in
the feature selection process leading to
'''
# %%


def set_style():
    sns.set(context="paper", font='serif', style="white", rc={"xtick.bottom": True,
                                                              "xtick.labelsize": "x-small",
                                                              "ytick.left": True,
                                                              "ytick.labelsize": "x-small",
                                                              "legend.fontsize": "x-small",
                                                              "ytick.major.size": 2,
                                                              "xtick.major.size": 2})


fig_width = 5.52
fig_height_scale = 1.2
selected_classifiers = ['KNN', 'SVM_linear', 'SVM_rbf', 'GaussianNB', 'RF']
# set golden ratio values
gr = (np.sqrt(5)-1)/2
# %%
fig_height_scale = 1.8
set_style()
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(results, classifiers, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(results, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(results, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.135)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_raw_nsm.png",bbox_inches="tight", dpi=1000)

# %%
#       RF
# Features
# ------------------
input1 = wrap_mrm_RF_050_f1
input2 = wrap_mrm_RF_050_auc
input3 = wrap_mrm_RF_050_gmean
print("Number of features selected:")
print("F1 Score: " + str(np.mean([n.n_features_ for n in input1[0]])
                         ) + " + " + str(np.std([n.n_features_ for n in input1[0]])))
print("AUC Score: " + str(np.mean([n.n_features_ for n in input2[0]])
                          ) + " + " + str(np.std([n.n_features_ for n in input2[0]])))
print("Geometric Mean Score: " +
      str(np.mean([n.n_features_ for n in input3[0]])) + " + " + str(np.std([n.n_features_ for n in input3[0]])))

# %%
# RFECV GridScore Outputs
# ------------------
input = input1[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C1")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
# %%
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
np.mean(top_score_list_1)
# %%
input = input2[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C1")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
# %%
top_score_list_2 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_2.append(top_score)
np.mean(top_score_list_2)
# %%
input = input3[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C1")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
# %%
top_score_list_3 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_3.append(top_score)
np.mean(top_score_list_3)
# %%
print("Top score direct comparison")
print("Mean F1:" + str(np.mean(top_score_list_1)) + " @ " + str(np.std(top_score_list_1)))
print("Mean AUC:" + str(np.mean(top_score_list_2)) + " @ " + str(np.std(top_score_list_2)))
print("Mean GMean:" + str(np.mean(top_score_list_3)) + " @ " + str(np.std(top_score_list_3)))
# %%
# Outer CV loop Predictive Performance
# ------------------
# RF
# ------------------
rf_mrm_nsm_f1 = predictive_ability(
    classifiers, input1[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_nsm_auc = predictive_ability(
    classifiers, input2[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_nsm_gmean = predictive_ability(
    classifiers, input3[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = rf_mrm_nsm_auc
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = rf_mrm_nsm_f1
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = rf_mrm_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
results = {
    "rf_f1": rf_mrm_nsm_f1,
    "rf_auc": rf_mrm_nsm_auc,
    "rf_gmean": rf_mrm_nsm_gmean
}
# %%
fig_height_scale = 1.8
set_style()
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(results, classifiers, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(results, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(results, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.135)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_raw_nsm.png",bbox_inches="tight", dpi=1000)
# %%
# ---------------
'''Effect of normalization (log)'''
# ---------------
# # Features
# ------------------
input1 = wrap_mrm_ll_svm_050_gmean
input2 = wrap_mrm_ll_RF_050_gmean
print('Number of selected features:')
print(str(np.mean([n.n_features_ for n in input1[0]])) +
      " + " + str(np.std([n.n_features_ for n in input1[0]])))
print(str(np.mean([n.n_features_ for n in input2[0]])) +
      " + " + str(np.std([n.n_features_ for n in input2[0]])))
# %%
# RFECV GridScore Outputs
# ------------------
input = input1[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C1")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
# %%
top_score_list = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list.append(top_score)
np.mean(top_score_list)
# %%
input = input2[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]

fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C1")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
# %%
top_score_list = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list.append(top_score)
np.mean(top_score_list)
# %%
# Outer CV loop Predictive Performance
# ------------------
svm_mrm_ll_nsm_f1 = predictive_ability(
    classifiers, input1[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_ll_nsm_f1 = predictive_ability(
    classifiers, input2[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_mrm_ll_nsm_gmean = predictive_ability(
    classifiers, input1[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_ll_nsm_gmean = predictive_ability(
    classifiers, input2[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = svm_mrm_ll_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = rf_mrm_ll_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
# ---------------
'''Standardization versus Normalization & Standardization'''
# SVM
# ---------------
# # Features
# ------------------
input1 = wrap_mrm_svm_050_gmean
input2 = wrap_mrm_ll_svm_050_gmean
print("Number of features selected")
print(str(np.mean([n.n_features_ for n in input1[0]])) +
      " + " + str(np.std([n.n_features_ for n in input1[0]])))
print(str(np.mean([n.n_features_ for n in input2[0]])) +
      " + " + str(np.std([n.n_features_ for n in input2[0]])))
# %%
# ---------------
# # Predictive Performance
# ------------------
# SVM
input = svm_mrm_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = svm_mrm_ll_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
# RF
# ---------------
# # Features
# ------------------
input1 = wrap_mrm_RF_050_gmean
input2 = wrap_mrm_ll_RF_050_gmean
print("Number of features selected:")
print(str(np.mean([n.n_features_ for n in input1[0]])) +
      " + " + str(np.std([n.n_features_ for n in input1[0]])))
print(str(np.mean([n.n_features_ for n in input2[0]])) +
      " + " + str(np.std([n.n_features_ for n in input2[0]])))
# %%
# ---------------
# # Predictive Performance
# ------------------
input = rf_mrm_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = rf_mrm_ll_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
'''
The effect of standardization versus standardization-normalization seems to be minimal,
thus for further testing only standardization will be applied. (The validity of this
can be confirmed with a wilcoxon test).
'''
# %%
# Basic important attributes


def set_style():
    sns.set(context="paper", font='serif', style="white", rc={"xtick.bottom": True,
                                                              "xtick.labelsize": "x-small",
                                                              "ytick.left": True,
                                                              "ytick.labelsize": "x-small",
                                                              "legend.fontsize": "x-small",
                                                              "ytick.major.size": 2,
                                                              "xtick.major.size": 2})


fig_width = 5.52
fig_height_scale = 1.2
# set golden ratio values
gr = (np.sqrt(5)-1)/2
# %%
# ---------------
'''SVM versus RF'''
# ---------------
# # Features
# ------------------
# wrap_mrm_RF_050bal_gmean
# wrap_boruta_mrm_RF_050bal_gmean
# wrap_boruta_mrm_svm_050_gmean

input_bor_svm = wrap_boruta_mrm_svm_050_gmean
input_bor_rf = wrap_boruta_mrm_RF_050bal_gmean
input_ens_svm = wrap_mrm_svm_050_gmean
input_ens_rf = wrap_mrm_RF_050bal_gmean
input_ens_svm25 = wrap_mrm_svm_025_gmean
input_ens_rf25 = wrap_mrm_RF_025bal_gmean

print("Number of selected features:")
print("Boruta SVM: " + str(np.mean([n.n_features_ for n in input_bor_svm[0]])) +
      " + " + str(np.std([n.n_features_ for n in input_bor_svm[0]])))
print("Boruta RF: " + str(np.mean([n.n_features_ for n in input_bor_rf[0]])) +
      " + " + str(np.std([n.n_features_ for n in input_bor_rf[0]])))

print("Ensemble SVM: " + str(np.mean([n.n_features_ for n in input_ens_svm[0]])) +
      " + " + str(np.std([n.n_features_ for n in input_ens_svm[0]])))
print("Ensemble RF: " + str(np.mean([n.n_features_ for n in input_ens_rf[0]])) +
      " + " + str(np.std([n.n_features_ for n in input_ens_rf[0]])))
print("Ensemble SVM @ 25: " + str(np.mean([n.n_features_ for n in input_ens_svm25[0]])) +
      " + " + str(np.std([n.n_features_ for n in input_ens_svm25[0]])))
print("Ensemble RF @ 25: " + str(np.mean([n.n_features_ for n in input_ens_rf25[0]])) +
      " + " + str(np.std([n.n_features_ for n in input_ens_rf25[0]])))
# %%
input = input_ens_svm[0]
# def lineplotcv(input, color1,color2,)
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)

# scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
# scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
# scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]

# crop results to include at least 4 results per number of features
cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)
scores_25 = np.nanmedian(cut_off_dataframe, axis=0)  # cut_off_dataframe.iloc[24]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)

fig_width = 5.52
fig_height_scale = 1
set_style()
fig, ax = plt.subplots(figsize=(fig_width, gr*fig_width*fig_height_scale))

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score of SVM")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C0", label='SVM (lin)')
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)

input = input_ens_rf[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
# sum(pd.DataFrame(scores_list).transform(np.sort).isnull().sum() == 0)
#
# scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
# scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
# scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
# crop results to include at least 4 results per number of features
cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)
scores_25 = np.nanmedian(cut_off_dataframe, axis=0)  # cut_off_dataframe.iloc[24]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)
# fig, ax = plt.subplots()

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C3")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C3")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C3", label="Random Forest")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3, color='C3')
ax.grid()


sns.despine()
ax.legend()
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/svmVSrfEnsemble.png",
#             bbox_inches="tight", dpi=1000)
''' These can be plotted together or seperately, but together, although not the prettiest \n enables the reader to see a clear comparison of the two RFE implementations. '''
# %%
# Zoom
''' Closer Analysis '''
input = input_ens_svm[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)

zoom1 = 40
zoom2 = 0
# scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4, zoom2:zoom1]
# scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24, zoom2:zoom1]
# scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44, zoom2:zoom1]

cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)[zoom2:zoom1]
scores_25 = np.nanmedian(cut_off_dataframe, axis=0)[zoom2:zoom1]  # cut_off_dataframe.iloc[24]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)[zoom2:zoom1]

fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score of SVM")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C0", label='SVM (lin)')
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)

input = input_ens_rf[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)


cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)[zoom2:zoom1]
scores_25 = np.nanmedian(cut_off_dataframe, axis=0)[zoom2:zoom1]  # cut_off_dataframe.iloc[24]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)[zoom2:zoom1]

# scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4, zoom2:zoom1]
# scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24, zoom2:zoom1]
# scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44, zoom2:zoom1]
# fig, ax = plt.subplots()

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C3")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C3")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C3", label="Random Forest")
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3, color="C3")
ax.grid()
sns.despine()
ax.legend()
# %%
input = input_bor_svm[0]
# def lineplotcv(input, color1,color2,)
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)

# crop results to include at least 4 results per number of features
cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)
scores_25 = np.nanmedian(cut_off_dataframe, axis=0)  # cut_off_dataframe.iloc[24]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)

# scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
# scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
# scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
fig_width = 5.52
fig_height_scale = 1
set_style()
fig, ax = plt.subplots(figsize=(fig_width, gr*fig_width*fig_height_scale))

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score of SVM")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C0", label='SVM (lin)')
ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)

input = input_bor_rf[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)

# crop results to include at least 4 results per number of features
cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)
scores_25 = np.nanmedian(cut_off_dataframe, axis=0)  # cut_off_dataframe.iloc[24]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)


# scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
# scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
# scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]
# fig, ax = plt.subplots()

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")
ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color="C3")
ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color="C3")
ax.plot(range(1, len(scores_25) + 1), scores_25, color="C3", label="Random Forest")


# axins1 = zoomed_inset_axes(ax, zoom = 5, loc=2)
# axins1.plot(y,datapts.T)
# axins1.plot(y,datapts.T,'mo')
#
# # SPECIFY THE LIMITS
# x1, x2, y1, y2 = 0,16,0.0,0.8
# axins1.set_xlim(x1, x2)
# axins1.set_ylim(y1, y2)
#
# mark_inset(ax, axins1, loc1=1, loc2=4, fc="none", ec="0.5")

ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3, color='C3')
ax.grid()
sns.despine()
ax.legend()
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/svmVSrfBoruta.png",
#             bbox_inches="tight", dpi=1000)
# %%
''' Sensitivity-Specificity Analysis '''
#
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsx/'
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_specificity', 'rb') as f:
    wrap_mrm_RF_050_spec = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_sensitivity', 'rb') as f:
    wrap_mrm_RF_050_sens = pickle.load(
        f)

input = wrap_mrm_RF_050_spec[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4, zoom2:zoom1]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24, zoom2:zoom1]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44, zoom2:zoom1]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score of SVM")
ax.plot(range(zoom2+1, zoom1 + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(zoom2+1, zoom1 + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(zoom2+1, zoom1 + 1), scores_25, color="C1")
ax.fill_between(range(zoom2+1, zoom1 + 1), scores_5, scores_45, alpha=0.3)

input = wrap_mrm_RF_050_sens[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4, zoom2:zoom1]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24, zoom2:zoom1]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44, zoom2:zoom1]
# fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score of RF")
ax.plot(range(zoom2+1, zoom1 + 1), scores_5, linestyle=':', color="m")
ax.plot(range(zoom2+1, zoom1 + 1), scores_45, linestyle=':', color="m")
ax.plot(range(zoom2+1, zoom1 + 1), scores_25, color="k")
ax.fill_between(range(zoom2+1, zoom1 + 1), scores_5, scores_45, alpha=0.3, color='m')

# %%
''' Analyse results directly '''
#
input = wrap_mrm_RF_050_gmean[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)
pd.DataFrame(scores_list).iloc[:, 4:25].transform(np.sort)
# %%
# ---------------
# # Predictive Performance
# ------------------
preproc = "ens"
# Boruta
svm_boruta_mrm_050b_gmean = predictive_ability(
    classifiers, wrap_boruta_mrm_svm_050_gmean[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_boruta_mrm_050b_gmean = predictive_ability(
    classifiers, wrap_boruta_mrm_RF_050bal_gmean[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
wrap_boruta_mrm_RF_050bal_gmean
input = wrap_boruta_mrm_svm_050_gmean[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list)
pd.DataFrame(scores_list).iloc[:, 4:25].transform(np.sort)
# %%
input1 = wrap_boruta_mrm_svm_050_gmean
print("Number of selected features:")
print("Boruta SVM: " + str(np.mean([n.n_features_ for n in input1[0]])
                           ) + " + " + str(np.std([n.n_features_ for n in input1[0]])))
input = wrap_boruta_mrm_svm_050_gmean[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
# SVM
input = svm_boruta_mrm_050b_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
input2 = wrap_boruta_mrm_RF_050bal_gmean
print("Number of selected features:")
print("Boruta RF: " + str(np.mean([n.n_features_ for n in input2[0]])
                          ) + " + " + str(np.std([n.n_features_ for n in input2[0]])))
input = wrap_boruta_mrm_RF_050bal_gmean[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
# RF
input = rf_boruta_mrm_050b_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
'''
################################################################################################
#                          Evaluating thersheld BORUTA - RFE results (@ 5,10,15)
################################################################################################
'''
# %%
# imports
# RF
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsx/New/'
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_boruta_50_RF_mrm_0_make_scorer(gmean)_5', 'rb') as f:
    wrap_mrm_RF_050bal_gmean_5 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_boruta_50_RF_mrm_0_make_scorer(gmean)_10', 'rb') as f:
    wrap_mrm_RF_050bal_gmean_10 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_boruta_50_RF_mrm_0_make_scorer(gmean)_15', 'rb') as f:
    wrap_mrm_RF_050bal_gmean_15 = pickle.load(
        f)
# SVM
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsx/New/'
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_boruta_50_SVM_linear_mrm_0_make_scorer(gmean)_5', 'rb') as f:
    wrap_mrm_svm_050_gmean_5 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_boruta_50_SVM_linear_mrm_0_make_scorer(gmean)_10', 'rb') as f:
    wrap_mrm_svm_050_gmean_10 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_boruta_50_SVM_linear_mrm_0_make_scorer(gmean)_15', 'rb') as f:
    wrap_mrm_svm_050_gmean_15 = pickle.load(
        f)
# %%
# Predictive ability
# RF
rf_mrm_050bal_gmean_5 = predictive_ability(
    classifiers, wrap_mrm_RF_050bal_gmean_5[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_050bal_gmean_10 = predictive_ability(
    classifiers, wrap_mrm_RF_050bal_gmean_10[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_050bal_gmean_15 = predictive_ability(
    classifiers, wrap_mrm_RF_050bal_gmean_15[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
# SVM
svm_mrm_050_gmean_5 = predictive_ability(
    classifiers, wrap_mrm_svm_050_gmean_5[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_mrm_050_gmean_10 = predictive_ability(
    classifiers, wrap_mrm_svm_050_gmean_10[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_mrm_050_gmean_15 = predictive_ability(
    classifiers, wrap_mrm_svm_050_gmean_15[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
# QUICK OVERVIEW OF Results
input = svm_mrm_050_gmean_15
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
'''
The following section overview the effect of thresholding on RFE with boruta first phase selected features
'''
# %%
results_bor = {
    "RF @ 5": rf_mrm_050bal_gmean_5,
    "SVM @ 5": svm_mrm_050_gmean_5,
    "RF @ 10": rf_mrm_050bal_gmean_10,
    "SVM @ 10": svm_mrm_050_gmean_10,
    "RF @ 15": rf_mrm_050bal_gmean_15,
    "SVM @ 15": svm_mrm_050_gmean_15,
    "RF @ CV": rf_boruta_mrm_050b_gmean,
    "SVM @ CV": svm_boruta_mrm_050b_gmean
}
# %%
fig_height_scale = 1.8
set_style()
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(results_bor, classifiers, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(results_bor, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(results_bor, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.135)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_raw_nsm.png",bbox_inches="tight", dpi=1000)

# %%
'''
# Evaluating ensemble @ cv
'''
rf_mrm_050bal_gmean = predictive_ability(
    classifiers, wrap_mrm_RF_050bal_gmean[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_mrm_050_gmean = predictive_ability(
    classifiers, wrap_mrm_svm_050_gmean[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
input2 = wrap_mrm_RF_050bal_gmean
print("Number of selected features:")
print("Boruta RF: " + str(np.mean([n.n_features_ for n in input2[0]])
                          ) + " + " + str(np.std([n.n_features_ for n in input2[0]])))
input = wrap_mrm_RF_050bal_gmean[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
# RF
input = rf_mrm_050bal_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
input2 = wrap_mrm_svm_050_gmean
print("Number of selected features:")
print("Boruta RF: " + str(np.mean([n.n_features_ for n in input2[0]])
                          ) + " + " + str(np.std([n.n_features_ for n in input2[0]])))
input = wrap_mrm_svm_050_gmean[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
# SVM
input = svm_mrm_050_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
'''
################################################################################################
#                          Evaluating thersheld ENSEMBLE - RFE results (@ 5,10,15)
################################################################################################
'''
# imports
# RF
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsx/New/'
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_make_scorer(gmean)_5', 'rb') as f:
    wrap_ens_mrm_RF_050bal_gmean_5 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_make_scorer(gmean)_10', 'rb') as f:
    wrap_ens_mrm_RF_050bal_gmean_10 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_make_scorer(gmean)_15', 'rb') as f:
    wrap_ens_mrm_RF_050bal_gmean_15 = pickle.load(
        f)
# SVM
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsx/New/'
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_SVM_linear_mrm_0_make_scorer(gmean)_5', 'rb') as f:
    wrap_ens_mrm_svm_050_gmean_5 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_SVM_linear_mrm_0_make_scorer(gmean)_10', 'rb') as f:
    wrap_ens_mrm_svm_050_gmean_10 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_SVM_linear_mrm_0_make_scorer(gmean)_15', 'rb') as f:
    wrap_ens_mrm_svm_050_gmean_15 = pickle.load(
        f)
# %%
# Predictive ability
# RF
preproc = 'ens'
rf_ens_mrm_050bal_gmean_5 = predictive_ability(
    classifiers, wrap_ens_mrm_RF_050bal_gmean_5[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_ens_mrm_050bal_gmean_10 = predictive_ability(
    classifiers, wrap_ens_mrm_RF_050bal_gmean_10[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_ens_mrm_050bal_gmean_15 = predictive_ability(
    classifiers, wrap_ens_mrm_RF_050bal_gmean_15[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
# SVM
svm_ens_mrm_050_gmean_5 = predictive_ability(
    classifiers, wrap_ens_mrm_svm_050_gmean_5[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_ens_mrm_050_gmean_10 = predictive_ability(
    classifiers, wrap_ens_mrm_svm_050_gmean_10[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_ens_mrm_050_gmean_15 = predictive_ability(
    classifiers, wrap_ens_mrm_svm_050_gmean_15[1], X_train, y_train, num_repeats, num_splits, preproc)

# %%
#
# i = 2
# data = np.array(auc_clf_compiler(classifiers, input[4], input[5]))[:,0].std()
# data
# st.norm.interval(alpha=0.95, loc=np.mean(data), scale=st.sem(data))
# %%
input = svm_ens_mrm_050_gmean_10
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%

results_ens = {
    "RF @ 5": rf_ens_mrm_050bal_gmean_5,
    "SVM @ 5": svm_ens_mrm_050_gmean_5,
    "RF @ 10": rf_ens_mrm_050bal_gmean_10,
    "SVM @ 10": svm_ens_mrm_050_gmean_10,
    "RF @ 15": rf_ens_mrm_050bal_gmean_15,
    "SVM @ 15": svm_ens_mrm_050_gmean_15,
    "RF @ CV": rf_mrm_050bal_gmean,
    "SVM @ CV": svm_mrm_050_gmean
}
# %%
fig_height_scale = 1.8
set_style()
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(results_ens, classifiers, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(results_ens, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(results_ens, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.135)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_raw_nsm.png",bbox_inches="tight", dpi=1000)
# %%
'''
################################################################################################
#                    RFE EXTERNAL RESULT GENEREATION @ MULTIPLE THRESHOLDS
################################################################################################
'''
# %%
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
# Initialize data for evaluation
# %%
# Initialize data for input into feature selection and classification
X_train = count_data.to_numpy()  # count matrix numpy array
X = X_train
y_categorical = labels.to_numpy().reshape(len(labels),)  # labels numpy array
# Change categorical labels to binary (controls - 0 and cases - 1)
Label_Encoder = LabelEncoder()
y_train = np.abs(Label_Encoder.fit_transform(y_categorical) - 1)
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
#                                 Import Boruta Selected Features
################################################################################################
# boruta
boruta_pickle_directory = 'D:/Thesis_to_big_file/xboruta outputsx/'

# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 7, 0.01
with open(boruta_pickle_directory+filename+'_boruta_filter_stage_105_16', 'rb') as f:
    boruta_out16 = pickle.load(f)


def extract_boruta_list(boruta_output):
    confirmed_list = []
    tentative_list = []
    selected_list = []
    for fold in range(0, 50):
        X_train_f = X[kf_train_idxcs[fold]]
        confirmed = pd.DataFrame(X_train_f).columns[boruta_output[0][fold].support_].to_list()
        confirmed_list.append(np.array(confirmed))
        tentative = pd.DataFrame(X_train_f).columns[boruta_output[0][fold].support_weak_].to_list()
        tentative_list.append(np.array(tentative))
        selected = confirmed.copy()
        selected.extend(tentative)
        selected_list.append(np.array(selected))
    return confirmed_list, tentative_list, selected_list


confirmed_list, tentative_list, selected_list = extract_boruta_list(boruta_out16)


# %%
################################################################################################
#                                 Import Ensemble Selected Features
################################################################################################
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xfilter outputsx/'
filename = 'ge_raw_6'
# import filter ensemble output
with open(filter_pickle_directory+filename+'_filter_stage_105', 'rb') as f:
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
#                                 Import RFE Selected Features
################################################################################################
# ge_raw_6_rfe_wrapper_stage_ens_50_RF_mrm_0_make_scorer(gmean)_1
# ge_raw_6_rfe_wrapper_stage_ens_50_SVM_linear_mrm_0_make_scorer(gmean)_1
# ge_raw_6_rfe_wrapper_stage_bor_50_RF_mrm_0_make_scorer(gmean)_1
# ge_raw_6_rfe_wrapper_stage_bor_50_SVM_linear_mrm_0_make_scorer(gmean)_1
input_name = 'ge_raw_6_rfe_wrapper_stage_ens_50_RF_mrm_0_make_scorer(gmean)_1'
with open(input_name, 'rb') as f:
    rfe_output = pickle.load(
        f)
# %%
################################################################################################
#                                        RESULTS GENERATION
################################################################################################

# %%
'''Stage 1'''
input_1 = idx_ensemble_list # selected_list #
'''Stage 2'''
input_2 = rfe_output
# %%
len(input_1[0])
len(input_2[0][0].grid_scores_  )
# %%
input_2[1][2]
input_1[2][input_2[0][2].ranking_ <= 1]
input_2[0][2].ranking_
# %%

''' Predictive Performance Generation'''
# input_1[0]
# results_list = []
# for j in range(1,len(input_1[0])):
#     ''' Grab top x number of features '''
#     number_features = j
#     top_list = []
#     for i in range(0, 50):
#         # identify the top {{number_features}} features for each fold
#         ranking = input_2[0][i].ranking_
#         top = input_1[i][ranking <=number_features]
#         top_list.append(top)
#     top_list[0]
#     preproc = 'ens'
#     input = top_list
#     result = predictive_ability(
#         classifiers, top_list, X_train, y_train, num_repeats, num_splits, preproc)
#     results_list.append(result)
# %%
''' Stability Generation'''

# results_list = []
# for j in range(1, len(input_1[0])):
#     ''' Grab top x number of features '''
#     number_features = j
#     top_list = []
#     for i in range(0, 50):
#         # identify the top {{number_features}} features for each fold
#         ranking = input_2[0][i].ranking_
#         top = input_1[i][ranking <= number_features]
#         top_list.append(top)
#     top_list[0]
#     preproc = 'ens'
#     input = top_list
#
#     result = average_tanimoto_index(top_list)
#     results_list.append(result)
# results_list
# ge_raw_6_rfe_wrapper_stage_ens_50_RF_mrm_stability_range = results_list

# ge_raw_6_rfe_wrapper_stage_bor_50_RF_mrm_stability_range
# ge_raw_6_rfe_wrapper_stage_bor_50_SVM_linear_mrm_stability_range
# ge_raw_6_rfe_wrapper_stage_ens_50_RF_mrm_stability_range
# ge_raw_6_rfe_wrapper_stage_ens_50_SVM_linear_mrm_stability_range
# %%
################################################################################################
#                                  PRED PERFORMANCE EVALUTION
################################################################################################
classifier_names = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}

selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF']
# %%
# Pickle dump feature subset score and index lists
# with open(input_name + '_fit_results', 'wb') as f:
#     pickle.dump(results_list, f)
# %%
# ge_raw_6_rfe_wrapper_stage_ens_50_RF_mrm_0_make_scorer(gmean)_1_fit_results
# ge_raw_6_rfe_wrapper_stage_ens_50_SVM_linear_mrm_0_make_scorer(gmean)_1_fit_results
# ge_raw_6_rfe_wrapper_stage_bor_50_RF_mrm_0_make_scorer(gmean)_1_fit_results
# ge_raw_6_rfe_wrapper_stage_bor_50_SVM_linear_mrm_0_make_scorer(gmean)_1_fit_results
results_input_name = 'ge_raw_6_rfe_wrapper_stage_bor_50_SVM_linear_mrm_0_make_scorer(gmean)_1_fit_results'
with open(results_input_name, 'rb') as f:
    rfe_output_results_svm = pickle.load(
        f)
results_input_name = 'ge_raw_6_rfe_wrapper_stage_bor_50_RF_mrm_0_make_scorer(gmean)_1_fit_results'
with open(results_input_name, 'rb') as f:
    rfe_output_results_rf = pickle.load(
        f)

internal_rfe_input_svm = input_ens_svm[0]
internal_rfe_input_rf = input_ens_rf[0]
# %%
''' ----------------------------- Individual Tests ----------------------------- '''
geomean_list_svm = []
geomean_list_svm.append(np.zeros(len(classifier_names)))
for i in range(0, len(rfe_output_results_svm)):
    sensitivity = pd.DataFrame(rfe_output_results_svm[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(rfe_output_results_svm[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_list_svm.append(geomean)
len((geomean_list_svm))
geomean_list_rf = []
geomean_list_rf.append(np.zeros(len(classifier_names)))
for i in range(0, len(rfe_output_results_rf)):
    sensitivity = pd.DataFrame(rfe_output_results_rf[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(rfe_output_results_rf[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_list_rf.append(geomean)
# %%
geomean_list = []
sensitivity_list = []
specificity_list = []
geomean_list.append(np.zeros(len(classifier_names)))
sensitivity_list.append(np.zeros(len(classifier_names)))
specificity_list.append(np.zeros(len(classifier_names)))
for i in range(0, len(rfe_output_results_rf)):
    sensitivity = pd.DataFrame(rfe_output_results_rf[i][2], columns=classifier_names.keys())
    sensitivity_list.append(np.nanmean(sensitivity, axis=0))
    specificity = pd.DataFrame(rfe_output_results_rf[i][3], columns=classifier_names.keys())
    specificity_list.append(np.nanmean(specificity, axis=0))
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_list.append(geomean)
# %%
set_style()
fig_width = 5.8
fig_height_scale = 0.8
fig, ax = plt.subplots(1, figsize=(fig_width, gr*fig_width*fig_height_scale))
for key in selected_classifiers:
    ax.plot(range(0, len(rfe_output_results_rf)+1),
            pd.DataFrame(sensitivity_list, columns=classifier_names)[key], label=key)
ax
# %%
''' ----------------------------- Comparison Tests ----------------------------- '''


def set_style():
    sns.set(context="paper", font='serif', style="white", rc={"xtick.bottom": True,
                                                              "xtick.labelsize": "x-small",
                                                              "ytick.left": True,
                                                              "ytick.labelsize": "x-small",
                                                              "legend.fontsize": "x-small",
                                                              "ytick.major.size": 2,
                                                              "xtick.major.size": 2})


fig_width = 5.52
fig_height_scale = 1.2
# set golden ratio values
gr = (np.sqrt(5)-1)/2
# %%

zoom1 = 32
zoom2 = 0
set_style()
fig_width = 5.8
fig_height_scale = 0.8
fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(fig_width, gr*fig_width*fig_height_scale))

''' SVM '''
input = internal_rfe_input_svm
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list)

# crop results to include at least 4 results per number of features
cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)[zoom2:zoom1]
scores_25 = np.nanmean(cut_off_dataframe, axis=0)[zoom2:zoom1]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)[zoom2:zoom1]

ax1.plot(range(0, len(scores_5)), scores_5, linestyle=':', color="C9")
ax1.plot(range(0, len(scores_45)), scores_45, linestyle=':', color="C9")
ax1.plot(range(0, len(scores_25)), scores_25, color="C9", label='Internal CV SVM (lin)')
ax1.fill_between(range(0, len(scores_5)), scores_5, scores_45, alpha=0.3, color="C9")

for key in selected_classifiers:
    ax1.plot(range(0, len(rfe_output_results_svm)+1),
             pd.DataFrame(geomean_list_svm, columns=classifier_names)[key], label=key)
ax1.grid()
ax1.set_ylim(-0.03, 0.8)
sns.despine()
ax1.set_xlabel("Number of Features")
ax1.set_ylabel("Geometric mean ")

ax1.set_title('Linear-SVM based RFE')
''' RF '''
input = internal_rfe_input_rf
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list)

# crop results to include at least 4 results per number of features
cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)[zoom2:zoom1]
scores_25 = np.nanmean(cut_off_dataframe, axis=0)[zoom2:zoom1]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)[zoom2:zoom1]

ax2.plot(range(0, len(scores_5)), scores_5, linestyle=':', color="C9")
ax2.plot(range(0, len(scores_45)), scores_45, linestyle=':', color="C9")
ax2.plot(range(0, len(scores_25)), scores_25, color="C9", label='Internal CV')
ax2.fill_between(range(0, len(scores_5)), scores_5, scores_45, alpha=0.3, color="C9")

for key in selected_classifiers:
    ax2.plot(range(0, len(rfe_output_results_rf)+1),
             pd.DataFrame(geomean_list_rf, columns=classifier_names)[key], label=key)
ax2.grid()
ax2.set_ylim(-0.03, 0.8)
ax2.set_xlabel("Number of Features")
sns.despine()
ax2.legend()

ax2.set_title('RF based RFE')
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/EnsembleRFEInternalVSExternal.png",
#             bbox_inches="tight", dpi=1000)
# %%
''' ----------------------------- Internal Tests ----------------------------- '''
zoom1 = 50
zoom2 = 0
input = input_bor_svm[0]
# def lineplotcv(input, color1,color2,)
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list)
# scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
# scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
# scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]

# crop results to include at least 4 results per number of features
cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)[zoom2:zoom1]
scores_25 = np.nanmean(cut_off_dataframe, axis=0)[zoom2:zoom1]  # cut_off_dataframe.iloc[24]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)[zoom2:zoom1]

fig_width = 5.52
fig_height_scale = 1
set_style()
fig, ax = plt.subplots(figsize=(fig_width, gr*fig_width*fig_height_scale))

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score of SVM")
ax.plot(range(0, len(scores_5)), scores_5, linestyle=':', color="C0")
ax.plot(range(0, len(scores_45)), scores_45, linestyle=':', color="C0")
ax.plot(range(0, len(scores_25)), scores_25, color="C0", label='SVM (lin)')
ax.fill_between(range(0, len(scores_5)), scores_5, scores_45, alpha=0.3)

input = input_bor_rf[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)

cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
    scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)[zoom2:zoom1]
scores_25 = np.nanmean(cut_off_dataframe, axis=0)[zoom2:zoom1]  # cut_off_dataframe.iloc[24]
scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)[zoom2:zoom1]

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")
ax.plot(range(0, len(scores_5)), scores_5, linestyle=':', color="C3")
ax.plot(range(0, len(scores_45)), scores_45, linestyle=':', color="C3")
ax.plot(range(0, len(scores_25)), scores_25, color="C3", label="Random Forest")
ax.fill_between(range(0, len(scores_5)), scores_5, scores_45, alpha=0.3, color='C3')
ax.grid()


sns.despine()
ax.legend()
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/svmVSrfEnsemble.png",
#             bbox_inches="tight", dpi=1000)
# %%
################################################################################################
#                                   STABILITY EVALUTION
################################################################################################
# ge_raw_6_rfe_wrapper_stage_bor_50_RF_mrm_stability_range
# ge_raw_6_rfe_wrapper_stage_bor_50_SVM_linear_mrm_stability_range
# ge_raw_6_rfe_wrapper_stage_ens_50_RF_mrm_stability_range
# ge_raw_6_rfe_wrapper_stage_ens_50_SVM_linear_mrm_stability_range
input = ge_raw_6_rfe_wrapper_stage_bor_50_RF_mrm_stability_range
np.max(input)
np.min(input)
np.mean(input)
input[14]
np.where(input == np.max(input))
# %%
input = ge_raw_6_rfe_wrapper_stage_bor_50_SVM_linear_mrm_stability_range
np.max(input)
np.min(input)
np.mean(input)
input[14]
np.where(input == np.max(input))
# %%
input = ge_raw_6_rfe_wrapper_stage_ens_50_RF_mrm_stability_range
np.max(input)
np.min(input)
np.mean(input)
input[14]
np.where(input == np.max(input))
# %%
input = ge_raw_6_rfe_wrapper_stage_ens_50_SVM_linear_mrm_stability_range
np.max(input)
np.min(input)
np.mean(input)
input[14]
np.where(input == np.max(input))

# %%
'''
################################################################################################
#                         Final RFE EVALUATION AND RESULTS @ 6 MONTHS
################################################################################################
'''


def set_style():
    sns.set(context="paper", font='serif', style="white", rc={"xtick.bottom": True,
                                                              "xtick.labelsize": "x-small",
                                                              "ytick.left": True,
                                                              "ytick.labelsize": "x-small",
                                                              "legend.fontsize": "x-small",
                                                              "ytick.major.size": 2,
                                                              "xtick.major.size": 2})


fig_width = 5.52
fig_height_scale = 1.2

classifier_names = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}

selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF']
# set golden ratio values
gr = (np.sqrt(5)-1)/2
# %%

results = {
    "Ensemble, RF": rf_mrm_050bal_gmean,
    "Ensemble, SVM (lin)": svm_mrm_050_gmean,
    "Boruta, RF": rf_boruta_mrm_050b_gmean,
    "Boruta, SVM (lin)": svm_boruta_mrm_050b_gmean
}
# %%
fig_height_scale = 1.5
set_style()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(results_bor, classifier_names, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xlabel("RFE implementations with Boruta selected features")
ax1.set_ylabel("External Cross-validation\n Predictive Performance")
ax1.set_ylim(0.2, 1)
# ax1.set_xticklabels([])
# ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(results_ens, classifier_names, "AUC", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xlabel("RFE implementations with ensemble selected features")
ax2.set_ylabel("External Cross-validation\n Predictive Performance")
ax2.set_ylim(0.2, 1)
# ax2.set_xticklabels([])
# ax2.get_xaxis().get_label().set_visible(False)
# ax3 = boxplot_filter(results, internal_rfe_input_svm, "Specificity", selected_classifiers, ax3)
# ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.545, 0), loc="lower center", ncol=len(classifiers))
fig.tight_layout()
fig.subplots_adjust(bottom=0.16)

# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/EnsBorExternalPredPerfGeo.png",bbox_inches="tight", dpi=1000)
# %%
# Stability
print("Stability Ensemble, RF - " + str(average_tanimoto_index(wrap_mrm_RF_050bal_gmean[1])))
print("Stability Ensemble, RF, 5 - " +
      str(average_tanimoto_index(wrap_ens_mrm_RF_050bal_gmean_5[1])))
print("Stability Ensemble, RF, 10 - " +
      str(average_tanimoto_index(wrap_ens_mrm_RF_050bal_gmean_10[1])))
print("Stability Ensemble, RF, 15 - " +
      str(average_tanimoto_index(wrap_ens_mrm_RF_050bal_gmean_15[1])))
print("\nStability Ensemble, SVM (lin) - " + str(average_tanimoto_index(wrap_mrm_svm_050_gmean[1])))
print("Stability Ensemble, SVM, 5 - " +
      str(average_tanimoto_index(wrap_ens_mrm_svm_050_gmean_5[1])))
print("Stability Ensemble, SVM, 10 - " +
      str(average_tanimoto_index(wrap_ens_mrm_svm_050_gmean_10[1])))
print("Stability Ensemble, SVM, 15 - " +
      str(average_tanimoto_index(wrap_ens_mrm_svm_050_gmean_15[1])))
print("\n==============================\nStability Boruta, RF - " +
      str(average_tanimoto_index(wrap_boruta_mrm_RF_050bal_gmean[1])))
print("Stability Boruta, RF, 5 - " + str(average_tanimoto_index(wrap_mrm_RF_050bal_gmean_5[1])))
print("Stability Boruta, RF, 10 - " + str(average_tanimoto_index(wrap_mrm_RF_050bal_gmean_10[1])))
print("Stability Boruta, RF, 15 - " + str(average_tanimoto_index(wrap_mrm_RF_050bal_gmean_15[1])))
print("\nStability Boruta, SVM (lin) - " +
      str(average_tanimoto_index(wrap_boruta_mrm_svm_050_gmean[1])))
print("Stability Boruta, SVM, 5 - " + str(average_tanimoto_index(wrap_mrm_svm_050_gmean_5[1])))
print("Stability Boruta, SVM, 10 - " + str(average_tanimoto_index(wrap_mrm_svm_050_gmean_10[1])))
print("Stability Boruta, SVM, 15 - " + str(average_tanimoto_index(wrap_mrm_svm_050_gmean_15[1])))


# print("Stability 5 - " + str(average_tanimoto_index(confirmed_lst_auto5)))
# print("Stability 6 - " + str(average_tanimoto_index(confirmed_lst_auto7)))
# print("Stability 7 - " + str(average_tanimoto_index(confirmed_lst_50003)))
# print("Stability 8 - " + str(average_tanimoto_index(confirmed_lst_50005)))
# print("Stability 9 - " + str(average_tanimoto_index(confirmed_lst_50007)))
# %%
ens_rfe_overlap = {
    "RF, 5": wrap_ens_mrm_RF_050bal_gmean_5[1],
    "RF, 10": wrap_ens_mrm_RF_050bal_gmean_10[1],
    "RF, 15": wrap_ens_mrm_RF_050bal_gmean_15[1],
    "SVM, 5": wrap_ens_mrm_svm_050_gmean_5[1],
    "SVM, 10": wrap_ens_mrm_svm_050_gmean_10[1],
    "SVM, 15": wrap_ens_mrm_svm_050_gmean_15[1]
}

bor_rfe_overlap = {
    "RF, 5": wrap_mrm_RF_050bal_gmean_5[1],
    "RF, 10": wrap_mrm_RF_050bal_gmean_10[1],
    "RF, 15": wrap_mrm_RF_050bal_gmean_15[1],
    "SVM, 5": wrap_mrm_svm_050_gmean_5[1],
    "SVM, 10": wrap_mrm_svm_050_gmean_10[1],
    "SVM, 15": wrap_mrm_svm_050_gmean_15[1]
}

all_rfe_overlap = {
    "Bor, RF, 5": wrap_mrm_RF_050bal_gmean_5[1],
    "Bor, RF, 10": wrap_mrm_RF_050bal_gmean_10[1],
    "Bor, RF, 15": wrap_mrm_RF_050bal_gmean_15[1],
    "Bor, SVM, 5": wrap_mrm_svm_050_gmean_5[1],
    "Bor, SVM, 10": wrap_mrm_svm_050_gmean_10[1],
    "Bor, SVM, 15": wrap_mrm_svm_050_gmean_15[1],
    "Ens, RF, 5": wrap_ens_mrm_RF_050bal_gmean_5[1],
    "Ens, RF, 10": wrap_ens_mrm_RF_050bal_gmean_10[1],
    "Ens, RF, 15": wrap_ens_mrm_RF_050bal_gmean_15[1],
    "Ens, SVM, 5": wrap_ens_mrm_svm_050_gmean_5[1],
    "Ens, SVM, 10": wrap_ens_mrm_svm_050_gmean_10[1],
    "Ens, SVM, 15": wrap_ens_mrm_svm_050_gmean_15[1]
}
# %%


set_list_th = bor_rfe_overlap
num_overlap_list2 = []
for key_1, filter_output_1 in set_list_th.items():
    one = pd.unique(pd.DataFrame(set_list_th[key_1]).values.ravel())
    num_overlap_list1 = []
    for key_2, filter_output_2 in set_list_th.items():
        two = pd.unique(pd.DataFrame(set_list_th[key_2]).values.ravel())
        overlapping_features = np.intersect1d(one, two)
        num_overlap = len(overlapping_features)
        num_overlap_list1.append(num_overlap)
    num_overlap_list2.append(num_overlap_list1)

num_overlap_list2

filter_names = set_list_th.keys()
# set_style()
fig, ax = plt.subplots(figsize=(10, 8.4))

# sns.heatmap(cmat, annot=True, xticklabels=['Faulty', 'Healthy'], cbar=False, ax=ax)
# ax.set_yticklabels(['Faulty', 'Healthy'], va='center', rotation = 90, position=(0,0.28))
mask = np.triu(np.ones_like(num_overlap_list2, dtype=bool))
~mask.T
overlap_heatmap = sns.heatmap(pd.DataFrame(num_overlap_list2, index=filter_names, columns=filter_names),
                              annot=True, annot_kws={"size": 10}, fmt='d', cbar=True, mask=~mask.T, ax=ax)
ax.set_yticklabels(labels=filter_names, va='center')
# %%
print("Boruta Size Confidence Interval")

# wrap_boruta_mrm_RF_050bal_gmean
input = wrap_mrm_RF_050bal_gmean[1]
input
size_list = []
for lst in input:
    size_list.append(len(lst))
size_list
print("Max - ", np.max(size_list))
print("10 - ", np.sort(size_list)[44])
print("50 - ", np.sort(size_list)[24])
print("90 - ", np.sort(size_list)[4])
print("Min - ", np.min(size_list))
print("Mean -", np.mean(size_list))
# %%
'''
# Evaluate ensemble-rfe with only 25 features from ensemble instead of 50
'''
# Ensemble
rf_mrm_025bal_gmean = predictive_ability(
    classifiers, wrap_mrm_RF_025bal_gmean[1], X_train, y_train, num_repeats, num_splits, preproc)
svm_mrm_025_gmean = predictive_ability(
    classifiers, wrap_mrm_svm_025_gmean[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
# RF
input = rf_mrm_025bal_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
input2 = wrap_mrm_RF_025bal_gmean
print("Number of selected features:")
print("Boruta RF: " + str(np.mean([n.n_features_ for n in input2[0]])
                          ) + " + " + str(np.std([n.n_features_ for n in input2[0]])))
input = wrap_mrm_RF_025bal_gmean[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
# SVM
input = svm_mrm_025_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
input2 = wrap_mrm_svm_025_gmean
print("Number of selected features:")
print("Boruta RF: " + str(np.mean([n.n_features_ for n in input2[0]])
                          ) + " + " + str(np.std([n.n_features_ for n in input2[0]])))
input = wrap_mrm_svm_025_gmean[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
'''
From these results it is evident that the SVM has generally higher AUC scores, but upon
closer inspection it can be seen this is a result of a higher bias towards the majority
class. Furthermore, when regarding the number of selected features, it is evident that
the Random Forest is better able to reduce the number of selected feature than the SVM.
It can however also be noted that the RF selected feature lead to more generally discri-
minitive features
'''
# %%
# ---------------
'''Effect of RFE reduction'''
# ---------------
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsx/'
# Pickle load wrapper outputs

with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_gmean_20', 'rb') as f:
    wrap_mrm_rf_050_20 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_gmean_5', 'rb') as f:
    wrap_mrm_rf_050_5 = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_gmean_50', 'rb') as f:
    wrap_mrm_rf_050_50 = pickle.load(
        f)
# %%
# Features
# ------------------
input1 = wrap_mrm_RF_050_gmean
input2 = wrap_mrm_rf_050_20
input3 = wrap_mrm_rf_050_5

print(str(np.mean([n.n_features_ for n in input1[0]])) +
      " + " + str(np.std([n.n_features_ for n in input1[0]])))
print(str(np.mean([n.n_features_ for n in input2[0]])) +
      " + " + str(np.std([n.n_features_ for n in input2[0]])))
print(str(np.mean([n.n_features_ for n in input3[0]])) +
      " + " + str(np.std([n.n_features_ for n in input3[0]])))
# %%
# ---------------
# # Predictive Performance
# ------------------

rf_mrm_nsm_gmean_20 = predictive_ability(
    classifiers, wrap_mrm_rf_050_20[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_nsm_gmean_5 = predictive_ability(
    classifiers, wrap_mrm_rf_050_5[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_nsm_gmean_50 = predictive_ability(
    classifiers, wrap_mrm_rf_050_50[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
# RF with 20 features
input = rf_mrm_nsm_gmean_20
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = rf_mrm_nsm_gmean_5
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
input = rf_mrm_nsm_gmean_50
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
# RF with all features
input = svm_mrm_nsm_gmean
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
# ---------------
''' Temporal Effect '''
# ---------------
with open(filter_pickle_directory+'ge_raw_18_rfe_wrapper_stage_50_RF_mrm_0_gmean', 'rb') as f:
    wrap_mrm_rf_050_18_cv = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_24_rfe_wrapper_stage_50_RF_mrm_0_gmean', 'rb') as f:
    wrap_mrm_rf_050_24_cv = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_12_rfe_wrapper_stage_50_RF_mrm_0_gmean', 'rb') as f:
    wrap_mrm_rf_050_12_cv = pickle.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_rfe_wrapper_stage_50_RF_mrm_0_gmean', 'rb') as f:
    wrap_mrm_rf_050_6_cv = pickle.load(
        f)
# %%
# ---------------
# # Predictive Performance
# ------------------
# %%
preproc
# %%
rf_mrm_nsm_gmean_cv6 = predictive_ability(
    classifiers, wrap_mrm_rf_050_6_cv[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_nsm_gmean_cv12 = predictive_ability(
    classifiers, wrap_mrm_rf_050_12_cv[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_nsm_gmean_cv18 = predictive_ability(
    classifiers, wrap_mrm_rf_050_18_cv[1], X_train, y_train, num_repeats, num_splits, preproc)
rf_mrm_nsm_gmean_cv24 = predictive_ability(
    classifiers, wrap_mrm_rf_050_24_cv[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
# 6
input = rf_mrm_nsm_gmean_cv6
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
# 12
input = rf_mrm_nsm_gmean_cv12
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
# 18
input = rf_mrm_nsm_gmean_cv18
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
# 24
input = rf_mrm_nsm_gmean_cv24
m_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms
# %%
# Features
# ------------------
input1 = wrap_mrm_rf_050_6_cv
input2 = wrap_mrm_rf_050_12_cv
input3 = wrap_mrm_rf_050_18_cv
input4 = wrap_mrm_rf_050_24_cv

print(str(np.mean([n.n_features_ for n in input1[0]])) +
      " + " + str(np.std([n.n_features_ for n in input1[0]])))
print(str(np.mean([n.n_features_ for n in input2[0]])) +
      " + " + str(np.std([n.n_features_ for n in input2[0]])))
print(str(np.mean([n.n_features_ for n in input3[0]])) +
      " + " + str(np.std([n.n_features_ for n in input3[0]])))
print(str(np.mean([n.n_features_ for n in input4[0]])) +
      " + " + str(np.std([n.n_features_ for n in input4[0]])))
# %%
# Gridsearch results
# ------------------
zoom1 = 50
zoom2 = 0

input = wrap_mrm_rf_050_6_cv[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4, zoom2:zoom1]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24, zoom2:zoom1]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44, zoom2:zoom1]
fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score of SVM")
ax.plot(range(zoom2+1, zoom1 + 1), scores_5, linestyle=':', color="C0")
ax.plot(range(zoom2+1, zoom1 + 1), scores_45, linestyle=':', color="C0")
ax.plot(range(zoom2+1, zoom1 + 1), scores_25, color="C1")
ax.fill_between(range(zoom2+1, zoom1 + 1), scores_5, scores_45, alpha=0.3)

input = wrap_mrm_rf_050_12_cv[0]
scores_list = []
for i in range(0, 50):
    scores = input[i].grid_scores_
    scores_list.append(scores)
pd.DataFrame(scores_list).transform(np.sort)

scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4, zoom2:zoom1]
scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24, zoom2:zoom1]
scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44, zoom2:zoom1]
# fig, ax = plt.subplots()

ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross validation Predictive Performance Score of RF")
ax.plot(range(zoom2+1, zoom1 + 1), scores_5, linestyle=':', color="m")
ax.plot(range(zoom2+1, zoom1 + 1), scores_45, linestyle=':', color="m")
ax.plot(range(zoom2+1, zoom1 + 1), scores_25, color="k")
ax.fill_between(range(zoom2+1, zoom1 + 1), scores_5, scores_45, alpha=0.3, color='m')
# %%
# ---------------
# Effect of SMOTE
''' Note: The SMOTE tests were completed prior to the implementation of the Pipeline, but it is assumed that
the use of SMOTE in the feature selection process will bias the process (this was also documented in an article).
Thus it was not tested with the pipelines'''
# %%
