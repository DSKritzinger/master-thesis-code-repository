'''
The following code aims to provide the evaluation results of the first phase (filter algorithms) of the feature selection process.

This code evaluates the filter algorithms in terms of their:
- stability: by use of the tanimoto index
- predictive performance: in terms of sensitivity and specificity

This code also allows the testing of some filter parameters, which include:
- thresholds
- ensembling vs non-ensembling

'''
# %%
# Imports
# Basics
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import stats
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from skfeature.function.statistical_based import gini_index
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import reliefF
from skfeature.function.statistical_based import chi_square
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc
from eval_functions import average_tanimoto_index, tanimoto_index, predictive_ability, intersystem_ATI
from median_ratio_method import geo_mean, median_ratio_standardization, median_ratio_standardization_, median_ratio_standardization_log
import matplotlib.ticker as ticker
import matplotlib
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import statistics as st
import math
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %%
################################################################################################
# Functions
'''
input:  ranker_score_lists = ranker filter scores (order: fisher, chi, reliefF, mim, gini, mrmr)  and fold sample indices
ouptput: ordered by score, ranker filter method indices (order: fisher, chi, reliefF, mim, gini, mrmr)
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


'''
input:  ranked_ranker_dict = ranked ranker filter indices in dictionary
        treshold = # of genes to select
ouptput: 'top-k' ranker filter method indices in dictionary
'''


def rank_thres_dict(ranked_ranker_dict, threshold):
    output_dict = dict(ranked_ranker_dict)
    item = 0
    for list_key, list in output_dict.items():
        output_dict[list_key] = [item[0:threshold] for item in list]
    return output_dict


'''
- plot_classifier_nfold_rocs

input:  classifiers = dictionary of classifiers to be used
        fpr_list = list of fpr's from predictive performance function of a specific filter method
        tprs_list = list of tpr's from predictive performance function of a specific filter method
        figure_size = size of output figure
ouptput: ROC Curve for said filter run on all classifiers
'''


def plot_classifier_nfold_rocs(classifiers_sel, fpr_list, tprs_list, figure_size, axis):
    # initialize graph variables
    mean_tprs_list = []
    mean_fprs_list = []
    mean_auc_list = []
    mean_fpr = np.linspace(0, 1, 100)

    ax = axis

    for clf_key in classifiers_sel:

        fpr_list_df = pd.DataFrame(fpr_list, columns=classifiers_sel)[clf_key]
        tpr_list_df = pd.DataFrame(tprs_list, columns=classifiers_sel)[clf_key]
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


'''
- auc__clf_compiler
input:  classifiers = dictionary of classifiers to be used
        fpr_list = list of fpr's from predictive performance function of a specific filter method
        tprs_list = list of tpr's from predictive performance function of a specific filter method
ouptput: list of auc for each fold of each classifier for said filter
'''


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


def mean_tables(fs_pa, classifiers_, data_name, sel_classifiers):
    # initialize empty dataframes for input to graph
    # ordered list of classifier names
    classifier_names = list(classifiers_.keys())
    mean_filter = pd.DataFrame(columns=classifier_names)
    std_filter = pd.DataFrame(columns=classifier_names)
    # extract
    for pa_keys, pa_values in fs_pa.items():
        # accuracy
        if data_name == "Accuracy":
            # accuracy
            ind_filter = pd.DataFrame(pa_values[1], columns=classifier_names)
            # mean
            mean_filter = mean_filter.append(ind_filter.iloc[-2], ignore_index=True)
            # std
            std_filter = std_filter.append(ind_filter.iloc[-1], ignore_index=True)
        elif data_name == "Sensitivity":
            # sensitivity
            ind_filter = pd.DataFrame(pa_values[2], columns=classifier_names)
            # mean
            mean_filter = mean_filter.append(ind_filter.iloc[-2], ignore_index=True)
            # std
            std_filter = std_filter.append(ind_filter.iloc[-1], ignore_index=True)
        elif data_name == "Specificity":
            ind_filter = pd.DataFrame(pa_values[3], columns=classifier_names)
            # mean
            mean_filter = mean_filter.append(ind_filter.iloc[-2], ignore_index=True)
            # std
            std_filter = std_filter.append(ind_filter.iloc[-1], ignore_index=True)
        elif data_name == "AUC":
            # fpr
            fpr_list = pa_values[-2]
            # tpr
            tpr_list = pa_values[-1]
            # extract auc for all Classifiers
            auc_clf_list = auc_clf_compiler(classifiers_, fpr_list, tpr_list)
            # mean
            ind_mean_filter = pd.DataFrame(auc_clf_list, columns=classifier_names).mean(axis=0)
            mean_filter = mean_filter.append(ind_mean_filter, ignore_index=True)
            # std
            ind_std_filter = pd.DataFrame(auc_clf_list, columns=classifier_names).std(axis=0)
            std_filter = std_filter.append(ind_std_filter, ignore_index=True)
        elif data_name == "Geo":
            spec_ind_filter = pd.DataFrame(pa_values[3], columns=classifier_names)
            sens_ind_filter = pd.DataFrame(pa_values[2], columns=classifier_names)
            # mean
            spec_mean = spec_ind_filter.iloc[-2]
            sens_mean = sens_ind_filter.iloc[-2]
            mean_filter = mean_filter.append(np.sqrt(sens_mean*spec_mean), ignore_index=True)
            # std
            spec_std = spec_ind_filter.iloc[-1]
            sens_std = sens_ind_filter.iloc[-1]
            std_filter = std_filter.append(np.sqrt(sens_std*spec_std), ignore_index=True)
    # mean
    mean_filter = mean_filter[sel_classifiers]
    mean_filter["Filters"] = fs_pa.keys()
    # standard deviation
    std_filter = std_filter[sel_classifiers]
    std_filter["Filters"] = fs_pa.keys()
    return mean_filter, std_filter


def data_tables(fs_pa, classifiers_, data_name):
    # initialize empty dataframes for input to graph
    classifier_names = list(classifiers_.keys())
    all_filter = pd.DataFrame(columns=classifier_names)
    # extract
    for pa_keys, pa_values in fs_pa.items():
        # accuracy
        if data_name == "accuracy":
            ind_filter = pd.DataFrame(
                pa_values[1][:-2], columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)
        elif data_name == "sensitivity":
            # sensitivity
            ind_filter = pd.DataFrame(
                pa_values[2][:-2], columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)
        elif data_name == "specificity":
            # specificity
            ind_filter = pd.DataFrame(
                pa_values[3][:-2], columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)
        elif data_name == "auc":
            # fpr
            fpr_list = pa_values[-2]
            # tpr
            tpr_list = pa_values[-1]
            # extract auc for all classifiers_
            auc_clf_list = auc_clf_compiler(classifiers_, fpr_list, tpr_list)
            ind_filter = pd.DataFrame(
                auc_clf_list, columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)

    # melt all_filter dataframe for input to graph
    all_filter_m = all_filter.melt(
        id_vars=['Filters'], var_name='Classifiers', value_name=data_name)
    return all_filter_m


'''
- boxplot_filter
input:  fs_pa =  feature selection methods predictive ability outputs
        classifiers = dictionary of classifiers to be used
        data_name = results to be graphed (accuracy, sensitivity, specificity, auc)
        figure_size = size of figure to be output
ouptput: boxplot of relevant filters
'''


def boxplot_filter(fs_pa, classifiers_, data_name, sel_classifiers, axis, ordering=None):
    #fs_pa = results
    #classifiers_ = classifiers
    #data_name = "Geomean"
    #sel_classifiers = selected_classifiers
    #axis = ax2
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
    #ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    return ax


def radar_plot(fs_pa, classifiers_, data_name, sel_classifiers, axis):
    # initialize empty dataframes for input to graph
    # ordered list of classifier names
    classifier_names = list(classifiers_.keys())
    mean_filter = pd.DataFrame(columns=classifier_names)
    # extract
    for pa_keys, pa_values in fs_pa.items():
        # accuracy
        if data_name == "accuracy":
            # accuracy
            ind_filter = pd.DataFrame(pa_values[1], columns=classifier_names)
            # mean
            mean_filter = mean_filter.append(ind_filter.iloc[-2], ignore_index=True)
        elif data_name == "sensitivity":
            # sensitivity
            ind_filter = pd.DataFrame(pa_values[2], columns=classifier_names)
            # mean
            mean_filter = mean_filter.append(ind_filter.iloc[-2], ignore_index=True)
        elif data_name == "specificity":
            ind_filter = pd.DataFrame(pa_values[3], columns=classifier_names)
            # mean
            mean_filter = mean_filter.append(ind_filter.iloc[-2], ignore_index=True)
        elif data_name == "auc":
            # fpr
            fpr_list = pa_values[-2]
            # tpr
            tpr_list = pa_values[-1]
            # extract auc for all Classifiers
            auc_clf_list = auc_clf_compiler(classifiers_, fpr_list, tpr_list)
            ind_filter = pd.DataFrame(auc_clf_list, columns=classifier_names).mean(axis=0)
            mean_filter = mean_filter.append(ind_filter, ignore_index=True)
    # set data for input to graph
    data_all = mean_filter
    # select classifiers to be displayed
    data = data_all[sel_classifiers]
    # create colour pallete for graph lines
    palette = plt.cm.get_cmap("Set2", len(data)+1)
    # initialize axis

    for i in range(0, len(data)):
        # Create background
        # number of variable
        sel_classifiers = selected_classifiers
        categories = sel_classifiers
        N = len(categories)
        # determine angle of each axis
        angles = [n/float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        angles
        # set first axis to be on top
        axis.set_theta_offset(np.pi / 2)
        axis.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        axis.set_xticks(angles[:-1])
        axis.set_xticklabels(categories)
        axis.tick_params(axis='x', pad=10)

        # Draw ylabels
        axis.set_rlabel_position(0)

        axis.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [
            "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
        rlabels = axis.get_ymajorticklabels()
        for label in rlabels:
            label.set_color('gray')

        if data_name == 'auc':
            axis.set_ylim(0.3, 1)
        else:
            axis.set_ylim(0, 1)

        colour = palette(i)

        values = data.iloc[i].values.tolist()
        values += values[:1]
        axis.plot(angles, values, color=colour, marker=".",
                  linestyle='solid', label=list(fs_pa.keys())[i])
        axis.legend(loc='upper left', bbox_to_anchor=(1.25, 1))
    return axis


# set golden ratio values
gr = (np.sqrt(5)-1)/2
# %%
'''##############################################Choose############################################'''
filename = 'ge_raw_6'
'''################################################################################################'''
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
# Raw
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
# ----------------- Train Indices Review -----------------
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
# %%
# Initialize data for input into feature selection and classification
X_train = count_data.to_numpy()  # count matrix numpy array
y_categorical = labels.to_numpy().reshape(len(labels),)  # labels numpy array
# Change categorical labels to binary (controls - 0 and cases - 1)
Label_Encoder = LabelEncoder()
y_train = Label_Encoder.fit_transform(y_categorical)
num_splits = 10
num_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=0)
# initialize lists
kf_train_idxcs = []
kf_test_idxcs = []

for kf_train_index, kf_test_index in rskf.split(X_train, y_train):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)
# Confirm that the folds of the filters are the same as the original data folds
# %%
mrm_log_subset_105[0]
pd.DataFrame(mrm_log_subset_105[2]).equals(pd.DataFrame(kf_train_idxcs))
# %%
################################################################################################
# -------------------------------------------Compare-------------------------------------------
################################################################################################
# Comparison of predictive performance
################################################################################################
# Import predictive performance results
################################################################################################
thres_num = 125
new_keys = ['mRMR', 'ReliefF', 'Chi-Square', 'Fisher-Score',
            'Info Gain', 'Gini Index', 'CFS', 'FCBF', 'All']
new_keys_bor = ['mRMR', 'ReliefF', 'Chi-Square', 'Fisher-Score',
                'Info Gain', 'Gini Index', 'CFS', 'FCBF', 'Boruta', 'All']
# ---- Raw smote ----
with open(filename + '_raw_105_predperf_th'+str(thres_num), 'rb') as f:
    fs_pa_raw_sm = pickle.load(f)
with open(filename + '_105_predperf_boruta' + '_raw_smote', 'rb') as f:
    fs_pa_bor_raw_sm = pickle.load(f)
# Append Boruta results
fs_pa_raw_sm['Boruta'] = fs_pa_bor_raw_sm
fs_pa_raw_sm.keys()
fs_pa_raw_sm = dict(zip(new_keys_bor, fs_pa_raw_sm.values()))
# ---- Raw no smote ----
with open(filename + '_raw_105_predperf_th'+str(thres_num)+"_nsm", 'rb') as f:
    fs_pa_raw_nsm = pickle.load(f)
with open(filename + '_105_predperf_boruta' + '_raw', 'rb') as f:
    fs_pa_bor_raw = pickle.load(f)
# Append Boruta results
fs_pa_raw_nsm['Boruta'] = fs_pa_bor_raw
fs_pa_raw_nsm.keys()
fs_pa_raw_nsm = dict(zip(new_keys_bor, fs_pa_raw_nsm.values()))

# ---- MRM smote ----
with open(filename + '_mrm_105_predperf_th'+str(thres_num), 'rb') as f:
    fs_pa_mrm_sm = pickle.load(f)
with open(filename + '_105_predperf_boruta' + '_mrm_smote', 'rb') as f:
    fs_pa_bor_mrm_sm = pickle.load(f)
# Append Boruta results
fs_pa_mrm_sm['Boruta'] = fs_pa_bor_mrm_sm
fs_pa_mrm_sm.keys()
fs_pa_mrm_sm = dict(zip(new_keys_bor, fs_pa_mrm_sm.values()))

# ---- MRM log (post-filter) + smote ----
with open(filename + '_mrm_plog_105_predperf_th'+str(thres_num), 'rb') as f:
    fs_pa_mrm_plog_sm = pickle.load(f)
fs_pa_mrm_plog_sm = dict(zip(new_keys, fs_pa_mrm_plog_sm.values()))

# ---- MRM no smote ----
with open(filename + '_mrm_105_predperf_th'+str(thres_num)+"_nsm", 'rb') as f:
    fs_pa_mrm_nsm = pickle.load(f)
fs_pa_mrm_nsm = dict(zip(new_keys, fs_pa_mrm_nsm.values()))
# MRM smote with minmax
# with open(filename + '_mrm_105_predperf_th'+str(thres_num)+"_minmax", 'rb') as f:
#     fs_pa_mrm_sm_mm = pickle.load(f)
# fs_pa_mrm_sm_mm = dict(zip(new_keys, fs_pa_mrm_sm_mm.values()))

# ---- MRM log smote ----
with open(filename + '_mrm_log_105_predperf_th'+str(thres_num), 'rb') as f:
    fs_pa_mrm_log_sm = pickle.load(f)
with open(filename + '_105_predperf_boruta' + '_mrm_log_smote', 'rb') as f:
    fs_pa_bor_log_sm = pickle.load(f)
# Append Boruta results
fs_pa_mrm_log_sm['Boruta'] = fs_pa_bor_log_sm
fs_pa_mrm_log_sm.keys()
fs_pa_mrm_log_sm = dict(zip(new_keys_bor, fs_pa_mrm_log_sm.values()))

# ---- MRM loglog smote ----
with open(filename + '_mrm_log_log_105_predperf_th'+str(thres_num), 'rb') as f:
    fs_pa_mrm_loglog_sm = pickle.load(f)
with open(filename + '_105_predperf_boruta' + '_mrm_log_log_smote', 'rb') as f:
    fs_pa_bor_loglog_sm = pickle.load(f)
# Append Boruta results
fs_pa_mrm_loglog_sm['Boruta'] = fs_pa_bor_loglog_sm
fs_pa_mrm_loglog_sm.keys()
fs_pa_mrm_loglog_sm = dict(zip(new_keys_bor, fs_pa_mrm_loglog_sm.values()))
# %%
# classifiers
classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}
# %%
################################################################################################
# Preditve Performance: Filter method comparison
################################################################################################
# Preprocessing evaluation
################################################################################################


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
selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF']
# %%
# RAW --------------------------------------------------
# ----------------- Effect of no SMOTE -----------------

fig_height_scale = 1.2
set_style()
fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
# ax1 = boxplot_filter(fs_pa_raw_nsm, classifiers, "Geo", selected_classifiers, ax1)
# ax1.legend_.remove()
# ax1.set_xticklabels([])
# ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(fs_pa_raw_nsm, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(fs_pa_raw_nsm, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax2.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.2)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_raw_nsm.png",bbox_inches="tight", dpi=1000)
# %%
# RAW -----------------------------------------------
# ----------------- Effect of SMOTE -----------------
# Polar plot
fig_width_p = 5.52
fig_height_scale_p = 1
set_style()
# Initialise the radar plot
#fig = plt.figure(figsize=(fig_width, fig_width*fig_height_scale))
fig, (ax1, ax2) = plt.subplots(ncols=2, subplot_kw=dict(projection='polar'), gridspec_kw={'wspace': 0.65, 'hspace': 0,
                                                                                          'top': 1., 'bottom': 0., 'left': 0., 'right': 1.}, figsize=(fig_width_p*2, fig_width_p))
ax1 = plt.subplot(121, polar=True)
ax1 = radar_plot(fs_pa_raw_sm, classifiers, "auc", selected_classifiers, ax1)
ax1.legend_.remove()
ax2 = plt.subplot(122, polar=True)
ax2 = radar_plot(fs_pa_mrm_sm, classifiers, "auc", selected_classifiers, ax2)
ax2.legend_.remove()
#handles, labels = ax.get_legend_handles_labels()
ax2.legend(loc='center left', bbox_to_anchor=(1.21, 0.5))
# %%
# Boxplots
fig_height_scale = 1.2
set_style()
fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
#ax1 = boxplot_filter(fs_pa_raw, classifiers, "AUC", (fig_width,gr*fig_width), selected_classifiers, ax1)
# ax1.legend_.remove()
# ax1.set_xticklabels([])
# ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(fs_pa_raw_sm, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(fs_pa_raw_sm, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax2.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.2)
#plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_raw_sm.png",bbox_inches="tight", dpi=1000)
# %%
# For detailed analysis
mean_tables(fs_pa_raw_sm, classifiers, "AUC", selected_classifiers)
mean_tables(fs_pa_raw_nsm, classifiers, "Sensitivity", selected_classifiers)
# MRM -----------------------------------------------------------------
# ----------------- Effect of Standardization + smote -----------------
# %%
fig_height_scale = 1.2
set_style()
fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax2 = boxplot_filter(fs_pa_mrm_sm, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(fs_pa_mrm_sm, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.2)
#plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_mrm_sm.png", bbox_inches="tight", dpi=1000)
# %%
# For detailed analysis
mean_tables(fs_pa_mrm_sm, classifiers, "Sensitivity", selected_classifiers)

# %%
# MRM --------------------------------------------------------------------
# ----------------- Effect of Standardization + NO smote -----------------
fig_height_scale = 1.8
set_style()
order = ['mRMR', 'ReliefF', 'Chi-Square', 'Fisher-Score',
         'Info Gain', 'Gini Index', 'CFS', 'FCBF', 'Boruta', 'All']
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(fs_pa_mrm_nsm, classifiers, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(fs_pa_mrm_nsm, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(fs_pa_mrm_nsm, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.135)
#plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_mrm_nsm.png",bbox_inches="tight", dpi=1000)
# %%
# For detailed analysis
metric = "Sensitivity"
a, a_ = mean_tables(fs_pa_mrm_sm, classifiers, metric, selected_classifiers)
b, b_ = mean_tables(fs_pa_raw_sm, classifiers, metric, selected_classifiers)

a.iloc[:, :-1].subtract(b.iloc[:, :-1])
a_.iloc[:, :-1].subtract(b_.iloc[:, :-1])

# %%
# Confirm effect of standardization preprocessing with t-test
# variable to test
var = "auc"
# mrm
out1 = data_tables(fs_pa_raw_sm, classifiers, var)
# mrm_log
out2 = data_tables(fs_pa_mrm_sm, classifiers, var)

# 'mRMR', 'ReliefF', 'Chi-Square', 'Fisher-Score', 'Info Gain', 'Gini Index', 'CFS', 'FCBF', 'All'
method = "Boruta"
# all
stats.wilcoxon(out1[out1["Filters"] == method][var], out2[out2["Filters"] == method][var])
# %%
# Stability analysis
threshold_feats = 50

set_rank = rank_105_625  # rank_105_625
set_subset = subset_105  # subset_105

ranked_filters = rank_rank(set_rank)
idx_fisher_list_th, idx_chi_list_th, idx_reliefF_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    ranked_filters, threshold_feats)

idx_cfs_list, idx_fcbf_list, _ = set_subset
# Determine the stability of the feature selection methods with Average Tanimoto Index
ATI_fisher = average_tanimoto_index(idx_fisher_list_th)
ATI_chi = average_tanimoto_index(idx_chi_list_th)
ATI_reliefF = average_tanimoto_index(idx_reliefF_list_th)
ATI_MIM = average_tanimoto_index(idx_mim_list_th)
ATI_gini = average_tanimoto_index(idx_gini_list_th)
ATI_mrmr = average_tanimoto_index(idx_mrmr_list_th)
ATI_cfs = average_tanimoto_index(idx_cfs_list)
ATI_fcbf = average_tanimoto_index(idx_fcbf_list)
# %%
print("Stability @ threshold " + str(threshold_feats))
print('mRMR ATI:   ', ATI_mrmr)
print('ReliefF ATI:   ', ATI_reliefF)
print('Chi-Square ATI:   ', ATI_chi)
print('Fisher Score ATI:   ', ATI_fisher)
print('Information Gain ATI:   ', ATI_MIM)
print('Gini Index ATI:   ', ATI_gini)
print('CFS ATI:   ', ATI_cfs)
print('FCBF ATI:   ', ATI_fcbf)
# %%
# MRM + LOG --------------------------------------------------------------
# ----------------- Effect of log scaling post-filter -----------------
fig_height_scale = 1.2
set_style()
fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
# ax1 = boxplot_filter(fs_pa_mrm_nsm, classifiers, "AUC", selected_classifiers, ax1)
# ax1.legend_.remove()
# ax1.set_xticklabels([])
# ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(fs_pa_mrm_plog_sm, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(fs_pa_mrm_plog_sm, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax2.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.2)
#plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_mrm_plog_sm.png",bbox_inches="tight", dpi=1000)
# %%
# MRM + LOG -----------------------------------------------------------------------
# ----------------- Effect of Standardization + pre-filter Normalization -----------------
# %%
fig_height_scale = 1.2
set_style()
fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
#ax1 = boxplot_filter(fs_pa_mrm, classifiers, "AUC", (fig_width,gr*fig_width), selected_classifiers, ax1)
# ax1.legend_.remove()
# ax1.set_xticklabels([])
# ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(fs_pa_mrm_log_sm, classifiers, "Sensitivity",
                     selected_classifiers, ax2, ordering=order)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(fs_pa_mrm_log_sm, classifiers, "Specificity",
                     selected_classifiers, ax3, ordering=order)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax2.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.2)
#plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_mrm_log_sm.png",bbox_inches="tight", dpi=1000)
# %%
# MRM + LOG -------------------------------------------------------------------
# ----------------- Effect of pre + post Standardization and Normalization -----------------
# %%
fig_height_scale = 1.2
set_style()
fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
#ax1 = boxplot_filter(fs_pa_mrm, classifiers, "AUC", (fig_width,gr*fig_width), selected_classifiers, ax1)
# ax1.legend_.remove()
# ax1.set_xticklabels([])
# ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(fs_pa_mrm_loglog_sm, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(fs_pa_mrm_loglog_sm, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax2.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.47, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.2)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_mrm_loglog_sm.png",
#             bbox_inches="tight", dpi=1000)
# %%
mean_tables(fs_pa_mrm_sm, classifiers, "Geo", selected_classifiers)
mean_tables(fs_pa_mrm_loglog_sm, classifiers, "Geo", selected_classifiers)
# %%
# ----------------- Standardization Versus Standardization-Normalized -----------------
fig_height_scale = 1.2
set_style()
metric = "AUC"
fig, (ax3, ax4) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
# ax2 = boxplot_filter(fs_pa_mrm_sm, classifiers, metric, selected_classifiers, ax2)
# ax2.legend_.remove()
# ax2.set_xticklabels([])
# ax2.get_xaxis().get_label().set_visible(False)
# ax2.set_ylabel("AUC (Standardized)")

ax3 = boxplot_filter(fs_pa_mrm_sm, classifiers, metric, selected_classifiers, ax3)
ax3.legend_.remove()
ax3.set_xticklabels([])
ax3.get_xaxis().get_label().set_visible(False)
ax3.set_ylabel(metric + " \n(Standardized)")

ax4 = boxplot_filter(fs_pa_mrm_log_sm, classifiers, metric, selected_classifiers, ax4)
ax4.legend_.remove()
ax4.set_ylabel(metric + " (Standardized \n& Normalized)")

# Put the legend out of the figure
handles, labels = ax4.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.2)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/preproc_mrm_v_mrmloglog.png",
#             bbox_inches="tight", dpi=1000)

# %%
mean_tables(fs_pa_mrm_sm, classifiers, metric, selected_classifiers)
# %%
# For detailed analysis
metric = "AUC"
a, a_ = mean_tables(fs_pa_mrm_log_sm, classifiers, metric, selected_classifiers)
b, b_ = mean_tables(fs_pa_mrm_loglog_sm, classifiers,
                    metric, selected_classifiers)
# b - a
b.iloc[:, :-1].subtract(a.iloc[:, :-1])  # if + good for b and if - good for a
b_.iloc[:, :-1].subtract(a_.iloc[:, :-1])  # if - good for b and if + good for a
# %%
# Confirm effect of normalization preprocessing with t-test
# variable to test
var = "auc"
# mrm
classifiers
out1 = data_tables(fs_pa_mrm_log_sm, classifiers, var)
out1 = out1[out1["Filters"].isin(
    ['ReliefF', 'Chi-Square', 'Fisher-Score', 'Info Gain', 'Gini Index'])]
out1
# mrm_log
out2 = data_tables(fs_pa_mrm_loglog_sm, classifiers, var)
out2 = out2[out2["Filters"].isin(
    ['ReliefF', 'Chi-Square', 'Fisher-Score', 'Info Gain', 'Gini Index'])]
out2

# 'mRMR', 'ReliefF', 'Chi-Square', 'Fisher-Score', 'Info Gain', 'Gini Index', 'CFS', 'FCBF', 'All'
method = "Info Gain"
#method = "NB"
# all
stats.wilcoxon(out1[out1["Filters"] == method][var], out2[out2["Filters"] == method][var])
# %%
################################################################################################
# Ranker threshold analysis
################################################################################################
# ----------------- Analysis Initialization variables -----------------
selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF', 'XGBoost']
new_keys = ['mRMR', 'ReliefF', 'Chi-Square', 'Fisher-Score',
            'Info Gain', 'Gini Index', 'CFS', 'FCBF', 'All']
# %%
# ----------------- Import Predictive Performance Results at different thresholds -----------------
# 5
# -------------------------

# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(5), 'rb') as f:
    fs_pa_mrm_5 = pickle.load(f)
fs_pa_mrm_5 = dict(zip(new_keys, fs_pa_mrm_5.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(5), 'rb') as f:
    fs_pa_mrm_log_log_5 = pickle.load(f)
fs_pa_mrm_log_log_5 = dict(zip(new_keys, fs_pa_mrm_log_log_5.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_5.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_5.update(new_dict)
filter_set_perf_5 = fs_pa_mrm_5
# %%
# -------------------------
# 10
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(10), 'rb') as f:
    fs_pa_mrm_10 = pickle.load(f)
fs_pa_mrm_10 = dict(zip(new_keys, fs_pa_mrm_10.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(10), 'rb') as f:
    fs_pa_mrm_log_log_10 = pickle.load(f)
fs_pa_mrm_log_log_10 = dict(zip(new_keys, fs_pa_mrm_log_log_10.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_10.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_10.update(new_dict)
filter_set_perf_10 = fs_pa_mrm_10
# %%
# -------------------------
# 25
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(25), 'rb') as f:
    fs_pa_mrm_25 = pickle.load(f)
fs_pa_mrm_25 = dict(zip(new_keys, fs_pa_mrm_25.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(25), 'rb') as f:
    fs_pa_mrm_log_log_25 = pickle.load(f)
fs_pa_mrm_log_log_25 = dict(zip(new_keys, fs_pa_mrm_log_log_25.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_25.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_25.update(new_dict)
filter_set_perf_25 = fs_pa_mrm_25
# %%
# -------------------------
# 50
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(50), 'rb') as f:
    fs_pa_mrm_50 = pickle.load(f)
fs_pa_mrm_50 = dict(zip(new_keys, fs_pa_mrm_50.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(50), 'rb') as f:
    fs_pa_mrm_log_log_50 = pickle.load(f)
fs_pa_mrm_log_log_50 = dict(zip(new_keys, fs_pa_mrm_log_log_50.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_50.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_50.update(new_dict)
filter_set_perf_50 = fs_pa_mrm_50
# -------------------------
# %%
# -------------------------
# 125
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(125), 'rb') as f:
    fs_pa_mrm_125 = pickle.load(f)
fs_pa_mrm_125 = dict(zip(new_keys, fs_pa_mrm_125.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(125), 'rb') as f:
    fs_pa_mrm_log_log_125 = pickle.load(f)
fs_pa_mrm_log_log_125 = dict(zip(new_keys, fs_pa_mrm_log_log_125.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_125.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_125.update(new_dict)
filter_set_perf_125 = fs_pa_mrm_125
# -------------------------
# %%
# -------------------------
# 250
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(250), 'rb') as f:
    fs_pa_mrm_250 = pickle.load(f)
fs_pa_mrm_250 = dict(zip(new_keys, fs_pa_mrm_250.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(250), 'rb') as f:
    fs_pa_mrm_log_log_250 = pickle.load(f)
fs_pa_mrm_log_log_250 = dict(zip(new_keys, fs_pa_mrm_log_log_250.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_250.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_250.update(new_dict)
filter_set_perf_250 = fs_pa_mrm_250
# -------------------------
# %%
# -------------------------
# 500
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(500), 'rb') as f:
    fs_pa_mrm_500 = pickle.load(f)
fs_pa_mrm_500 = dict(zip(new_keys, fs_pa_mrm_500.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(500), 'rb') as f:
    fs_pa_mrm_log_log_500 = pickle.load(f)
fs_pa_mrm_log_log_500 = dict(zip(new_keys, fs_pa_mrm_log_log_500.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_500.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_500.update(new_dict)
filter_set_perf_500 = fs_pa_mrm_500
# -------------------------
# 1000
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(1000), 'rb') as f:
    fs_pa_mrm_1000 = pickle.load(f)
fs_pa_mrm_1000 = dict(zip(new_keys, fs_pa_mrm_1000.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(1000), 'rb') as f:
    fs_pa_mrm_log_log_1000 = pickle.load(f)
fs_pa_mrm_log_log_1000 = dict(zip(new_keys, fs_pa_mrm_log_log_1000.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_1000.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_1000.update(new_dict)
filter_set_perf_1000 = fs_pa_mrm_1000
# -------------------------
# 2000
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(2000), 'rb') as f:
    fs_pa_mrm_2000 = pickle.load(f)
fs_pa_mrm_2000 = dict(zip(new_keys, fs_pa_mrm_2000.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(2000), 'rb') as f:
    fs_pa_mrm_log_log_2000 = pickle.load(f)
fs_pa_mrm_log_log_2000 = dict(zip(new_keys, fs_pa_mrm_log_log_2000.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_2000.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_2000.update(new_dict)
filter_set_perf_2000 = fs_pa_mrm_2000
# -------------------------
# 4000
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(4000), 'rb') as f:
    fs_pa_mrm_4000 = pickle.load(f)
fs_pa_mrm_4000 = dict(zip(new_keys, fs_pa_mrm_4000.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(4000), 'rb') as f:
    fs_pa_mrm_log_log_4000 = pickle.load(f)
fs_pa_mrm_log_log_4000 = dict(zip(new_keys, fs_pa_mrm_log_log_4000.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_4000.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_4000.update(new_dict)
filter_set_perf_4000 = fs_pa_mrm_4000
# -------------------------
# 5000
# -------------------------
# Get: mRMR, Fisher-Score, Info Gain, Gini-Index, from here
with open(filename + '_mrm_105_predperf_th'+str(5000), 'rb') as f:
    fs_pa_mrm_5000 = pickle.load(f)
fs_pa_mrm_5000 = dict(zip(new_keys, fs_pa_mrm_5000.values()))
# Get: ReliefF, Chi-Square, from here
with open(filename + '_mrm_log_log_105_predperf_th'+str(5000), 'rb') as f:
    fs_pa_mrm_log_log_5000 = pickle.load(f)
fs_pa_mrm_log_log_5000 = dict(zip(new_keys, fs_pa_mrm_log_log_5000.values()))
keys_not_to_del = ["ReliefF", "Chi-Square"]
new_dict = {}
for key, value in fs_pa_mrm_log_log_5000.items():
    if key in keys_not_to_del:
        new_dict[key] = value
new_dict.keys()
# Create filter stage set performance set
fs_pa_mrm_5000.update(new_dict)
filter_set_perf_5000 = fs_pa_mrm_5000
# %%
# ----------------- Graphing Initialization variables -----------------


def set_style():
    sns.set(context="paper", font='serif', style="whitegrid", rc={"xtick.bottom": True,
                                                                  "xtick.labelsize": "x-small",
                                                                  "ytick.left": True,
                                                                  "ytick.labelsize": "x-small",
                                                                  "legend.fontsize": "x-small",
                                                                  "ytick.major.size": 2,
                                                                  "xtick.major.size": 2,
                                                                  "lines.linewidth": 0.8,
                                                                  "axes.edgecolor": "black",
                                                                  "axes.linewidth": "0.8"})


# %%
# ----------------- Graphing Input Variables Preperation -----------------
# Filter set (with preprocessing procedure to be used)
# Extract mean auc results
selected_classifiers = ['SVM (lin)', 'SVM (rbf)', 'NB', 'RF']
metric = "Geo"
table_perf_5 = mean_tables(filter_set_perf_5, classifiers, metric, selected_classifiers)[0]
table_perf_10 = mean_tables(filter_set_perf_10, classifiers, metric, selected_classifiers)[0]
table_perf_25 = mean_tables(filter_set_perf_25, classifiers, metric, selected_classifiers)[0]
table_perf_50 = mean_tables(filter_set_perf_50, classifiers, metric, selected_classifiers)[0]
table_perf_125 = mean_tables(filter_set_perf_125, classifiers, metric, selected_classifiers)[0]
table_perf_250 = mean_tables(filter_set_perf_250, classifiers, metric, selected_classifiers)[0]
table_perf_500 = mean_tables(filter_set_perf_500, classifiers, metric, selected_classifiers)[0]
table_perf_1000 = mean_tables(filter_set_perf_1000, classifiers, metric, selected_classifiers)[0]
table_perf_2000 = mean_tables(filter_set_perf_2000, classifiers, metric, selected_classifiers)[0]
table_perf_4000 = mean_tables(filter_set_perf_4000, classifiers, metric, selected_classifiers)[0]
table_perf_5000 = mean_tables(filter_set_perf_5000, classifiers, metric, selected_classifiers)[0]
# Prepare data tables by labelling results wtith applied thresholds
table_perf_5["Number of features"] = int(5)
table_perf_10["Number of features"] = 10
table_perf_25["Number of features"] = 25
table_perf_50["Number of features"] = 50
table_perf_125["Number of features"] = 125
table_perf_250["Number of features"] = 250
table_perf_500["Number of features"] = 500
table_perf_1000["Number of features"] = 1000
# table_perf_2000["Number of features"] = 2000
# table_perf_4000["Number of features"] = 4000
table_perf_5000["Number of features"] = 5000
# Combine tables
mean_thrshold_table = pd.concat([table_perf_5, table_perf_10, table_perf_25,
                                 table_perf_50, table_perf_125, table_perf_250, table_perf_500, table_perf_1000, table_perf_5000], axis=0).reset_index(drop=True)
# Remove Filter methods which do not want to be tested
mean_thrshold_table = mean_thrshold_table.loc[mean_thrshold_table["Filters"].isin(
    ["mRMR", "ReliefF", "Chi-Square", "Fisher-Score", "Info Gain", "Gini Index"])]
# Melt for graphing
mean_thrshold_table_m = mean_thrshold_table.melt(
    id_vars=['Filters', 'Number of features'], var_name='Classifiers', value_name=metric)
# %%
# ----------------- Threshold predictive performance effect -----------------
height = 1.8

set_style()
thres_effect_plot = sns.catplot(data=mean_thrshold_table_m, x="Number of features", y=metric, hue="Classifiers", col="Filters",
                                kind="point", col_wrap=2, height=height, aspect=2.8/height)  # , legend = False)

# plt.legend( loc="lower right") #bbox_to_anchor=(-0.1, -0.5), ncol=len(classifiers)
#thres_effect_plot.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/thres_effect_auc.png",bbox_inches="tight", dpi=1000)
# %%
# ----------------- Threshold stability effect -----------------
threshold_feats = 50
# Standardization applied to: mRMR, Fisher-Score, Info Gain, Gini-Index
idx_fisher_list, n, n, idx_mim_list, idx_gini_list, idx_mrmr_list = rank_rank(
    mrm_rank_105_625)
# Standardization + Normalization applied to: ReliefF, Chi-Square
n, idx_chi_list, idx_reliefF_list, n, n, n = rank_rank(
    mrm_log_rank_105_625)

# Create filter set
filter_set_105_625 = idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_mim_list, idx_gini_list, idx_mrmr_list
# apply threshold to ranker indices and save to relevant list
idx_fisher_list_th, idx_chi_list_th, idx_reliefF_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    filter_set_105_625, threshold_feats)

# Determine the stability of the feature selection methods with Average Tanimoto Index
ATI_fisher = average_tanimoto_index(idx_fisher_list_th)
ATI_chi = average_tanimoto_index(idx_chi_list_th)
ATI_reliefF = average_tanimoto_index(idx_reliefF_list_th)
ATI_MIM = average_tanimoto_index(idx_mim_list_th)
ATI_gini = average_tanimoto_index(idx_gini_list_th)
ATI_mrmr = average_tanimoto_index(idx_mrmr_list_th)
# %%
print("Stability @ threshold " + str(threshold_feats))
print('mRMR ATI:   ', ATI_mrmr)
print('ReliefF ATI:   ', ATI_reliefF)
print('Chi-Square ATI:   ', ATI_chi)
print('Fisher Score ATI:   ', ATI_fisher)
print('Information Gain ATI:   ', ATI_MIM)
print('Gini Index ATI:   ', ATI_gini)
# %%
################################################################################################
# Ensemble analysis
################################################################################################
# ----------------- Evaluate Similarity of filter algorithms -----------------
# Standardization applied to: mRMR, Fisher-Score, Info Gain, Gini-Index
idx_fisher_list, n, n, idx_mim_list, idx_gini_list, idx_mrmr_list = rank_rank(
    mrm_rank_105_625)
# Standardization + Normalization applied to: ReliefF, Chi-Square
n, idx_chi_list, idx_reliefF_list, n, n, n = rank_rank(
    mrm_log_rank_105_625)

# Create filter set
filter_set_105_625 = {
    'Fisher-Score': idx_fisher_list,
    'Chi-squared': idx_chi_list,
    'ReliefF': idx_reliefF_list,
    'Info Gain': idx_mim_list,
    'Gini Index': idx_gini_list,
    'mRMR': idx_mrmr_list
}

# apply threshold to ranker indices and save to relevant list
# idx_fisher_list_th, idx_chi_list_th, idx_reliefF_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
#    filter_set_105_625, threshold_feats)
# %%
threshold = 5
set_list_th = rank_thres_dict(filter_set_105_625, threshold)
set_list_th
output_2 = []
for key, filt_l_1 in set_list_th.items():
    print(key+' and its length: ' + str(len(filt_l_1[0])))
    output_1 = []
    for key, filt_l_2 in set_list_th.items():
        o = intersystem_ATI(filt_l_1, filt_l_2)  # mATI measure
        output_1.append(o)
    output_2.append(output_1)
filter_names = set_list_th.keys()
set_style()
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(output_2, index=filter_names, columns=filter_names),
            annot=True, annot_kws={"size": 9})
# %%
# confirm difference between the filter outputs by determining how many genes overlap in each
# filter methods outputs
threshold = 25
set_list = filter_set_105_625
set_list_th = rank_thres_dict(set_list, threshold)
set_list_th.pop('mRMR')
#set_list_th['CFS'] = mrm_subset_105[0]
#set_list_th['FCBF'] = mrm_subset_105[1]

filter1 = 'Gini Index'  # 'Fisher-Score', 'Chi-squared', 'ReliefF', 'Info Gain', 'Gini Index', 'mRMR'
one = pd.unique(pd.DataFrame(set_list_th[filter1]).values.ravel())

filter2 = "Fisher-Score"  # 'Fisher-Score', 'Chi-squared', 'ReliefF', 'Info Gain', 'Gini Index', 'mRMR'
two = pd.unique(pd.DataFrame(set_list_th[filter2]).values.ravel())

len(np.intersect1d(one, two))

# %%
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
set_style()
fig, ax = plt.subplots(figsize=(5, 4.2))

# sns.heatmap(cmat, annot=True, xticklabels=['Faulty', 'Healthy'], cbar=False, ax=ax)
# ax.set_yticklabels(['Faulty', 'Healthy'], va='center', rotation = 90, position=(0,0.28))
mask = np.triu(np.ones_like(num_overlap_list2, dtype=bool))
mask
overlap_heatmap = sns.heatmap(pd.DataFrame(num_overlap_list2, index=filter_names, columns=filter_names),
                              annot=True, annot_kws={"size": 11}, fmt='d', cbar=True, mask=~mask.T, ax=ax)
ax.set_yticklabels(labels=filter_names, va='center')
#plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/overlap_heatmap.png",bbox_inches="tight", dpi=1000)
# %%
unique_filter_outputs_list = []
for filter_keys, filter_values in set_list_th.items():
    unique_filter_outputs, unique_counts = np.unique(
        pd.DataFrame(filter_values).values.ravel(), return_counts=True)
    unique_filter_outputs_th = unique_filter_outputs[unique_counts > 20]
    unique_filter_outputs_list.append(unique_filter_outputs_th)
unique_filter_outputs_list

combined_thres_uniq_filter_outputs = pd.DataFrame(
    unique_filter_outputs_list, index=set_list_th.keys())
combined_thres_uniq_filter_outputs.reset_index(inplace=True)
combined_thres_uniq_filter_outputs


melt_thres_uniq_filter_outputs = pd.melt(
    combined_thres_uniq_filter_outputs, id_vars="index", ignore_index=False).drop("variable", axis=1)
melt_thres_uniq_filter_outputs


def set_style():
    sns.set(context="paper", font='serif', style="whitegrid", rc={"xtick.bottom": True,
                                                                  "xtick.labelsize": "x-small",
                                                                  "ytick.left": True,
                                                                  "ytick.labelsize": "x-small",
                                                                  "legend.fontsize": "x-small",
                                                                  "ytick.major.size": 2,
                                                                  "xtick.major.size": 2,
                                                                  "lines.linewidth": 0.8,
                                                                  "axes.edgecolor": "black",
                                                                  "axes.linewidth": "0.8"})


height = 2
sns.catplot(data=melt_thres_uniq_filter_outputs, x="value", y="index",
            jitter=False, height=height, aspect=5.52/height, s=1.2)
''' Although rather pretty, the above does not tell me much regarding the similarity of the different feature selection algorithms '''
# %%
# ----------------- Evaluate Predictive Performance of -----------------
#                      Filter ensembles vs Boruta
filename = "ge_raw_6"
# Import ensmbles predictive performance at different feature thresholds vs boruta
# mrm
directory = "C:/Users/Daniel/Documents/Thesis/Python Code/Ensemble Pred Perf Results/"
with open(directory + filename + '_mrm_105_predperf_ensemble', 'rb') as f:
    ensemble_thres_mrm = pickle.load(f)
ensemble_thres_mrm.keys()
# ens
with open(directory + filename + '_ens_105_predperf_ensemble', 'rb') as f:
    ensemble_thres_ens = pickle.load(f)
ensemble_thres_ens.keys()

# mrm log log
with open(directory + filename + '_mrm_log_log_105_predperf_ensemble', 'rb') as f:
    ensemble_thres_mrm_loglog = pickle.load(f)
ensemble_thres_mrm_loglog.keys()

# append Boruta
with open(filename + '_105_predperf_boruta' + '_auto_7_001_ens_sel', 'rb') as f:
    fs_pa_bor_mrm_auto_7_001 = pickle.load(f)
# Append Boruta results
ensemble_thres_ens['Boruta'] = fs_pa_bor_mrm_auto_7_001
ensemble_thres_ens.keys()
new_keys_bor = ['Ensemble @5', 'Ensemble @10', 'Ensemble @25',
                'Ensemble @50', 'Ensemble @125', 'Ensemble @250', 'Boruta']
ensemble_thres_ens = dict(zip(new_keys_bor, ensemble_thres_ens.values()))
ensemble_thres_ens.pop('Ensemble @125')
ensemble_thres_ens.pop('Ensemble @250')
# ensemble_thres_ens.pop('Ensemble @5')
# ensemble_thres_ens.pop('Ensemble @10')
ensemble_thres_ens.keys()
# %%
# classifiers
classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}
# Graphing Initialization


def set_style():
    sns.set(context="paper", font='serif', style="white", rc={"xtick.bottom": True,
                                                              "xtick.labelsize": "x-small",
                                                              "ytick.left": True,
                                                              "ytick.labelsize": "x-small",
                                                              "legend.fontsize": "x-small",
                                                              "ytick.major.size": 2,
                                                              "xtick.major.size": 2})


fig_width = 5.52
selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF']
# %%
# -----------------  Pre + Post Standardization -----------------
fig_height_scale = 2
set_style()
order = ensemble_thres_ens.keys()
fig, (ax1) = plt.subplots(figsize=(fig_width*fig_height_scale, fig_width*gr*fig_height_scale))
ax1 = boxplot_filter(ensemble_thres_ens, classifiers, "Geomean",
                     selected_classifiers, ax1, ordering=order)
# ax1.legend_.remove()
# ax1.set_xticklabels([])
# ax1.get_xaxis().get_label().set_visible(False)
# ax2 = boxplot_filter(ensemble_thres_ens, classifiers, "Sensitivity",
#                      selected_classifiers, ax2, ordering=order)
# ax2.legend_.remove()
# ax2.set_xticklabels([])
# ax2.get_xaxis().get_label().set_visible(False)
# ax3 = boxplot_filter(ensemble_thres_ens, classifiers, "Specificity",
#                      selected_classifiers, ax3, ordering=order)
ax1.legend_.remove()
# Put the legend out of the figure
ax1.set_xlabel("First phase feature selection components")
ax1.xaxis.labelpad = 10
ax1.set_ylim(0.2, 1)
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.25)

# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/borutaVSensemble2.png",
#             bbox_inches="tight", dpi=1000)
# %%
fig_height_scale = 1.2
set_style()
fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(fig_width, gr*fig_width*fig_height_scale))
# ax1 = boxplot_filter(ensemble_thres_ens, classifiers, "AUC", selected_classifiers, ax1)
# ax1.legend_.remove()
# ax1.set_xticklabels([])
# ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(ensemble_thres_ens, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(ensemble_thres_ens, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
ax3.set_xlabel("First phase feature selection components")
ax3.xaxis.labelpad = 10
handles, labels = ax2.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.18)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/borutaVSensemble_ss.png",
#             bbox_inches="tight", dpi=1000)
# %%
input = ensemble_thres_ens['Ensemble @5']
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
input = ensemble_thres_ens['Boruta']
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
input = ensemble_thres_ens['Ensemble @50']
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
# ----------------- Pre + Post filter standardization and normalization -----------------
fig_height_scale = 1.8
set_style()

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(ensemble_thres_mrm_loglog, classifiers, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax2 = boxplot_filter(ensemble_thres_mrm_loglog, classifiers,
                     "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(ensemble_thres_mrm_loglog, classifiers,
                     "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax3.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.465, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.12)
# %%
# -----------------  Pre + Post filter standardization -----------------
fig_height_scale = 2
set_style()
order = ['ens_5', 'ens_10', 'ens_25', 'ens_50', 'ens_125', 'ens_250', 'Boruta']
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(ensemble_thres_mrm, classifiers, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax2 = boxplot_filter(ensemble_thres_mrm, classifiers, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(ensemble_thres_mrm, classifiers, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax3.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.465, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.12)
# %%
# ----------------- Ensemble + Boruta Stability Analysis  -----------------
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xfilter outputsx/'
# import filter ensemble output
with open(filter_pickle_directory + filename+'_filter_stage_105', 'rb') as f:
    filter_set_105 = pickle.load(f)
# Import boruta selected features
# n_est| iter | perc | depth | alpha
# auto, 250, 100, 7, 0.01
with open(filename + '_' + '105_list_boruta' + '_' + 'auto_7_001', 'rb') as f:
    boruta_list_auto_7_001 = pickle.load(f)
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

idx_ensemble_list_50 = idx_ensemble_list
# %%
# Select filter threshold
# -----------------
threshold_feats = 6
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

idx_ensemble_list_25 = idx_ensemble_list

# %%
# Determine the stability of the feature selection methods with Average Tanimoto Index
ATI_ens_25 = average_tanimoto_index(idx_ensemble_list_25)
ATI_ens_50 = average_tanimoto_index(idx_ensemble_list_50)
ATI_bor = average_tanimoto_index(boruta_list_auto_7_001[2])
# %%
print(filename + " stability @ threshold " + str(threshold_feats))
print('Ensemble @25 ATI:   ', ATI_ens_25)
print('Ensemble @50 ATI:   ', ATI_ens_50)
print('Boruta ATI:   ', ATI_bor)
# %%
print("Ensemble Size Confidence Interval")
size_list = []
for lst in boruta_list_auto_7_001[2]:
    size_list.append(len(lst))
print("Max - ", np.max(size_list))
print("10 - ", np.sort(size_list)[44])
print("50 - ", np.sort(size_list)[24])
print("90 - ", np.sort(size_list)[4])
print("Min - ", np.min(size_list))
print("Mean - ", np.mean(size_list))
# %%
set_list_th = {
    "Boruta": boruta_list_auto_7_001[2],
    "Ensemble @25": idx_ensemble_list_25,
    "Ensemble @50": idx_ensemble_list_50

}
# %%
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

pd.DataFrame(num_overlap_list2)
# %%
filter_names = set_list_th.keys()
set_style()
fig, ax = plt.subplots(figsize=(5, 4.2))

# sns.heatmap(cmat, annot=True, xticklabels=['Faulty', 'Healthy'], cbar=False, ax=ax)
# ax.set_yticklabels(['Faulty', 'Healthy'], va='center', rotation = 90, position=(0,0.28))
mask = np.triu(np.ones_like(num_overlap_list2, dtype=bool))
mask
overlap_heatmap = sns.heatmap(pd.DataFrame(num_overlap_list2, index=filter_names, columns=filter_names),
                              annot=True, annot_kws={"size": 11}, fmt='d', cbar=True, mask=~mask.T, ax=ax)
ax.set_yticklabels(labels=filter_names, va='center')
#plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/overlap_heatmap.png",bbox_inches="tight", dpi=1000)
# %%
''' Save features which were not selected by Ensemble to determine if they become important later'''

# %%
# ROC comparison between methods
# Compare feature selection method outputs
set = ensemble_thres_ens
fs_predictive_ability = set["Ensemble @25"]
fig_size = ((14/2.54)*gr, (14/2.54)*gr*3)
set_style()
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
plot_classifier_nfold_rocs(
    selected_classifiers, fs_predictive_ability[4], fs_predictive_ability[5], fig_size, ax1)

fs_predictive_ability = set["Ensemble @50"]
set_style()
plot_classifier_nfold_rocs(
    selected_classifiers, fs_predictive_ability[4], fs_predictive_ability[5], fig_size, ax2)

fs_predictive_ability = set["Ensemble @125"]
set_style()
plot_classifier_nfold_rocs(
    selected_classifiers, fs_predictive_ability[4], fs_predictive_ability[5], fig_size, ax3)
# %%


# ----------------- Create Ensemble -----------------
# Ensemble parameters
ensemble_threshold = 50  # caps the number of features of each algorithm to put into ensemble

# Apply thresholding for ensemble
idx_fisher_list_th_e, idx_chi_list_th_e, idx_reliefF_list_th_e, idx_mim_list_th_e, idx_gini_list_th_e, idx_mrmr_list_th_e = rank_thres(
    filter_set_105_625, ensemble_threshold)

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
# approximate number of features in ensemble
len(idx_ensemble_list[0])
# size of ensemble list
len(idx_ensemble_list)
# Ensemble selected features
print('ensemble predictive ability')
ensemble_effectiveness = predictive_ability(
    classifiers, idx_ensemble_list, X_train, y_train, num_repeats, num_splits)

'''
# Create new normalization procedure filter set
################################################################################################
Die is nie reg nie
'''
# Standardization applied to: mRMR, Fisher-Score, Info Gain, Gini-Index
idx_fisher_list, n, n, idx_mim_list, idx_gini_list, idx_mrmr_list = rank_rank(
    mrm_rank_105_625)
# Standardization + Normalization applied to: ReliefF, Chi-Square
n, idx_chi_list, idx_reliefF_list, n, n, n = rank_rank(
    mrm_log_rank_105_625)
# filter stage algorithm set
filter_set_105_625 = idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_gini_list, idx_mim_list, idx_mrmr_list
len(filter_set_105_625[0][0])
################################################################################################
# %%


def mati_heatmap(rank_idx_list, subset_idx_list, threshold):
    #
    set_list = rank_idx_list
    for subset_idx in subset_idx_list:
        set_list.append(subset_idx)
    output_2 = []
    for i in range(0, len(set_list)):
        output_1 = []
        for j in range(0, len(set_list)):
            o = intersystem_ATI(set_list[i], set_list[j])  # mATI measure
            output_1.append(o)
        output_2.append(output_1)
    filter_names = ['Fisher-Score', 'Chi-Square', 'ReliefF',
                    'Information Gain', 'Gini Index', 'mRMR']
    set_style()
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(output_2, index=filter_names, columns=filter_names),
                annot=True, annot_kws={"size": 12})


# idx_reliefF_list, idx_chi_list, idx_fisher_list, idx_mim_list, idx_gini_list, idx_mrmr_list
filter_set_105_625
mati_heatmap(filter_set_105_625, [], 5)
# %%
# ----------------- Threshold Anlysis Boxplots -----------------
selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF']
selected_filters =
len()
fig_width = 5.52
fig_height_scale = 3
metric = "AUC"
set_style()
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
    nrows=5, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(filter_set_perf_5, classifiers, metric, selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax1.get_xaxis().get_label().set_visible(False)
ax1.set_ylabel("5")

ax2 = boxplot_filter(filter_set_perf_10, classifiers, metric, selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax2.set_ylabel("10")

ax3 = boxplot_filter(filter_set_perf_25, classifiers, metric, selected_classifiers, ax3)
ax3.legend_.remove()
ax3.set_xticklabels([])
ax3.get_xaxis().get_label().set_visible(False)
ax3.set_ylabel("25")

ax4 = boxplot_filter(filter_set_perf_50, classifiers, metric, selected_classifiers, ax4)
ax4.legend_.remove()
ax4.set_xticklabels([])
ax4.get_xaxis().get_label().set_visible(False)
ax4.set_ylabel("50")

ax5 = boxplot_filter(filter_set_perf_125, classifiers, metric, selected_classifiers, ax5)
ax5.legend_.remove()
ax5.set_ylabel("125")

# Put the legend out of the figure
handles, labels = ax4.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=len(classifiers))
fig.subplots_adjust(bottom=0.2)
#plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/mrmVnorm_preproc.png",bbox_inches="tight", dpi=1000)


# %%
# Average number of selected features by subset methods
idx_cfs_list, idx_fcbf_list = mrm_subset_105  # mrm_log_subset_105
n_sum = 0
for list in idx_cfs_list:
    n = len(list)
    n_sum += n
n_sum/50
# ----------------- Comparison of filter method outputs -----------------

threshold_feats = 125
# ranker scores
rank = rank_105_625
rank_m = mrm_rank_105_625
rank_ml = mrm_log_rank_105_625
# subset indices
subset = subset_105
subset_m = mrm_subset_105
subset_ml = mrm_log_subset_105
# %%
# rank ranker indices
ranked_rankers = rank_rank(rank_ml)
# apply threshold to ranker indices and save to relevant list
idx_fisher_list_th, idx_chi_list_th, idx_reliefF_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    ranked_rankers, threshold_feats)
# subset
idx_cfs_list, idx_fcbf_list = subset_ml

# select filter to test
filter = idx_fisher_list_th

list1 = pd.unique(pd.DataFrame(filter).values.ravel())
np.sort(pd.unique(pd.DataFrame(filter).values.ravel()))

# %%
# rank ranker indices
ranked_rankers = rank_rank(rank_m)
# apply threshold to ranker indices and save to relevant list
idx_fisher_list_th, idx_chi_list_th, idx_reliefF_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    ranked_rankers, threshold_feats)
# subset
idx_cfs_list, idx_fcbf_list = subset_m

# select filter to test
filter = idx_fisher_list_th

list2 = pd.unique(pd.DataFrame(filter).values.ravel())
np.sort(pd.unique(pd.DataFrame(filter).values.ravel()))
# %%
iN = sum(val in list1 for val in list2)
num_uniq_values = (len(list1) + len(list2))/2
print(iN, " of ", num_uniq_values, "(average number of unique values) are the same")
# %%
# ----------------- Filter Stability -----------------
threshold_feats = 5
# ranker scores
rank = rank_105_625
rank_m = mrm_rank_105_625
rank_ml = mrm_log_rank_105_625
# subset indices
subset = subset_105
subset_m = mrm_subset_105
subset_ml = mrm_log_subset_105
# rank ranker indices
ranked_rankers = rank_rank(mrm_log_rank_105_625)
# apply threshold to ranker indices and save to relevant list
idx_fisher_list_th, idx_chi_list_th, idx_reliefF_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    ranked_rankers, threshold_feats)
# save subsets to relevant list
idx_cfs_list, idx_fcbf_list = mrm_log_subset_105
# Determine the stability of the feature selection methods with Average Tanimoto Index
ATI_fisher = average_tanimoto_index(idx_fisher_list_th)
ATI_chi = average_tanimoto_index(idx_chi_list_th)
ATI_reliefF = average_tanimoto_index(idx_reliefF_list_th)
ATI_MIM = average_tanimoto_index(idx_mim_list_th)
ATI_gini = average_tanimoto_index(idx_gini_list_th)
ATI_mrmr = average_tanimoto_index(idx_mrmr_list_th)
ATI_cfs = average_tanimoto_index(idx_cfs_list)
ATI_fcbf = average_tanimoto_index(idx_fcbf_list)
# %%
print("Stability @ threshold " + str(threshold_feats))
print('Fisher Score ATI:   ', ATI_fisher)
print('Chi-Square ATI:   ', ATI_chi)
print('ReliefF ATI:   ', ATI_reliefF)
print('Information Gain ATI:   ', ATI_MIM)
print('Gini Index ATI:   ', ATI_gini)
print('mRMR ATI:   ', ATI_mrmr)
#print('CFS ATI:   ', ATI_cfs)
#print('FCBF ATI:   ', ATI_fcbf)
# %%


# Individual filter evaluations
################################################################################################
'''set preprocessing and filter to be evaluated'''
fs_pa = fs_pa_mrm
fs_name = 'Fisher-Score'
fs_predictive_ability = fs_pa[fs_name]

classifier_names = list(classifiers.keys())

print('Predictive Performance of:' + filename)
# Accuracy
acc = pd.DataFrame(fs_predictive_ability[1], columns=classifier_names)
print('\n'+fs_name+' Classifiers Accuracy:\n', acc)
# Sensitivity
sensitivity = pd.DataFrame(fs_predictive_ability[2], columns=classifier_names)
print('\n'+fs_name+' Classifiers Sensitivity:\n', sensitivity)
# Specificity
specificity = pd.DataFrame(fs_predictive_ability[3], columns=classifier_names)
print('\n'+fs_name+' Classifiers Specificity:\n', specificity)
# Predictions
predictions = pd.DataFrame(fs_predictive_ability[0], columns=classifier_names)
print('\n'+fs_name+' Classifiers Predictions:\n', predictions)
# %%
################################################################################################


def plot_classifier_nfold_rocs(classifiers_, fpr_list, tprs_list, figure_size, axis):
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


# %%
# ROC comparison between methods
# Compare feature selection method outputs
set = filter_set_perf_500
fs_predictive_ability = set["Fisher-Score"]
fig_size = (14/2.54, (14/2.54)*gr)
set_style()
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
plot_classifier_nfold_rocs(
    classifiers, fs_predictive_ability[4], fs_predictive_ability[5], fig_size, ax1)

fs_predictive_ability = set["Gini Index"]
set_style()
plot_classifier_nfold_rocs(
    classifiers, fs_predictive_ability[4], fs_predictive_ability[5], fig_size, ax2)

fs_predictive_ability = set["ReliefF"]
set_style()
plot_classifier_nfold_rocs(
    classifiers, fs_predictive_ability[4], fs_predictive_ability[5], fig_size, ax3)
