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
import pandas as pd
import numpy as np
import scipy
import math
import statistics as st
import pickle
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
# Evaluation functions
from eval_functions import average_tanimoto_index, tanimoto_index, predictive_ability, intersystem_ATI
from sklearn.metrics import auc
# Data Prep functions
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
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
# Plotting style function


def set_style():
    # set figure style
    sns.set_context("paper")
    # set the font to be serif, rather than sans
    sns.set(font='serif')
    # make the background white, and specify the specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


# %%
################################################################################################
# Functions
# rank filter method output score indices
'''
input:  ranker_score_lists = ranker filter scores (order: fisher, chi, reliefF, mim, gini, mrmr)
ouptput: ordered by score, ranker filter method indices (order: fisher, chi, reliefF, mim, gini, mrmr)
'''


def rank_rank(ranker_score_lists):
    # extract features from rank_rank() output
    fisher_score_list, chi_score_list, reliefF_score_list, mim_score_list, gini_score_list, idx_mrmr_list = ranker_score_lists
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

    return idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_gini_list, idx_mim_list, idx_mrmr_list


# apply threshold to ranker method outputs ('top-k')
'''
input:  ranker_score_lists = ranker filter scores (order: fisher, chi, reliefF, mim, gini, mrmr)
        treshold = # of genes to select
ouptput: 'top-k' ranker filter method indices (order: fisher, chi, reliefF, mim, gini, mrmr)
'''


def rank_thres(ranker_score_lists, threshold):
    rank_rank_list = rank_rank(ranker_score_lists)
    list_th_out = []
    for list in rank_rank_list:
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
# Change categorical labels to binary (controls - 1 and cases - 0)
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
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xfilter ouputx/'
# Pickle load feature subset lists RANKERS
with open(filter_pickle_directory+filename+'_rank_105_625', 'rb') as f:
    rank_105_625 = pickle.load(
        f)
with open(filter_pickle_directory+filename+'_mrm_rank_105_625', 'rb') as f:
    mrm_rank_105_625 = pickle.load(
        f)
with open(filter_pickle_directory+filename+'_mrm_log_rank_105_625', 'rb') as f:
    mrm_log_rank_105_625 = pickle.load(
        f)
# Pickle load feature subset lists SUBSET
with open(filter_pickle_directory+filename+'_subset_105', 'rb') as f:
    subset_105 = pickle.load(f)
with open(filter_pickle_directory+filename+'_mrm_subset_105', 'rb') as f:
    mrm_subset_105 = pickle.load(f)
with open(filter_pickle_directory+filename+'_mrm_log_subset_105', 'rb') as f:
    mrm_log_subset_105 = pickle.load(f)
# %%
################################################################################################
# Select preprocessing procedure to evaluate
'''##############################################Choose############################################'''
preproc = "mrm_log"  # "raw", "mrm", "mrm_log"
'''################################################################################################'''
if preproc == "raw":
    pp_proc_rank = rank_105_625
    pp_proc_subset = subset_105
elif preproc == "mrm":
    pp_proc_rank = mrm_rank_105_625
    pp_proc_subset = mrm_subset_105
elif preproc == "mrm_log":
    pp_proc_rank = mrm_log_rank_105_625
    pp_proc_subset = mrm_log_subset_105

idx_fisher_score_list, idx_chi_score_list, idx_reliefF_score_list, idx_mim_score_list, idx_gini_score_list, idx_mrmr_list = pp_proc_rank
idx_cfs_list, idx_fcbf_list = pp_proc_subset
len(pp_proc_rank)
len(pp_proc_subset)
# %%
################################################################################################
# Rank ranker methods selected features
################################################################################################
# rank feature index lists
idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_gini_list, idx_mim_list, idx_mrmr_list = rank_rank(
    pp_proc_rank)
# %%
################################################################################################
# Threshold ranker methods selected features
'''##############################################Choose############################################'''
threshold_feats = 125
'''################################################################################################'''
# selecting x number of features from ranked methods
idx_reliefF_list_th, idx_chi_list_th, idx_fisher_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    pp_proc_rank, threshold_feats)
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
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM_linear': LinearSVC(dual=False),
    'SVM_rbf': SVC(kernel="rbf"),
    'GaussianNB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}
# %%
# Evaluate Predictive Performance
# All features
'''
with open('C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Method Development/Developmental results/'+filename+'_all_effectiveness_' + splits_string + repeats_string, 'rb') as f:
    all_effectiveness_pickle = pickle.load(f)
all_effectiveness = all_effectiveness_pickle[0]
'''
# All features predictive performance evaluation
# Initialize output variables
all_subset = list(range(0, X.shape[1]))
all_subset_list = [all_subset]*(num_repeats*num_splits)

print('ALL predictive ability')
all_effectiveness = predictive_ability(
    classifiers, all_subset_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
# with open(filename + '_all_effectiveness_' + splits_string + repeats_string, 'wb') as f:
#    pickle.dump(all_effectiveness, f)
# %%
# Ranker selected features
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
# %%
# Save filter predictive performance results
repeats_string = str(num_repeats)
splits_string = str(num_splits)
threshold_string = str(threshold_feats)
with open(filename + '_' + preproc + '_' + splits_string + repeats_string + '_predperf_th' + threshold_string, 'wb') as f:
    pickle.dump(fs_pa, f)
# %%
################################################################################################
# -------------------------------------------Compare-------------------------------------------
################################################################################################
# Functions
# For inidividual filters
'''
- plot_classifier_nfold_rocs
input:  classifiers = dictionary of classifiers to be used
        fpr_list = list of fpr's from predictive performance function of a specific filter method
        tprs_list = list of tpr's from predictive performance function of a specific filter method
        figure_size = size of output figure
ouptput: ROC Curve for said filter run on all classifiers
'''


def plot_classifier_nfold_rocs(classifiers, fpr_list, tprs_list, figure_size):
    # initialize graph variables
    mean_tprs_list = []
    mean_fprs_list = []
    mean_auc_list = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=figure_size)

    for clf_key, clf in classifiers.items():

        fpr_list_df = pd.DataFrame(fpr_list, columns=classifiers.keys())[clf_key]
        tpr_list_df = pd.DataFrame(tprs_list, columns=classifiers.keys())[clf_key]
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
        ax.set_xlabel('False Positive Rate', fontsize=18)
        ax.set_ylabel('True Positive Rate', fontsize=18)
        ax.legend(loc="lower right", prop={'size': 15})
        sns.despine()


'''
- auc_compiler
input:  classifiers = dictionary of classifiers to be used
        fpr_list = list of fpr's from predictive performance function of a specific filter method
        tprs_list = list of tpr's from predictive performance function of a specific filter method
ouptput: list of auc for each fold of each classifier for said filter
'''


def auc_clf_compiler(classifiers, fpr_list, tpr_list):
    # initialize graph variables
    auc_clf_list = []
    for clf_key, clf in classifiers.items():

        fpr_list_df = pd.DataFrame(fpr_list, columns=classifiers.keys())[clf_key]
        tpr_list_df = pd.DataFrame(tpr_list, columns=classifiers.keys())[clf_key]
        # initialize classifier graph input variables
        auc_list = []

        for fold in range(0, (num_repeats*num_splits)):
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


# For comparison of filters
'''
- boxplot_filter
input:  fs_pa =  feature selection methods predictive ability outputs
        classifiers = dictionary of classifiers to be used
        data_name = results to be graphed (accuracy, sensitivity, specificity, auc)
        figure_size = size of figure to be output
ouptput: boxplot of relevant filters
'''

def boxplot_filter(fs_pa, classifiers, data_name, figure_size):
    # initialize empty dataframes for input to graph
    classifier_names = list(classifiers.keys())
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
            # extract auc for all Classifiers
            auc_clf_list = auc_clf_compiler(classifiers, fpr_list, tpr_list)
            ind_filter = pd.DataFrame(
                auc_clf_list, columns=classifier_names).assign(Filters=pa_keys)
            all_filter = all_filter.append(ind_filter, ignore_index=True)

    # melt all_filter dataframe for input to graph
    all_filter_m = all_filter.melt(
        id_vars=['Filters'], var_name='Classifiers', value_name=data_name)

    fig, ax = plt.subplots(figsize=figure_size)

    sns.boxplot(x=all_filter_m['Filters'], y=all_filter_m[data_name],
                hue=all_filter_m['Classifiers'])
    sns.despine()
    ax.set(ylim=(0, 1.04))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # Put the legend out of the figure
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")


def radar_plot(fs_pa, classifiers, data_name, figure_size):
    #data_name = "accuracy"
    # initialize empty dataframes for input to graph
    # ordered list of classifier names
    classifier_names = list(classifiers.keys())
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
            auc_clf_list = auc_clf_compiler(classifiers, fpr_list, tpr_list)
            ind_filter = pd.DataFrame(auc_clf_list, columns=classifier_names).mean(axis=0)
            mean_filter = mean_filter.append(ind_filter, ignore_index=True)
    # set data for input to graph
    data = mean_filter
    # create colour pallete for graph lines
    palette = plt.cm.get_cmap("Set2", len(data)+1)
    # initialize plot figure
    plt.figure(figsize=figure_size)

    for i in range(0, len(data)):
        # Create background
        # number of variable
        categories = classifier_names
        N = len(categories)
        # determine angle of each axis
        angles = [n/float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Initialise the radar plot
        ax = plt.subplot(111, polar=True)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, size=10)

        # Draw ylabels
        ax.set_rlabel_position(0)

        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ["0.1", "0.2", "0.3",
                                                                   "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], color="grey", size=10)
        if data_name == 'auc':
            plt.ylim(0.4, 1)
        else:
            plt.ylim(0, 1)

        colour = palette(i)

        values = data.iloc[i].values.tolist()
        values += values[:1]
        ax.plot(angles, values, color=colour, marker="o", linewidth=2,
                linestyle='solid', label=list(fs_pa.keys())[i])
        plt.legend(loc='upper left', bbox_to_anchor=( 1.02, 1))

# %%
################################################################################################
# Comparison of predictive performance
################################################################################################
# Initialize for comparison
## classifiers
classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (Linear)': LinearSVC(dual=False),
    'SVM (RBF)': SVC(kernel="rbf"),
    'GaussianNB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}
## predictive performance results
with open(filename + '_raw_105_predperf_th125', 'rb') as f:
    fs_pa_raw = pickle.load(f)
with open(filename + '_mrm_105_predperf_th125', 'rb') as f:
    fs_pa_mrm = pickle.load(f)
with open(filename + '_mrm_log_105_predperf_th125', 'rb') as f:
    fs_pa_mrm_log = pickle.load(f)
# %%
# Individual filter evaluations
################################################################################################
'''set preprocessing and filter to be evaluated'''
fs_pa = fs_pa_mrm
fs_name = 'Fisher-Score'
fs_predictive_ability = fs_pa[fs_name]

classifier_names = list(classifiers.keys())

print('Predictive Performance of:' + filename + preproc + " preprocessing")
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
# ROC comparison between methods
## Compare feature selection method outputs
fs_predictive_ability = fs_pa_raw[fs_name]
set_style()
plot_classifier_nfold_rocs(classifiers, fs_predictive_ability[4], fs_predictive_ability[5])
# %%
fs_predictive_ability = fs_pa_mrm[fs_name]
set_style()
plot_classifier_nfold_rocs(classifiers, fs_predictive_ability[4], fs_predictive_ability[5])
# %%
fs_predictive_ability = fs_pa_mrm_log[fs_name]
set_style()
plot_classifier_nfold_rocs(classifiers, fs_predictive_ability[4], fs_predictive_ability[5])
# %%
################################################################################################
# Preditve Performance: Filter method comparison
################################################################################################
# Radarplot
################################################################################################
# sensitivity
radar_plot(fs_pa, classifiers, "sensitivity", (14, 10))
# specificity

radar_plot(fs_pa, classifiers, "specificity", (14, 10))
# auc
radar_plot(fs_pa_raw, classifiers, "auc", (14, 10))

radar_plot(fs_pa_mrm, classifiers, "auc", (14, 10))

radar_plot(fs_pa_mrm_log, classifiers, "auc", (14, 10))
# %%
# Boxplot
################################################################################################
# sensitivity
fs_pa = fs_pa_mrm
boxplot_filter(fs_pa, classifiers, "sensitivity", (14, 10))
# specificity
boxplot_filter(fs_pa, classifiers, "specificity", (14, 10))
# auc
boxplot_filter(fs_pa_raw, classifiers, "auc", (14, 7))

boxplot_filter(fs_pa_mrm, classifiers, "auc", (14, 7))

boxplot_filter(fs_pa_mrm_log, classifiers, "auc", (14, 7))
# %%
##############################################Stability#########################################
threshold_feats = 25
rank = mrm_rank_105_625
subset = mrm_subset_105
idx_reliefF_list_th, idx_chi_list_th, idx_fisher_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    rank, threshold_feats)
idx_cfs_list, idx_fcbf_list = subset
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
print('CFS ATI:   ', ATI_cfs)
print('FCBF ATI:   ', ATI_fcbf)
# %%
threshold_feats = 25
rank = mrm_log_rank_105_625
subset = mrm_log_subset_105
idx_reliefF_list_th, idx_chi_list_th, idx_fisher_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    rank, threshold_feats)
idx_cfs_list, idx_fcbf_list = subset
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
print('CFS ATI:   ', ATI_cfs)
print('FCBF ATI:   ', ATI_fcbf)
# %%
threshold_feats = 25
rank = rank_105_625
subset = subset_105
idx_reliefF_list_th, idx_chi_list_th, idx_fisher_list_th, idx_mim_list_th, idx_gini_list_th, idx_mrmr_list_th = rank_thres(
    rank, threshold_feats)
idx_cfs_list, idx_fcbf_list = subset
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
print('CFS ATI:   ', ATI_cfs)
print('FCBF ATI:   ', ATI_fcbf)
# %%
###########################################Similarity###########################################
# Intersystem similarity with mATI
# Initialize heatmap input


def mati_heatmap(rank_idx_list, subset_idx_list, threshold):
    #
    set_list = rank_thres(rank_idx_list, threshold)
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
                    'Information Gain', 'Gini Index', 'mRMR', 'CFS', 'FCBF']
    set_style()
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(output_2, index=filter_names, columns=filter_names),
                annot=True, annot_kws={"size": 12})


mati_heatmap(mrm_rank_105_625, mrm_subset_105, 10)


# %%
################################################################################################
# Stability & Ouput: Filter method comparison
################################################################################################
# ---- Output Comparison
# raw
# extract

# %%

# %%
# Ensemble implementation
# Ensemble parameters
ensemble_threshold = 50  # caps the number of features of each algorithm to put into ensemble
# Initialize feature list
idx_ensemble_list = []

# Create a ensemble from each fold's features
# Apply thresholding for ensemble
idx_reliefF_list_th_e = [item[0:ensemble_threshold] for item in idx_reliefF_list]
idx_chi_list_th_e = [item[0:ensemble_threshold] for item in idx_chi_list]
idx_fisher_list_th_e = [item[0:ensemble_threshold] for item in idx_fisher_list]
idx_mim_list_th_e = [item[0:ensemble_threshold] for item in idx_mim_list]
idx_gini_list_th_e = [item[0:ensemble_threshold] for item in idx_gini_list]
idx_mrmr_list_th_e = [item[0:ensemble_threshold] for item in idx_mrmr_list]
# append features from different methods together
for i in range(0, (num_repeats*num_splits)):
    ensembled_features = np.append(idx_fisher_list_th_e[i], [
                                   idx_reliefF_list_th_e[i],
                                   idx_chi_list_th_e[i],
                                   idx_mim_list_th_e[i],
                                   idx_gini_list_th_e[i]
                                   ]
                                   )
    # remove features which are duplicated
    ensembled_features = np.array(list(dict.fromkeys(ensembled_features)))
    idx_ensemble_list.append(ensembled_features)


# approximate number of features in ensemble
len(idx_ensemble_list[0])
# size of ensemble list
len(idx_ensemble_list)
# Ensemble selected features
print('ensemble predictive ability')
ensemble_effectiveness = predictive_ability(
    classifiers, idx_ensemble_list, X_train, y_train, num_repeats, num_splits)
# %%
# SAVE RESULTS for comparison
'''
NOTE: THIS HAS TO BE FINISHED
- Firstly create code that easily saves both ranker and subset filter method results
- Then make a script (either here or a new script) which easily provides a method of a comparison between different algorithms
    * It must compare averages of different learning algorithms as well as the standard deviation of those averages
    * It must compare stability between different feature selection methods (make sure stability is correctly being taken into account)
    * For this to happen a script for results extraction must be written.

'''


# %%
# concatenate stability metrics
fs_stability = {
    'mRMR': ATI_mrmr,
    'reliefF': ATI_reliefF,
    'chi-squared': ATI_chi,
    'fisher-score': ATI_fisher,
    'CFS': ATI_cfs,
    'FCBF': ATI_fcbf
}

repeats_string = str(num_repeats)
splits_string = str(num_splits)
threshold_string = str(threshold_feats)
with open(filename+'_r' + repeats_string+'_F'+num_splits '_results_th' + threshold_string, 'wb') as f:
    pickle.dump([fs_stability, fs_pa, similarity_heatmap], f)
# %%
print(idx_fisher_list_th)
print('\n')
print(idx_chi_list_th)
print('\n')
print(idx_reliefF_list_th)
print('\n')
print(idx_mrmr_list_th)
print('\n')
print(idx_cfs_list)
print('\n')
print(idx_fcbf_list)
# %%


################################################################################################
# Comparison Rough
################################################################################################

filename = 'ge_raw_6'
repeats = 5
threshold_feats = 50
repeats_string = str(repeats)
threshold_string = str(threshold_feats)
with open(filename+'_results_th'+threshold_string+'_r' + repeats_string, 'rb') as f:
    fs_stability, fs_pa, similarity_heatmap = pickle.load(f)
# %%
classifier_names = ['KNN', 'SVM linear', 'SVM rbf',
                    'GaussianNB', 'ComplementNB', 'Random Forest', 'XGBoost']

acc_comparison = pd.DataFrame()
prec_comparison = pd.DataFrame()
sens_comparison = pd.DataFrame()
spec_comparison = pd.DataFrame()


for fsm_key, fsm in fs_pa.items():
    fs_predictive_ability = fs_pa[fsm_key]
    # Accuracy
    acc = pd.DataFrame(fs_predictive_ability[1], columns=classifier_names)
    acc_std = acc.std(axis=0)
    acc = acc.append(acc_std, ignore_index=True)

    avg_colname = fsm_key+' Accuracy'
    std_colname = fsm_key+' std (%)'
    acc_comparison[avg_colname] = acc.iloc[-2]
    acc_comparison[std_colname] = acc.iloc[-1]
    # Precision
    prec = pd.DataFrame(fs_predictive_ability[2], columns=classifier_names)
    prec_std = prec.std(axis=0)
    prec = prec.append(prec_std, ignore_index=True)

    avg_colname = fsm_key+' Precision'
    std_colname = fsm_key+' std (%)'
    prec_comparison[avg_colname] = prec.iloc[-2]
    prec_comparison[std_colname] = prec.iloc[-1]
    # Sensitivity
    sensitivity = pd.DataFrame(fs_predictive_ability[3], columns=classifier_names)
    sensitivity_std = sensitivity.std(axis=0)
    sensitivity = sensitivity.append(sensitivity_std, ignore_index=True)

    avg_colname = fsm_key+' Sensitivity'
    std_colname = fsm_key+' std (%)'
    sens_comparison[avg_colname] = sensitivity.iloc[-2]
    sens_comparison[std_colname] = sensitivity.iloc[-1]
    # Specificity
    specificity = pd.DataFrame(fs_predictive_ability[4], columns=classifier_names)
    specificity_std = specificity.std(axis=0)
    specificity = specificity.append(specificity_std, ignore_index=True)

    avg_colname = fsm_key+' Specificity'
    std_colname = fsm_key+' std (%)'
    spec_comparison[avg_colname] = specificity.iloc[-2]
    spec_comparison[std_colname] = specificity.iloc[-1]
    # %%
    print(filename+'_results_th'+threshold_string+'_r' + repeats_string)
    acc_comparison
    # %%
    prec_comparison
    # %%
    sens_comparison
    # %%
    spec_comparison
