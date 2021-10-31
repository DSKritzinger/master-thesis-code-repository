'''
The following code aims to provide the evaluation results of the boruta all relevant feature selection method
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
from eval_functions import intersystem_ATI, average_tanimoto_index, tanimoto_index, predictive_ability
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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
# Visualizations
from matplotlib import pyplot as plt
import seaborn as sns
# %%
# Functions


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
    # fs_pa = fs_pa_raw_sm
    # classifiers_ = classifiers
    # data_name = "Sensitivity"
    # sel_classifiers = selected_classifiers
    # axis = ax2
    # initialize empty dataframes for input to graph
    classifier_names = list(classifiers_.keys())
    all_filter = pd.DataFrame(columns=classifier_names)
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
    # select classifiers to be displayed
    if "Filters" not in sel_classifiers:
        sel_classifiers.append("Filters")
    sel_filter = all_filter[sel_classifiers]
    sel_classifiers.pop()
    # melt all_filter dataframe for input to graph
    all_filter_m = sel_filter.melt(
        id_vars=['Filters'], var_name='Classifiers', value_name=data_name)
    ax = sns.boxplot(ax=axis, x=all_filter_m['Filters'], y=all_filter_m[data_name],
                     hue=all_filter_m['Classifiers'], order=ordering, fliersize=2, linewidth=0.8)
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
X.shape
y_categorical = labels.to_numpy().reshape(len(labels),)  # labels numpy array
# Change categorical labels to binary (controls - 0 and cases - 1)
Label_Encoder = LabelEncoder()
y = np.abs(Label_Encoder.fit_transform(y_categorical) - 1)
y
# %%
############################################Split Data##########################################
# CV procedure variables
num_splits = 10  # number of folds
num_repeats = 5  # number of repeats
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=0)

'''
Important to note the random_state of the train_test_split function as well as the random_state and splitting criteria of the RepeatedStratifiedKFold
function for future use.

These criteria are essentially the data splitting criteria.
'''
# initialize lists
kf_train_idxcs = []
kf_test_idxcs = []

for kf_train_index, kf_test_index in rskf.split(X, y):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)

# %%
################################################################################################
#   Load filter method outputs
################################################################################################
filter_pickle_directory = 'D:/Thesis_to_big_file/xboruta outputsx/'
# filter_pickle_directory = 'C:\Users\Daniel\Documents\Thesis\Python Code\temporal results'
# Pickle load feature subset lists BORUTA
# n_est| iter | perc | depth
# 500, 250, 100, 5
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_4', 'rb') as f:
    boruta_out4 = pickle.load(f)
# n_est| iter | perc | depth
# 1000, 250, 100, 5
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_5', 'rb') as f:
    boruta_out5 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 5, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_6', 'rb') as f:
    boruta_out6 = pickle.load(f)
# n_est| iter | perc | depth
# 5000, 250, 100, 5
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_7', 'rb') as f:
    boruta_out7 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# auto, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_8', 'rb') as f:
    boruta_out8 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# auto, 250, 100, 5, 0.01
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_9', 'rb') as f:
    boruta_out9 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 1000, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_10', 'rb') as f:
    boruta_out10 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 1000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_11', 'rb') as f:
    boruta_out11 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_12', 'rb') as f:
    boruta_out12 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_13', 'rb') as f:
    boruta_out13 = pickle.load(f)
# %%
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_14', 'rb') as f:
    boruta_out14 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 'auto', 500, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_15', 'rb') as f:
    boruta_out15 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 7, 0.01
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_16', 'rb') as f:
    boruta_out16 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 'auto', 500, 100, 7, 0.01
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_17', 'rb') as f:
    boruta_out17 = pickle.load(f)
# %%
# Extract info function


def extract_boruta(boruta_output):
    confirmed_list = []
    tentative_list = []
    selected_list = []
    for fold in range(0, 50):
        X_train_f = X[kf_train_idxcs[fold]]
        confirmed = pd.DataFrame(X_train_f).columns[boruta_output[0][fold].support_].to_list()
        confirmed_list.append(confirmed)
        tentative = pd.DataFrame(X_train_f).columns[boruta_output[0][fold].support_weak_].to_list()
        tentative_list.append(tentative)
        selected = confirmed.copy()
        selected.extend(tentative)
        selected_list.append(selected)
    return pd.DataFrame(confirmed_list), pd.DataFrame(tentative_list), pd.DataFrame(selected_list)


def extract_boruta_list(boruta_output):
    confirmed_list = []
    tentative_list = []
    selected_list = []
    for fold in range(0, 50):
        X_train_f = X[kf_train_idxcs[fold]]
        confirmed = pd.DataFrame(X_train_f).columns[boruta_output[0][fold].support_].to_list()
        confirmed_list.append(confirmed)
        tentative = pd.DataFrame(X_train_f).columns[boruta_output[0][fold].support_weak_].to_list()
        tentative_list.append(tentative)
        selected = confirmed.copy()
        selected.extend(tentative)
        selected_list.append(selected)
    return confirmed_list, tentative_list, selected_list


# %%
######################### Predictive Performance & Stability Eval #########################
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
    'RF': RandomForestClassifier(n_jobs=1, class_weight = "balanced"),
    'XGBoost': XGBClassifier(n_jobs=1)
}
# %%
# ------------------------------- Preporcessing Evaluation -------------------------------
# %%
# ----------------------- raw, no smote: 1000, 250, 100, 5
# n_est| iter | perc | depth
# import data
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_raw', 'rb') as f:
    boruta_out_raw = pickle.load(f)
# %%
# extract data
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out_raw)
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out_raw)

average_tanimoto_index(confirmed_lst)
# %%
preproc = 'raw'
boruta_auto_effectiveness_rn = predictive_ability(
    classifiers, selected_lst, X_train, y_train, num_repeats, num_splits, preproc)
boruta_auto_effectiveness_rn
# %%
# save results
# with open(filename + '_' + splits_string + repeats_string + '_predperf_boruta' + '_' + preproc, 'wb') as f:
#     pickle.dump(boruta_auto_effectiveness_rn, f)
# %%
# ----------------------- raw, smote: 1000, 250, 100, 5
# n_est| iter | perc | depth
# import data
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_raw', 'rb') as f:
    boruta_out_raw = pickle.load(f)
# %%
# extract data
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out_raw)
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out_raw)

average_tanimoto_index(confirmed_lst)
# %%
preproc = 'raw'
boruta_auto_effectiveness_rs = predictive_ability(
    classifiers, selected_lst, X_train, y_train, num_repeats, num_splits, preproc)
boruta_auto_effectiveness_rs
# %%
# save results
# with open(filename + '_' + splits_string + repeats_string + '_predperf_boruta' + '_' + preproc + '_smote', 'wb') as f:
#     pickle.dump(boruta_auto_effectiveness_rs, f)
# %%
# ----------------------- mrm, smote: 1000, 250, 100, 5
# n_est| iter | perc | depth
# import data
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_mrm', 'rb') as f:
    boruta_out_mrm = pickle.load(f)
# %%
# extract data
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out_mrm)
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out_mrm)

average_tanimoto_index(confirmed_lst)
# %%
preproc = 'mrm'
boruta_auto_effectiveness_ms = predictive_ability(
    classifiers, selected_lst, X_train, y_train, num_repeats, num_splits, preproc)
boruta_auto_effectiveness_ms
# %%
# save results
# with open(filename + '_' + splits_string + repeats_string + '_predperf_boruta' + '_' + preproc + '_smote', 'wb') as f:
#     pickle.dump(boruta_auto_effectiveness_ms, f)
# %%
# ----------------------- mrm-log, smote: 1000, 250, 100, 5
# n_est| iter | perc | depth
# import data
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_mrm_log', 'rb') as f:
    boruta_out_mrm_log = pickle.load(f)
boruta_out_mrm_log
# %%
# extract data
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out_mrm_log)
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out_mrm_log)

average_tanimoto_index(confirmed_lst)
# %%
preproc = 'mrm_log_log'
boruta_auto_effectiveness_mls = predictive_ability(
    classifiers, selected_lst, X_train, y_train, num_repeats, num_splits, preproc)
boruta_auto_effectiveness_mls
# %%
# save results
# with open(filename + '_' + splits_string + repeats_string + '_predperf_boruta' + '_' + preproc + '_smote', 'wb') as f:
#     pickle.dump(boruta_auto_effectiveness_mls, f)
# %%
preproc = 'mrm_log'
boruta_auto_effectiveness_mls = predictive_ability(
    classifiers, selected_lst, X_train, y_train, num_repeats, num_splits, preproc)
boruta_auto_effectiveness_mls
# %%
# save results
# with open(filename + '_' + splits_string + repeats_string + '_predperf_boruta' + '_' + preproc + '_smote', 'wb') as f:
#     pickle.dump(boruta_auto_effectiveness_mls, f)
# %%
# -----------------------------------------------------------------------------------------------------------
#                                                  PLAYGROUND
#                                                  PLAYGROUND
#                                                  PLAYGROUND
# -----------------------------------------------------------------------------------------------------------
sum(pd.DataFrame(boruta_out8[0][0].importance_history_).iloc[9] > 0)
sel_boruta = boruta_out8
i = 2
fold = 47
drop_off_arr = []
for fold in range(0,50):
    drop_off_pf = []
    for i in range(0,len(sel_boruta[0][fold].importance_history_)):
        val = sum(pd.DataFrame(sel_boruta[0][fold].importance_history_).iloc[i] > 0)
        #val = sum(np.isnan(pd.DataFrame(sel_boruta[0][fold].importance_history_).iloc[i]))
        drop_off_pf.append(val)
        #print(sum(pd.DataFrame(sel_boruta[0][fold].importance_history_).iloc[i] > 0))
    drop_off_arr.append(drop_off_pf)
arr = pd.DataFrame(drop_off_arr).drop(8)
print(hey)
fig, ax = plt.subplots()
for i in range(0,49):
    ax.plot(range(0, 250), arr.iloc[i], linestyle='-', color="C1")

fig.show()

sel_boruta[0][fold].n_features_
sum(sel_boruta[0][fold].support_)
sum(sel_boruta[0][fold].support_weak_)
sum(np.isnan(pd.DataFrame(sel_boruta[0][45].importance_history_).iloc[i]))
# sum(np.nonzero(pd.DataFrame(boruta_out6[0][45].importance_history_).iloc[i]))
pd.DataFrame(sel_boruta[0][45].importance_history_).iloc[i][~np.isnan(
    pd.DataFrame(sel_boruta[0][45].importance_history_).iloc[i])]
pd.DataFrame(sel_boruta[0][45].importance_history_).iloc[249]
# -----------------------------------------------------------------------------------------------------------
#                                                  PLAYGROUND
#                                                  PLAYGROUND
#                                                  PLAYGROUND
# -----------------------------------------------------------------------------------------------------------
# %%
# ------------------------------- RF Hyper-parameters Evaluation -------------------------------
# %%
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_4', 'rb') as f:
    boruta_out4 = pickle.load(f)
# n_est| iter | perc | depth
# 1000, 250, 100, 5
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_5', 'rb') as f:
    boruta_out5 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 5, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_6', 'rb') as f:
    boruta_out6 = pickle.load(f)
# n_est| iter | perc | depth
# 5000, 250, 100, 5
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_7', 'rb') as f:
    boruta_out7 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# auto, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_8', 'rb') as f:
    boruta_out8 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# auto, 250, 100, 5, 0.01
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_9', 'rb') as f:
    boruta_out9 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 1000, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_10', 'rb') as f:
    boruta_out10 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 1000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_11', 'rb') as f:
    boruta_out11 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_12', 'rb') as f:
    boruta_out12 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_13', 'rb') as f:
    boruta_out13 = pickle.load(f)
# %%
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_14', 'rb') as f:
    boruta_out14 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 'auto', 500, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_15', 'rb') as f:
    boruta_out15 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 7, 0.01
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_16', 'rb') as f:
    boruta_out16 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 'auto', 500, 100, 7, 0.01
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_17', 'rb') as f:
    boruta_out17 = pickle.load(f)
# %%
# Preprocessing procedure prior to predictive performance evaluation
preproc = "mrm"
# Predictive Performance Result Generation
# -------------------------------
# %%
# n_est| iter | perc | depth
# 1000, 250, 100, 5
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_5', 'rb') as f:
    boruta_out5 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 1000, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_10', 'rb') as f:
    boruta_out10 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 1000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_11', 'rb') as f:
    boruta_out11 = pickle.load(f)
# %%
# 5: 1000, 250, 100, 5, 0.05
print('5: 1000, 250, 100, 5, 0.05')
confirmed_lst_10005, tentative_lst_10005, selected_lst_10005 = extract_boruta_list(boruta_out5)
boruta_10005_effectiveness = predictive_ability(
    classifiers, confirmed_lst_10005, X_train, y_train, num_repeats, num_splits, preproc)
# 10: 1000, 250, 100, 7, 0.05
print('10: 1000, 250, 100, 7, 0.05')
confirmed_lst_10007, tentative_lst_10007, selected_lst_10007 = extract_boruta_list(boruta_out10)
boruta_10007_effectiveness = predictive_ability(
    classifiers, confirmed_lst_10007, X_train, y_train, num_repeats, num_splits, preproc)
# 11: 1000, 250, 100, 3, 0.05
print('11: 1000, 250, 100, 3, 0.05')
confirmed_lst_10003, tentative_lst_10003, selected_lst_10003 = extract_boruta_list(boruta_out11)
boruta_10003_effectiveness = predictive_ability(
    classifiers, confirmed_lst_10003, X_train, y_train, num_repeats, num_splits, preproc)
# %%
# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 5, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_6', 'rb') as f:
    boruta_out6 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# auto, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_8', 'rb') as f:
    boruta_out8 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_14', 'rb') as f:
    boruta_out14 = pickle.load(f)
# %%
# -------------------------------
# 6: 'auto', 250, 100, 5, 0.05
print("6: 'auto', 250, 100, 5, 0.05")
confirmed_lst_auto5, tentative_lst_auto5, selected_lst_auto5 = extract_boruta_list(boruta_out6)
boruta_auto5_effectiveness = predictive_ability(
    classifiers, confirmed_lst_auto5, X_train, y_train, num_repeats, num_splits, preproc)
# 8: 'auto', 250, 100, 7, 0.05
print("8: 'auto', 250, 100, 7, 0.05")
confirmed_lst_auto7, tentative_lst_auto7, selected_lst_auto7 = extract_boruta_list(boruta_out8)
boruta_auto7_effectiveness = predictive_ability(
    classifiers, confirmed_lst_auto7, X_train, y_train, num_repeats, num_splits, preproc)
# 14: 'auto', 250, 100, 3, 0.05
print("14: 'auto', 250, 100, 3, 0.05")
confirmed_lst_auto3, tentative_lst_auto3, selected_lst_auto3 = extract_boruta_list(boruta_out14)
boruta_auto3_effectiveness = predictive_ability(
    classifiers, confirmed_lst_auto3, X_train, y_train, num_repeats, num_splits, preproc)
# %%
# n_est| iter | perc | depth
# 5000, 250, 100, 5
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_7', 'rb') as f:
    boruta_out7 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 7, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_12', 'rb') as f:
    boruta_out12 = pickle.load(f)
# n_est| iter | perc | depth | alpha
# 5000, 250, 100, 3, 0.05
with open(filter_pickle_directory+filename+'_boruta_filter_stage_105_13', 'rb') as f:
    boruta_out13 = pickle.load(f)
# %%
# -------------------------------
# 7: 5000, 250, 100, 3, 0.05
print('7: 5000, 250, 100, 5, 0.05')
confirmed_lst_50005, tentative_lst_50005, selected_lst_50005 = extract_boruta_list(boruta_out7)
boruta_50005_effectiveness = predictive_ability(
    classifiers, confirmed_lst_50005, X_train, y_train, num_repeats, num_splits, preproc)
# 12: 5000, 250, 100, 5
print('12: 5000, 250, 100, 7, 0.05')
confirmed_lst_50007, tentative_lst_50007, selected_lst_50007 = extract_boruta_list(boruta_out12)
boruta_50007_effectiveness = predictive_ability(
    classifiers, confirmed_lst_50007, X_train, y_train, num_repeats, num_splits, preproc)
# 13: 5000, 250, 100, 5
print('13: 5000, 250, 100, 3, 0.05')
confirmed_lst_50003, tentative_lst_50003, selected_lst_50003 = extract_boruta_list(boruta_out13)
boruta_50003_effectiveness = predictive_ability(
    classifiers, confirmed_lst_50003, X_train, y_train, num_repeats, num_splits, preproc)
# %%
# 15: 'auto', 500, 100, 7, 0.05
print("15: 'auto', 500, 100, 7, 0.05")
confirmed_lst_15, tentative_lst_15, selected_lst_15 = extract_boruta_list(boruta_out15)
boruta_15_effectiveness_sel = predictive_ability(
    classifiers, selected_lst_15, X_train, y_train, num_repeats, num_splits, preproc)
# 16: 'auto', 250, 100, 7, 0.01
print("16: 'auto', 250, 100, 7, 0.01")
confirmed_lst_16, tentative_lst_16, selected_lst_16 = extract_boruta_list(boruta_out16)
boruta_16_effectiveness_sel = predictive_ability(
    classifiers, selected_lst_16, X_train, y_train, num_repeats, num_splits, preproc)
# %%

# %%
# Predictive Performance Result Comparison
# -------------------------------
classifier_names = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}
# %%
input = boruta_50003_effectiveness
m_auc = pd.DataFrame(auc_clf_compiler(
    classifier_names, input[4], input[5]), columns=classifier_names.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifier_names, input[4], input[5]), columns=classifier_names.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifier_names.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifier_names.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifier_names.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifier_names.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
input = boruta_50005_effectiveness
m_auc = pd.DataFrame(auc_clf_compiler(
    classifier_names, input[4], input[5]), columns=classifier_names.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifier_names, input[4], input[5]), columns=classifier_names.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifier_names.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifier_names.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifier_names.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifier_names.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms

# %%
input = boruta_50007_effectiveness
m_auc = pd.DataFrame(auc_clf_compiler(
    classifier_names, input[4], input[5]), columns=classifier_names.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifier_names, input[4], input[5]), columns=classifier_names.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifier_names.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifier_names.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifier_names.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifier_names.keys()).std(axis=0)
m_gmean = np.sqrt(m_spec*m_sens)
s_gmean = np.sqrt(s_sens*s_spec)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
              "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
ms
# %%
input = boruta_15_effectiveness_sel
m_auc = pd.DataFrame(auc_clf_compiler(
    classifier_names, input[4], input[5]), columns=classifier_names.keys()).mean(axis=0)
s_auc = pd.DataFrame(auc_clf_compiler(
    classifier_names, input[4], input[5]), columns=classifier_names.keys()).std(axis=0)
m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifier_names.keys()).mean(axis=0)
s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifier_names.keys()).std(axis=0)
m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifier_names.keys()).mean(axis=0)
s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifier_names.keys()).std(axis=0)
ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec], axis=1)
ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens", "Mean Spec", "Std Spec"]
ms_10007 = ms
ms.mean(0)

# %%
boruta = {
    "1000, 3": boruta_10003_effectiveness,
    "1000, 5": boruta_10005_effectiveness,
    "1000, 7": boruta_10007_effectiveness,
    "'Auto', 3": boruta_auto3_effectiveness,
    "'Auto', 5": boruta_auto5_effectiveness,
    "'Auto', 7": boruta_auto7_effectiveness,
    "5000, 3": boruta_50003_effectiveness,
    "5000, 5": boruta_50005_effectiveness,
    "5000, 7": boruta_50007_effectiveness
}

boruta_2 = {
"'Auto', 250, 7, 0.05": boruta_auto7_effectiveness,
"'Auto', 500, 7, 0.05": boruta_15_effectiveness,
"'Auto', 250, 7, 0.01": boruta_16_effectiveness
}

boruta_3 = {
"'Auto', 250, 7, 0.05": boruta_auto7_effectiveness_sel,
"'Auto', 500, 7, 0.05": boruta_15_effectiveness_sel,
"'Auto', 250, 7, 0.01": boruta_16_effectiveness_sel
}
# %%
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
selected_classifiers = ['KNN','SVM (lin)', 'SVM (rbf)', 'NB', 'RF', "XGBoost"]
# set golden ratio values
gr = (np.sqrt(5)-1)/2
# %%
# -----------------  RF parameter evaluation -----------------
fig_height_scale = 2
set_style()
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, gr*fig_width*fig_height_scale))
ax1 = boxplot_filter(boruta, classifier_names, "AUC", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xticklabels([])
ax2 = boxplot_filter(boruta, classifier_names, "Sensitivity", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xticklabels([])
ax2.get_xaxis().get_label().set_visible(False)
ax3 = boxplot_filter(boruta, classifier_names, "Specificity", selected_classifiers, ax3)
ax3.legend_.remove()
# Put the legend out of the figure
handles, labels = ax3.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.465, 0), loc="lower center", ncol=len(classifier_names))
fig.subplots_adjust(bottom=0.12)
# %%
a,b =mean_tables(boruta_3, classifier_names, "Sensitivity", selected_classifiers)
a
a.mean(1)
# %%
print("Stability 1 - " + str(average_tanimoto_index(confirmed_lst_10003)))
print("Stability 2 - " + str(average_tanimoto_index(confirmed_lst_10005)))
print("Stability 3 - " + str(average_tanimoto_index(confirmed_lst_10007)))
print("Stability 4 - " + str(average_tanimoto_index(confirmed_lst_auto3)))
print("Stability 5 - " + str(average_tanimoto_index(confirmed_lst_auto5)))
print("Stability 6 - " + str(average_tanimoto_index(confirmed_lst_auto7)))
print("Stability 7 - " + str(average_tanimoto_index(confirmed_lst_50003)))
print("Stability 8 - " + str(average_tanimoto_index(confirmed_lst_50005)))
print("Stability 9 - " + str(average_tanimoto_index(confirmed_lst_50007)))
# %%
print("Stability 1 - " + str(average_tanimoto_index(selected_lst_10003)))
print("Stability 2 - " + str(average_tanimoto_index(selected_lst_10005)))
print("Stability 3 - " + str(average_tanimoto_index(selected_lst_10007)))
print("Stability 4 - " + str(average_tanimoto_index(selected_lst_auto3)))
print("Stability 5 - " + str(average_tanimoto_index(selected_lst_auto5)))
print("Stability 6 - " + str(average_tanimoto_index(selected_lst_auto7)))
print("Stability 7 - " + str(average_tanimoto_index(selected_lst_50003)))
print("Stability 8 - " + str(average_tanimoto_index(selected_lst_50005)))
print("Stability 9 - " + str(average_tanimoto_index(selected_lst_50007)))
# %%
print("Stability 6 - " + str(average_tanimoto_index(confirmed_lst_auto7)))
print("Stability 10 - " + str(average_tanimoto_index(confirmed_lst_15)))
print("Stability 11 - " + str(average_tanimoto_index(confirmed_lst_16)))
# %%
print("Stability 6 - " + str(average_tanimoto_index(selected_lst_auto7)))
print("Stability 10 - " + str(average_tanimoto_index(selected_lst_15)))
print("Stability 11 - " + str(average_tanimoto_index(selected_lst_16)))
# %%
boruta_confirmed = {
    #"10003": confirmed_lst_10003,
    "10005": confirmed_lst_10005,
    "10007": confirmed_lst_10007,
    #"auto3": confirmed_lst_auto3,
    "auto5": confirmed_lst_auto5,
    "auto7": confirmed_lst_auto7,
    "50003": confirmed_lst_50003,
    "50005": confirmed_lst_50005,
    "50007": confirmed_lst_50007
}

boruta_selected = {
    "10003": selected_lst_10003,
    "10005": selected_lst_10005,
    "10007": selected_lst_10007,
    #"auto3": confirmed_lst_auto3,
    "auto5": selected_lst_auto5,
    "auto7": selected_lst_auto7,
    "50003": selected_lst_50003,
    "50005": selected_lst_50005,
    "50007": selected_lst_50007
}

boruta_confirmed_2 = {
    "auto7": selected_lst_auto7,
    #"15": selected_lst_15,
    "16": selected_lst_16
}

boruta_confirmed_2 = {
    "auto7": confirmed_lst_auto7,
    #"15": confirmed_lst_15,
    "16": confirmed_lst_16
}

boruta_confirmed_2 = {
    "auto7": tentative_lst_auto7,
    #"15": tentative_lst_15,
    "16": tentative_lst_16
}
# %%
set_list_th = boruta_confirmed_2
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
#set_style()
fig, ax = plt.subplots(figsize=(5, 4.2))

# sns.heatmap(cmat, annot=True, xticklabels=['Faulty', 'Healthy'], cbar=False, ax=ax)
# ax.set_yticklabels(['Faulty', 'Healthy'], va='center', rotation = 90, position=(0,0.28))
mask = np.triu(np.ones_like(num_overlap_list2, dtype=bool))
~mask.T
overlap_heatmap = sns.heatmap(pd.DataFrame(num_overlap_list2, index=filter_names, columns=filter_names),
            annot=True, annot_kws={"size": 10}, fmt='d', cbar=True,mask=~mask.T, ax=ax)
ax.set_yticklabels(labels = filter_names, va='center')
# %%
num_overlap_list2
# %%
'The best RF hyper-parameters were auto, 7, 0.05'
# save results
# with open(filename + '_' + splits_string + repeats_string + '_predperf_boruta' + '_' + 'auto_7_005', 'wb') as f:
#     pickle.dump(boruta_auto7_effectiveness, f)
# confirmed_lst_auto7, tentative_lst_auto7, selected_lst_auto7 = extract_boruta_list(boruta_out8)
# boruta_auto7_list = confirmed_lst_auto7, tentative_lst_auto7, selected_lst_auto7
# with open(filename + '_' + splits_string + repeats_string + '_list_boruta' + '_' + 'auto_7_005', 'wb') as f:
#     pickle.dump(boruta_auto7_list, f)
# %%
# --------------------- Regarding the best RF results
confirmed_lst_auto7
tentative_lst_auto7
selected_lst_auto7
confirmed_lst_auto7
# %%
print("Boruta Size Confidence Interval")
input = selected_lst_16
# with open("idx_boruta_list", 'wb') as f:
#     pickle.dump(selected_lst_16, f)
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
# Final predictive performance evaluation with "ens" preproccessing
preproc = "ens"
# 16: 'auto', 250, 100, 7, 0.01
print("16: 'auto', 250, 100, 7, 0.01")
confirmed_lst_16, tentative_lst_16, selected_lst_16 = extract_boruta_list(boruta_out16)
boruta_16_features = confirmed_lst_16, tentative_lst_16, selected_lst_16
# %%
# save selected features
with open(filename + '_105_list_boruta_auto_7_001', 'wb') as f:
    pickle.dump(boruta_16_features, f)
# %%
boruta_16_effectiveness = predictive_ability(
    classifiers, selected_lst_16, X_train, y_train, num_repeats, num_splits, preproc)
    # %%
# save results
# with open(filename + '_' + splits_string + repeats_string + '_predperf_boruta' + '_' + 'auto_7_001_ens_sel', 'wb') as f:
#     pickle.dump(boruta_16_effectiveness, f)

# %%
preproc = "mrm"
# 9: auto, 250, 100, 5, 0.01
print('# of selected estimators: "auto with p-value 0f 0.01"')
confirmed_lst_auto_p01, tentative_lst_auto_p01, selected_lst_auto_p01 = extract_boruta_list(
    boruta_out9)
confirmed_lst_auto_p01
boruta_auto_p01_effectiveness = predictive_ability(
    ss, confirmed_lst_auto_p01, X_train, y_train, num_repeats, num_splits, preproc)
# 8: auto, 250, 100, 7
print('# of selected estimators: "auto with depth of 7"')
confirmed_lst_auto7, tentative_lst_auto7, selected_lst_auto7 = extract_boruta_list(boruta_out8)
confirmed_lst_auto7
boruta_auto7_effectiveness = predictive_ability(
    classifiers, confirmed_lst_auto7, X_train, y_train, num_repeats, num_splits, preproc)
# 7: 5000, 250, 100, 5
print('# of selected estimators: "5000"')
confirmed_lst_5000, tentative_lst_5000, selected_lst_5000 = extract_boruta_list(boruta_out7)
confirmed_lst_5000
boruta_5000_effectiveness = predictive_ability(
    classifiers, confirmed_lst_5000, X_train, y_train, num_repeats, num_splits, preproc)
# 6: 'auto', 250, 100, 5
print('# of selected estimators: "Auto"')
confirmed_lst_auto, tentative_lst_auto, selected_lst_auto = extract_boruta_list(boruta_out6)
confirmed_lst_auto
boruta_auto_effectiveness = predictive_ability(
    classifiers, confirmed_lst_auto, X_train, y_train, num_repeats, num_splits, preproc)
# 5: 1000, 250, 100, 5
print('# of selected estimators: 1000')
confirmed_lst_1000, tentative_lst_1000, selected_lst_1000 = extract_boruta_list(boruta_out5)
boruta_1000_effectiveness = predictive_ability(
    classifiers, confirmed_lst_1000, X_train, y_train, num_repeats, num_splits, preproc)
# 4: 500, 250, 100, 5
print('# of selected estimators: 500')
confirmed_lst_500, tentative_lst_500, selected_lst_500 = extract_boruta_list(boruta_out4)
boruta_500_effectiveness = predictive_ability(
    classifiers, confirmed_lst_500, X_train, y_train, num_repeats, num_splits, preproc)
# %%
print('1: 500, 150, 100, 5')
confirmed_lst_1, tentative_lst_1, selected = extract_boruta_list(boruta_out1)
boruta_1_effectiveness = predictive_ability(
    classifiers, confirmed_lst_1, X_train, y_train, num_repeats, num_splits, preproc)
print('2: 1000, 150, 100, 5')
confirmed_lst_2, tentative_lst_2 = extract_boruta_list(boruta_out2)
boruta_2_effectiveness = predictive_ability(
    classifiers, confirmed_lst_2, X_train, y_train, num_repeats, num_splits, preproc)
print('3: 500, 200, 100, 5')
confirmed_lst_3, tentative_lst_3 = extract_boruta_list(boruta_out3)
boruta_3_effectiveness = predictive_ability(
    classifiers, confirmed_lst_3, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = boruta_auto_p01_effectiveness
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
print("Stability - " + str(average_tanimoto_index(confirmed_lst_auto_p01)))
print("Boruta Size Confidence Interval")
input = tentative_lst_auto_p01
size_list = []
for lst in input:
    size_list.append(len(lst))
print("Max - ", np.max(size_list))
print("10 - ", np.sort(size_list)[44])
print("50 - ", np.sort(size_list)[24])
print("90 - ", np.sort(size_list)[4])
print("Min - ", np.min(size_list))
# %%
input = boruta_auto7_effectiveness
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
print("Stability - " + str(average_tanimoto_index(confirmed_lst_auto7)))
print("Boruta Size Confidence Interval")
input = tentative_lst_auto7
size_list = []
for lst in input:
    size_list.append(len(lst))
print("Max - ", np.max(size_list))
print("10 - ", np.sort(size_list)[44])
print("50 - ", np.sort(size_list)[24])
print("90 - ", np.sort(size_list)[4])
print("Min - ", np.min(size_list))
# %%
input = boruta_5000_effectiveness
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
print("Stability - " + str(average_tanimoto_index(confirmed_lst_5000)))
print("Boruta Size Confidence Interval")
input = confirmed_lst_5000
size_list = []
for lst in input:
    size_list.append(len(lst))
print("Max - ", np.max(size_list))
print("10 - ", np.sort(size_list)[44])
print("50 - ", np.sort(size_list)[24])
print("90 - ", np.sort(size_list)[4])
print("Min - ", np.min(size_list))
# %%
input = boruta_auto_effectiveness
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
print("Stability - " + str(average_tanimoto_index(confirmed_lst_auto)))
print("Boruta Size Confidence Interval")
input = tentative_lst_auto  # confirmed_lst_auto
size_list = []
for lst in input:
    size_list.append(len(lst))
print("Max - ", np.max(size_list))
print("10 - ", np.sort(size_list)[44])
print("50 - ", np.sort(size_list)[24])
print("90 - ", np.sort(size_list)[4])
print("Min - ", np.min(size_list))
# %%
input = boruta_1000_effectiveness
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
print("Stability - " + str(average_tanimoto_index(confirmed_lst_1000)))
print("Boruta Size Confidence Interval")
input = confirmed_lst_1000
size_list = []
for lst in input:
    size_list.append(len(lst))
print("Max - ", np.max(size_list))
print("10 - ", np.sort(size_list)[44])
print("50 - ", np.sort(size_list)[24])
print("90 - ", np.sort(size_list)[4])
print("Min - ", np.min(size_list))
# %%
input = boruta_500_effectiveness
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
print("Stability - " + str(average_tanimoto_index(confirmed_lst_500)))
print("Boruta Size Confidence Interval")
input = confirmed_lst_500
size_list = []
for lst in input:
    size_list.append(len(lst))
print("Max - ", np.max(size_list))
print("10 - ", np.sort(size_list)[44])
print("50 - ", np.sort(size_list)[24])
print("90 - ", np.sort(size_list)[4])
print("Min - ", np.min(size_list))
# %%
input = boruta_1_effectiveness
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
input = boruta_2_effectiveness
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
input = boruta_3_effectiveness
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
# ----------------- Within fold evaluations -----------------
# 1: 500, 150, 100, 5
# Number of TENTATIVE features
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out1)
# %%
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out1)
average_tanimoto_index(selected_lst)
# %%
# tentative
num_tent = tentative_df.count(axis=1).to_frame(name="# Features")
num_tent['Folds'] = num_tent.index
# %%
# sns.barplot(data=num_tent, x="Folds", y="# Features", color='b')
# %%
print("Mean number of tentatively selected features: \n" + str(np.mean(num_tent['# Features'])))
print("Standard deviation of number of tentatively selected features: \n" +
      str(np.std(num_tent['# Features'])))
# %%
print("Number of tentatively selected features confidence intervals:")
print("10 - " + str(num_tent['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_tent['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_tent['# Features'].transform(np.sort)[44]))
# %%
# Number of CONFIRMED features
num_conf = confirmed_df.count(axis=1).to_frame(name="# Features")
num_conf['Folds'] = num_conf.index

# %%
# sns.barplot(data=num_conf, x="Folds", y="# Features", color='b')
# %%
print("Mean number of confirmed selected features: \n" + str(np.mean(num_conf['# Features'])))
print("Standard deviation of number of confirmed selected features: \n" +
      str(np.std(num_conf['# Features'])))
# %%
print("Number of confirmed selected features confidence intervals:")
print("10 - " + str(num_conf['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_conf['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_conf['# Features'].transform(np.sort)[44]))
# %%
# 2: 1000, 150, 100, 5
# Number of TENTATIVE features
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out2)
# %%
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out2)
average_tanimoto_index(selected_lst)
# %%
# tentative
num_tent = tentative_df.count(axis=1).to_frame(name="# Features")
num_tent['Folds'] = num_tent.index
# %%
# sns.barplot(data=num_tent, x="Folds", y="# Features", color='b')
# %%
print("Mean number of tentatively selected features: \n" + str(np.mean(num_tent['# Features'])))
print("Standard deviation of number of tentatively selected features: \n" +
      str(np.std(num_tent['# Features'])))
# %%
print("Number of tentatively selected features confidence intervals:")
print("10 - " + str(num_tent['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_tent['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_tent['# Features'].transform(np.sort)[44]))
# %%
# Number of CONFIRMED features
num_conf = confirmed_df.count(axis=1).to_frame(name="# Features")
num_conf['Folds'] = num_conf.index

# %%
# sns.barplot(data=num_conf, x="Folds", y="# Features", color='b')
# %%
print("Mean number of confirmed selected features: \n" + str(np.mean(num_conf['# Features'])))
print("Standard deviation of number of confirmed selected features: \n" +
      str(np.std(num_conf['# Features'])))
# %%
print("Number of confirmed selected features confidence intervals:")
print("10 - " + str(num_conf['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_conf['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_conf['# Features'].transform(np.sort)[44]))
# %%
# 3: 500, 200, 100, 5
# Number of TENTATIVE features
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out3)
# %%
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out3)
average_tanimoto_index(selected_lst)
# %%
# tentative
num_tent = tentative_df.count(axis=1).to_frame(name="# Features")
num_tent['Folds'] = num_tent.index
# %%
# sns.barplot(data=num_tent, x="Folds", y="# Features", color='b')
# %%
print("Mean number of tentatively selected features: \n" + str(np.mean(num_tent['# Features'])))
print("Standard deviation of number of tentatively selected features: \n" +
      str(np.std(num_tent['# Features'])))
# %%
print("Number of tentatively selected features confidence intervals:")
print("10 - " + str(num_tent['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_tent['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_tent['# Features'].transform(np.sort)[44]))
# %%
# Number of CONFIRMED features
num_conf = confirmed_df.count(axis=1).to_frame(name="# Features")
num_conf['Folds'] = num_conf.index

# %%
# sns.barplot(data=num_conf, x="Folds", y="# Features", color='b')
# %%
print("Mean number of confirmed selected features: \n" + str(np.mean(num_conf['# Features'])))
print("Standard deviation of number of confirmed selected features: \n" +
      str(np.std(num_conf['# Features'])))
# %%
print("Number of confirmed selected features confidence intervals:")
print("10 - " + str(num_conf['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_conf['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_conf['# Features'].transform(np.sort)[44]))
# %%
# 4: 500, 250, 100, 5
# Number of TENTATIVE features
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out4)
# %%
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out4)
average_tanimoto_index(selected_lst)
# %%
# tentative
num_tent = tentative_df.count(axis=1).to_frame(name="# Features")
num_tent['Folds'] = num_tent.index
# %%
# sns.barplot(data=num_tent, x="Folds", y="# Features", color='b')
# %%
print("Mean number of tentatively selected features: \n" + str(np.mean(num_tent['# Features'])))
print("Standard deviation of number of tentatively selected features: \n" +
      str(np.std(num_tent['# Features'])))
# %%
print("Number of tentatively selected features confidence intervals:")
print("10 - " + str(num_tent['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_tent['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_tent['# Features'].transform(np.sort)[44]))
# %%
# Number of CONFIRMED features
num_conf = confirmed_df.count(axis=1).to_frame(name="# Features")
num_conf['Folds'] = num_conf.index

# %%
# sns.barplot(data=num_conf, x="Folds", y="# Features", color='b')
# %%
print("Mean number of confirmed selected features: \n" + str(np.mean(num_conf['# Features'])))
print("Standard deviation of number of confirmed selected features: \n" +
      str(np.std(num_conf['# Features'])))
# %%
print("Number of confirmed selected features confidence intervals:")
print("10 - " + str(num_conf['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_conf['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_conf['# Features'].transform(np.sort)[44]))
# %%
# 5: 1000, 250, 100, 5
# Number of TENTATIVE features
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out5)
# %%
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out5)
average_tanimoto_index(selected_lst)
# %%
# tentative
num_tent = tentative_df.count(axis=1).to_frame(name="# Features")
num_tent['Folds'] = num_tent.index
# %%
# sns.barplot(data=num_tent, x="Folds", y="# Features", color='b')
# %%
print("Mean number of tentatively selected features: \n" + str(np.mean(num_tent['# Features'])))
print("Standard deviation of number of tentatively selected features: \n" +
      str(np.std(num_tent['# Features'])))
# %%
print("Number of tentatively selected features confidence intervals:")
print("10 - " + str(num_tent['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_tent['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_tent['# Features'].transform(np.sort)[44]))
# %%
# Number of CONFIRMED features
num_conf = confirmed_df.count(axis=1).to_frame(name="# Features")
num_conf['Folds'] = num_conf.index

# %%
# sns.barplot(data=num_conf, x="Folds", y="# Features", color='b')
# %%
print("Mean number of confirmed selected features: \n" + str(np.mean(num_conf['# Features'])))
print("Standard deviation of number of confirmed selected features: \n" +
      str(np.std(num_conf['# Features'])))
# %%
print("Number of confirmed selected features confidence intervals:")
print("10 - " + str(num_conf['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_conf['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_conf['# Features'].transform(np.sort)[44]))
# %%
# 6: 'auto', 250, 100, 5
# Number of TENTATIVE features
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out6)
# %%
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out6)
average_tanimoto_index(selected_lst)
# %%
# tentative
num_tent = tentative_df.count(axis=1).to_frame(name="# Features")
num_tent['Folds'] = num_tent.index
# %%
# sns.barplot(data=num_tent, x="Folds", y="# Features", color='b')
# %%
print("Mean number of tentatively selected features: \n" + str(np.mean(num_tent['# Features'])))
print("Standard deviation of number of tentatively selected features: \n" +
      str(np.std(num_tent['# Features'])))
# %%
print("Number of tentatively selected features confidence intervals:")
print("10 - " + str(num_tent['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_tent['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_tent['# Features'].transform(np.sort)[44]))
# %%
# Number of CONFIRMED features
num_conf = confirmed_df.count(axis=1).to_frame(name="# Features")
num_conf['Folds'] = num_conf.index

# %%
# sns.barplot(data=num_conf, x="Folds", y="# Features", color='b')
# %%
print("Mean number of confirmed selected features: \n" + str(np.mean(num_conf['# Features'])))
print("Standard deviation of number of confirmed selected features: \n" +
      str(np.std(num_conf['# Features'])))
# %%
print("Number of confirmed selected features confidence intervals:")
print("10 - " + str(num_conf['# Features'].transform(np.sort)[4]))
print("50 - " + str(num_conf['# Features'].transform(np.sort)[24]))
print("90 - " + str(num_conf['# Features'].transform(np.sort)[44]))
# %%
