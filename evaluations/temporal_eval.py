'''
The following code aims to provide the evaluation results of the temporal effects on the best identified
features selection methods
'''
# %%
# Imports
# Basics
import pandas as pd
import numpy as np
import pickle
import time
import sys
import dill
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
# Evaluation functions
from eval_functions import intersystem_ATI, average_tanimoto_index, tanimoto_index, predictive_ability
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
# Data Prep functions
from sklearn.preprocessing import LabelEncoder
from median_ratio_method import geo_mean, median_ratio_standardization, median_ratio_standardization_, median_ratio_standardization_log
from sklearn.preprocessing import StandardScaler
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
from sklearn.ensemble import VotingClassifier
# Boruta
from boruta_py import BorutaPy  # forked master boruta_py
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
# Visualizations
from matplotlib import pyplot as plt
import seaborn as sns
# %%
# classes


class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

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


def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features_ind = features[selected_elements_indices]
    return reduced_features_ind

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

# Gemean


def gmean(y_true, y_predicted):
    sensitivity = recall_score(y_true, y_predicted)
    specificity = recall_score(y_true, y_predicted, pos_label=0)
    error = np.sqrt(sensitivity*specificity)
    return error


geometric_mean = make_scorer(gmean, greater_is_better=True)

# ga evaluation pipelines

# Standardization transformer definitions
mrstand = FunctionTransformer(median_ratio_standardization_)
mrstand_log = FunctionTransformer(median_ratio_standardization_log)
# Estimator Pipeline definitions
pipe_sse_SVMrbf = [('standardizer', mrstand),
                   ('scaler', StandardScaler()),
                   ('estimator', SVC(kernel="rbf", probability=True))]
pipe_sse_SVMlin = [('standardizer', mrstand),
                   ('scaler', StandardScaler()),
                   ('estimator', SVC(kernel="linear", probability=True))]
pipe_sse_KNN = [('standardizer', mrstand),
                ('scaler', StandardScaler()),
                ('estimator', KNeighborsClassifier())]
pipe_se_NB = [('standardizer', mrstand),
              ('estimator', GaussianNB())]
pipe_se_RF = [('standardizer', mrstand),
              ('estimator', RandomForestClassifier())]
# standardization, scaling, svm (rbf)
pipeline_SVMrbf = Pipeline(pipe_sse_SVMrbf)
# standardization, scaling, svm (rbf)
pipeline_SVMlin = Pipeline(pipe_sse_SVMlin)
# standardization, scaling, svm (rbf)
pipeline_KNN = Pipeline(pipe_sse_KNN)
# standardization, NB
pipeline_NB = Pipeline(pipe_se_NB)
# standardization, RF
pipeline_RF = Pipeline(pipe_se_RF)
# Combinations of estimator pipelines in voting classifier
voting_classifier_pipeline_combo = VotingClassifier(estimators=[('SVM_rbf', pipeline_SVMrbf), ('NB', pipeline_NB), ('KNN', pipeline_KNN), ('SVM_lin', pipeline_SVMlin)],
                                                    voting='soft')


def ci_plot_out(input, color, legend, axis, only_mean=False):
    scores_list = []
    for i in range(0, 50):
        scores = input[i]
        scores_list.append(scores)
    pd.DataFrame(scores_list).transform(np.sort)

    scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
    scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
    scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]

    axis.set_xlabel("Number of Generations")
    axis.set_ylabel("Cross validation Score")
    if only_mean == False:
        axis.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color=color)
        axis.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color=color)
        axis.plot(range(1, len(scores_25) + 1), scores_25, color=color, label=legend)
        axis.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
        axis.legend()
    else:
        axis.plot(range(1, len(scores_5) + 1), scores_25, color=color, label=legend)
        axis.legend()


def ci_plot(input, color, legend):
    scores_list = []
    for i in range(0, 50):
        scores = input[i]
        scores_list.append(scores)
    pd.DataFrame(scores_list).transform(np.sort)

    scores_5 = pd.DataFrame(scores_list).transform(np.sort).iloc[4]
    scores_25 = pd.DataFrame(scores_list).transform(np.sort).iloc[24]
    scores_45 = pd.DataFrame(scores_list).transform(np.sort).iloc[44]

    ax.set_xlabel("Number of Generations")
    ax.set_ylabel("Cross validation Score")
    ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':', color=color)
    ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':', color=color)
    ax.plot(range(1, len(scores_25) + 1), scores_25, color=color, label=legend)
    ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
    ax.legend()

# set visualisation style


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
############################################Split Data##########################################
# CV procedure variables
num_splits = 10  # number of folds
num_repeats = 5  # number of repeats

rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=0)
# %%
######################### Classifiers #########################
# %%
# Initialize classifiers to be used
classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=1),
    'SVM_linear': LinearSVC(dual=False),
    'SVM_rbf': SVC(kernel="rbf"),
    'GaussianNB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=1, class_weight="balanced"),
    'XGBoost': XGBClassifier(n_jobs=1)
}
classifier_names = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}
# %%
# Import dataset
'''##############################################Choose############################################'''
filename = 'ge_raw_12'
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
# Initialize variables
X_train = X
y_train = y
# %%
# initialize lists
kf_train_idxcs = []
kf_test_idxcs = []

for kf_train_index, kf_test_index in rskf.split(X, y):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)
# %%
################################################################################################
directory = "C:/Users/Daniel/Documents/Thesis/Python Code/temporal results/"
# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 7, 0.01
with open(directory+filename+'_boruta_filter_stage_105_auto_7_001_250', 'rb') as f:
    boruta_out_temp_12 = pickle.load(f)
# extract data
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out_temp_12)
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out_temp_12)
# %%
# stability
average_tanimoto_index(selected_lst)
# %%
# preditctive performance
preproc = "ens"
boruta_out_temp_12_pp = predictive_ability(
    classifiers, selected_lst, X_train, y_train, num_repeats, num_splits, preproc)
bor_selected_list_12 = selected_lst
# %%
input = boruta_out_temp_12_pp
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
with open(directory+filename+'_rfe_wrapper_stage_bor__SVM_linear_mrm_0_make_scorer(gmean)', 'rb') as f:
    rfecv_svm_12 = pickle.load(f)
with open(directory+filename+'_rfe_wrapper_stage_bor__RF_mrm_0_make_scorer(gmean)', 'rb') as f:
    rfecv_rf_12 = pickle.load(f)
# %%
input = rfecv_svm_12[0]
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

input = rfecv_rf_12[0]
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
# %%
'''
Here is where you select the cut-off point for the number of features you would like to select.
Rerun the algorithm.
And then evaluate the predictive performance of the approach.
'''
preproc = "ens"

rfecv_svm_12_pp = predictive_ability(
    classifiers, rfecv_svm_12[1], X_train, y_train, num_repeats, num_splits, preproc)
rfecv_rf_12_pp = predictive_ability(
    classifiers, rfecv_rf_12[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
input1 = rfecv_rf_12
print("Number of selected features:")
print("Boruta SVM: " + str(np.mean([n.n_features_ for n in input1[0]])
                           ) + " + " + str(np.std([n.n_features_ for n in input1[0]])))
input = input1[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
input = rfecv_rf_12_pp
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
# stability
average_tanimoto_index(rfecv_svm_12[1])
average_tanimoto_index(rfecv_rf_12[1])
# %%
# GA
# load outputs
with open(directory+'ge_raw_12_ga_rws_250_bor_uniform_0.8_random_0.05_vote_w03', 'rb') as f:
    ga_12 = dill.load(f)
# %%
# internal results
# fitness
fig, ax = plt.subplots()
ci_plot(ga_12[1], 'C4', 12)
# %%
# number of features
fig, ax = plt.subplots()
ci_plot(ga_12[2], 'C1', 12)
# %%
# predictive performance
fig, ax = plt.subplots()
ci_plot(ga_12[3], 'C3', 12)
# %%
# External results
# Predictive performance
# Binary feature position to feature indices
all_best_solutions = ga_12[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, np.array(bor_selected_list_12[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# Extract best features per fold (last iteration)
ga_thres = 1
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - ga_thres]
    fold_best_solution_list.append(fold_best_solution)
ga_12_best_indices = fold_best_solution_list
# %%
ga_12_PP_best = predictive_ability(
    classifiers, ga_12_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_12_PP_best
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
# Import dataset
'''##############################################Choose############################################'''
filename = 'ge_raw_18'
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
# Initialize variables
X_train = X
y_train = y
# %%
# initialize lists
kf_train_idxcs = []
kf_test_idxcs = []

for kf_train_index, kf_test_index in rskf.split(X, y):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)
# %%
################################################################################################
directory = "C:/Users/Daniel/Documents/Thesis/Python Code/temporal results/"
# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 7, 0.01
with open(directory+filename+'_boruta_filter_stage_105_auto_7_001_250', 'rb') as f:
    boruta_out_temp_18 = pickle.load(f)
# extract data
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out_temp_18)
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out_temp_18)
# %%
# stability
average_tanimoto_index(selected_lst)
# %%
# preditctive performance
preproc = "ens"
boruta_out_temp_18_pp = predictive_ability(
    classifiers, selected_lst, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = boruta_out_temp_18_pp
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
filename = 'ge_raw_18'
with open(directory+filename+'_rfe_wrapper_stage_bor__SVM_linear_mrm_0_make_scorer(gmean)', 'rb') as f:
    rfecv_svm_18 = pickle.load(f)
with open(directory+filename+'_rfe_wrapper_stage_bor__RF_mrm_0_make_scorer(gmean)', 'rb') as f:
    rfecv_rf_18 = pickle.load(f)
# %%
input = rfecv_svm_18[0]
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

input = rfecv_rf_18[0]
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
# %%
'''
Here is where you select the cut-off point for the number of features you would like to select.
Rerun the algorithm.
And then evaluate the predictive performance of the approach.
'''
preproc = "ens"

rfecv_svm_18_pp = predictive_ability(
    classifiers, rfecv_svm_18[1], X_train, y_train, num_repeats, num_splits, preproc)
rfecv_rf_18_pp = predictive_ability(
    classifiers, rfecv_rf_18[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
input1 = rfecv_rf_18
print("Number of selected features:")
print("Boruta SVM: " + str(np.mean([n.n_features_ for n in input1[0]])
                           ) + " + " + str(np.std([n.n_features_ for n in input1[0]])))
input = input1[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
input = rfecv_rf_18_pp
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
# stability
average_tanimoto_index(rfecv_svm_18[1])
average_tanimoto_index(rfecv_rf_18[1])
# %%
# Import dataset
'''##############################################Choose############################################'''
filename = 'ge_raw_24'
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
# Initialize variables
X_train = X
y_train = y
# %%
# initialize lists
kf_train_idxcs = []
kf_test_idxcs = []

for kf_train_index, kf_test_index in rskf.split(X, y):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)
# %%
################################################################################################
directory = "C:/Users/Daniel/Documents/Thesis/Python Code/temporal results/"
# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 7, 0.01
with open(directory+filename+'_boruta_filter_stage_105_auto_7_001_250', 'rb') as f:
    boruta_out_temp_24 = pickle.load(f)
# extract data
confirmed_df, tentative_df, selected_df = extract_boruta(boruta_out_temp_24)
confirmed_lst, tentative_lst, selected_lst = extract_boruta_list(boruta_out_temp_24)
# %%
# stability
average_tanimoto_index(selected_lst)
# %%
# preditctive performance
preproc = "ens"
boruta_out_temp_24_pp = predictive_ability(
    classifiers, selected_lst, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = boruta_out_temp_24_pp
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
filename = 'ge_raw_24'
with open(directory+filename+'_rfe_wrapper_stage_bor__SVM_linear_mrm_0_make_scorer(gmean)', 'rb') as f:
    rfecv_svm_24 = pickle.load(f)
with open(directory+filename+'_rfe_wrapper_stage_bor__RF_mrm_0_make_scorer(gmean)', 'rb') as f:
    rfecv_rf_24 = pickle.load(f)
# %%
input = rfecv_svm_24[0]
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

input = rfecv_rf_24[0]
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
# %%
'''
Here is where you select the cut-off point for the number of features you would like to select.
Rerun the algorithm.
And then evaluate the predictive performance of the approach.
'''
preproc = "ens"

rfecv_svm_24_pp = predictive_ability(
    classifiers, rfecv_svm_24[1], X_train, y_train, num_repeats, num_splits, preproc)
rfecv_rf_24_pp = predictive_ability(
    classifiers, rfecv_rf_24[1], X_train, y_train, num_repeats, num_splits, preproc)
# %%
input1 = rfecv_rf_24
print("Number of selected features:")
print("Boruta SVM: " + str(np.mean([n.n_features_ for n in input1[0]])
                           ) + " + " + str(np.std([n.n_features_ for n in input1[0]])))
input = input1[0]
top_score_list_1 = []
for i in range(0, 50):
    top_score = input[i].grid_scores_[input[i].n_features_-1]
    top_score_list_1.append(top_score)
print("\nMean of the top scoring results: " + str(np.mean(top_score_list_1)))
print("Std of the top scoring results: " + str(np.std(top_score_list_1)))
# %%
input = rfecv_rf_24_pp
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
# stability
average_tanimoto_index(rfecv_svm_24[1])
average_tanimoto_index(rfecv_rf_24[1])
