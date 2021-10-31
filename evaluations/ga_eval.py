'''
The following code aims to provide the evaluation results of the second phase (wrapper algorithms) of the feature selection process.

Specifically for the wrapper stage development process.

This code evaluates the wrapper algorithms in terms of their:
- predictive performance: in terms of sensitivity and specificity
- stability: ATI
- feature set cardinality
- computation time
'''

# Basics
import sys
import time
import dill
import numpy as np
import pandas as pd
import pickle
# Plotting
from matplotlib import pyplot as plt
import seaborn as sns
# Metrics
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import auc
from eval_functions import intersystem_ATI, average_tanimoto_index, tanimoto_index, predictive_ability
# Preprocessing
from sklearn.preprocessing import StandardScaler
from median_ratio_method import geo_mean, median_ratio_standardization, median_ratio_standardization_, median_ratio_standardization_log
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# Learning models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
# Feature selection
from skfeature.function.statistical_based import gini_index
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import reliefF
from skfeature.function.statistical_based import chi_square
from skfeature.function.similarity_based import fisher_score
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


def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features_ind = features[selected_elements_indices]
    return reduced_features_ind

# Evaluation Metric - Geometric mean of sensitivity and specificity


def gmean(y_true, y_predicted):
    sensitivity = recall_score(y_true, y_predicted)
    specificity = recall_score(y_true, y_predicted, pos_label=0)
    error = np.sqrt(sensitivity*specificity)
    return error


geometric_mean = make_scorer(gmean, greater_is_better=True)
eval_measure = geometric_mean
# %%
# ------------------- Evaluation Pipelines -------------------
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
# Select preprocessing procedure to evaluate
'''##############################################Choose############################################'''
preproc = "ens"  # "mrm", "mrm_log"
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
    'XGBoost': XGBClassifier(n_jobs=1),
    'VotingClassifier': voting_classifier_pipeline_combo,
}
# %%
# Import first phase idx
with open('idx_boruta_list', 'rb') as f:
    idx_boruta_list = dill.load(
        f)
with open('idx_ensemble_list_25', 'rb') as f:
    idx_ensemble_list_25 = dill.load(
        f)
with open('idx_ensemble_list_8', 'rb') as f:
    idx_ensemble_list_8 = dill.load(
        f)
# %%
################################################################################################
#   Load filter method outputs
################################################################################################
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsGAx/'
# Pickle load wrapper outputs

with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_knn_10', 'rb') as f:
    wrap_prelim_exploit_knn_10 = dill.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_lin_10', 'rb') as f:
    wrap_prelim_exploit_lin_10 = dill.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_rbf_10', 'rb') as f:
    wrap_prelim_exploit_rbf_10 = dill.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_NB_10', 'rb') as f:
    wrap_prelim_exploit_nb_10 = dill.load(
        f)
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_rf_10', 'rb') as f:
    wrap_prelim_exploit_rf_10 = dill.load(
        f)
# %%
# ------------------ Exploitive ------------------
'''
    # parameters
    imp_weight = 0.3
    num_generations = 200
    sol_per_pop = 50
    num_parents_mating = np.uint8(sol_per_pop/2)
    fitness_func = fitness_func
    num_genes = len(selected_features)
    parent_selection_type = "sss"  # "sus","rank"
    keep_parents = 1
    crossover_type = "uniform"  # "single_point","two_points"
    crossover_probability = 0.8
    mutation_type = "random"
    mutation_probability = 0.05
    # mutation_percent_genes = 0.1
    gene_space = [0, 1]
 '''
# -------- Interior CV Results
# KNN
# Fitness
# %%
# Fitness
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_knn_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_knn_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_knn_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_knn_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_knn_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_knn_10[6], "C2", "Population Averages Mean")
# %%
# SVM linear
# Fitness
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_lin_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_lin_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_lin_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_lin_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_lin_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_lin_10[6], "C2", "Population Averages Mean")
# %%
# SVM rbf
# Fitness
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rbf_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rbf_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rbf_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rbf_10[5], "C2", "Population Averages Mean")
# %%
fig, ax = plt.subplots()
# Predictive Performance
ci_plot(wrap_prelim_exploit_rbf_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rbf_10[6], "C2", "Population Averages Mean")
# %%
# Naive Bayes
# Fitness
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_nb_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_nb_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_nb_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_nb_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_nb_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_nb_10[6], "C2", "Population Averages Mean")
# %%
# Random Forest
# Fitness
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rf_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rf_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rf_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rf_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rf_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rf_10[6], "C2", "Population Averages Mean")
# %%
fig, ax = plt.subplots()
# ci_plot(wrap_prelim_exploit_rf_10[3], "C1", "Random Forest")
ci_plot(wrap_prelim_exploit_nb_10[3], "C2", "Naive Bayes")
ci_plot(wrap_prelim_exploit_lin_10[3], "C3", "SVM (lin)")
ci_plot(wrap_prelim_exploit_rbf_10[3], "C4", "SVM (rbf)")
ci_plot(wrap_prelim_exploit_knn_10[3], "C5", "KNN")
# ci_plot(wrap_prelim_exploit_votes_10[3], "C6", "Voting Classifier")
# %%
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rf_10[2], "C1", "Random Forest")
# ci_plot(wrap_prelim_exploit_nb_10[2], "C2", "Naive Bayes")
# ci_plot(wrap_prelim_exploit_lin_10[2], "C3", "SVM (lin)")
# ci_plot(wrap_prelim_exploit_rbf_10[2], "C4", "SVM (rbf)")
ci_plot(wrap_prelim_exploit_knn_10[2], "C5", "KNN")
# ci_plot(wrap_prelim_exploit_votes_10[2], "C6", "Voting Classifier")
# %%
# -------- Exterior CV Results
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_rf_10[0]:
    print(i)
    pipe = pipeline_RF
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
# %%
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
rf_mrm_ga_idx_list = best_ens_idx_list
rf_mrm_ga_idx_list
# %%
rf_mrm_ga = predictive_ability(
    classifiers, rf_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = rf_mrm_ga
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

with open('idx_ensemble_list_10', 'rb') as f:
    idx_ensemble_list_10 = dill.load(
        f)

# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_nb_10[0]:
    print(i)
    pipe = pipeline_NB
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1

prelim_exploit_nb_10_solutions_list = best_solutions_list

# %%
best_solutions_list = prelim_exploit_nb_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
nb_mrm_ga_idx_list = best_ens_idx_list
# %%
nb_mrm_ga = predictive_ability(
    classifiers, nb_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = nb_mrm_ga
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
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_lin_10[0]:
    print(i)
    pipe = pipeline_SVMlin
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1

prelim_exploit_lin_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_lin_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
svmlin_mrm_ga_idx_list = best_ens_idx_list
# %%
svmlin_mrm_ga = predictive_ability(
    classifiers, svmlin_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = svmlin_mrm_ga
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
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_rbf_10[0]:
    print(i)
    pipe = pipeline_SVMrbf
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_exploit_rbf_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_rbf_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
svmrbf_mrm_ga_idx_list = best_ens_idx_list
svmrbf_mrm_ga_idx_list
# %%
svmrbf_mrm_ga = predictive_ability(
    classifiers, svmrbf_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = svmrbf_mrm_ga
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
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_knn_10[0]:
    print(i)
    pipe = pipeline_KNN
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_exploit_knn_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_knn_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
knn_mrm_ga_idx_list = best_ens_idx_list
knn_mrm_ga_idx_list
# %%
knn_mrm_ga = predictive_ability(
    classifiers, knn_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = knn_mrm_ga
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
# Output Similarity
ATI_nb_mrm_ga = average_tanimoto_index(nb_mrm_ga_idx_list)
ATI_svmlin_mrm_ga = average_tanimoto_index(svmlin_mrm_ga_idx_list)
ATI_svmrbf_mrm_ga = average_tanimoto_index(svmrbf_mrm_ga_idx_list)
ATI_knn_mrm_ga = average_tanimoto_index(knn_mrm_ga_idx_list)
ATI_rf_mrm_ga = average_tanimoto_index(rf_mrm_ga_idx_list)
print('RF ATI:   ', ATI_rf_mrm_ga)
print('NB ATI:   ', ATI_nb_mrm_ga)
print('SVM(lin) ATI:   ', ATI_svmlin_mrm_ga)
print('SVM(rbf) ATI:   ', ATI_svmrbf_mrm_ga)
print('KNN ATI:   ', ATI_knn_mrm_ga)
# %%
# SeLected feature similarity/overlap
all_rfe_overlap = {
    "Random Forest": rf_mrm_ga_idx_list,
    "SVM (lin)": svmlin_mrm_ga_idx_list,
    "SVM (rbf)": svmrbf_mrm_ga_idx_list,
    "NB": nb_mrm_ga_idx_list,
    "KNN": knn_mrm_ga_idx_list
}
# %%
set_list_th = all_rfe_overlap
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
# Ensembling the estimators in order to improve generalizability
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_votes_10', 'rb') as f:
    wrap_prelim_exploit_votes_10 = dill.load(
        f)
# %%
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_votes_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_votes_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_votes_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_votes_10[6], "C2", "Population Averages Mean")
# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_votes_10[0]:
    print(i)
    pipe = voting_classifier_pipeline_combo
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1

prelim_exploit_votes_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_votes_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
vote_mrm_ga_idx_list = best_ens_idx_list
vote_mrm_ga_idx_list
# %%
vote_mrm_ga = predictive_ability(
    classifiers, vote_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = vote_mrm_ga
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
ATI_votes_mrm_ga = average_tanimoto_index(vote_mrm_ga_idx_list)
print('Voting Classifier ATI:   ', ATI_votes_mrm_ga)
# %%
# SeLected feature similarity/overlap
all_rfe_overlap = {
    "Random Forest": rf_mrm_ga_idx_list,
    "SVM (lin)": svmlin_mrm_ga_idx_list,
    "SVM (rbf)": svmrbf_mrm_ga_idx_list,
    "NB": nb_mrm_ga_idx_list,
    "KNN": knn_mrm_ga_idx_list,
    "Vote":  vote_mrm_ga_idx_list
}
# %%
set_list_th = all_rfe_overlap
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
# %%
# Ensembling the estimators in order to improve generalizability
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_explorative_sus_votes_10', 'rb') as f:
    wrap_prelim_explore_vote_sus_10 = dill.load(
        f)
# %%
fig, ax = plt.subplots()
ci_plot(wrap_prelim_explore_vote_sus_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_explore_vote_sus_10[4], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_explore_vote_sus_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_explore_vote_sus_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_explore_vote_sus_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_explore_vote_sus_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_explore_vote_sus_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_explore_vote_sus_10[6], "C2", "Population Averages Mean")
# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_explore_vote_sus_10[0]:
    print(i)
    pipe = voting_classifier_pipeline_combo
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_explore_vote_sus_10 = best_solutions_list
# %%
best_solutions_list = prelim_explore_vote_sus_10
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
vote_explore_sus_mrm_ga_idx_list = best_ens_idx_list
vote_explore_sus_mrm_ga_idx_list
# %%
vote_explore_sus_mrm_ga = predictive_ability(
    classifiers, vote_explore_sus_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = vote_explore_sus_mrm_ga
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
ATI_vote_explore_sus = average_tanimoto_index(vote_explore_sus_mrm_ga_idx_list)
print('Voting Classifier ATI:   ', ATI_vote_explore_sus)
# %%
set_list_th = {
    'Exploitive': vote_mrm_ga_idx_list,
    'Explorative': vote_explore_sus_mrm_ga_idx_list,
}
set_list_th = set_list_th
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
pd.DataFrame(vote_explore_sus_mrm_ga_idx_list)
pd.DataFrame(vote_mrm_ga_idx_list)
# %%
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_200m01_vote_10', 'rb') as f:
    wrap_prelim_exploit_200m01_votes_10 = dill.load(
        f)
# %%
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_200m01_votes_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_200m01_votes_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_200m01_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_200m01_votes_10[5], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_votes_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_200m01_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_200m01_votes_10[6], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_votes_10[6], "C2", "Population Averages Mean")
# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_200m01_votes_10[0]:
    print(i)
    pipe = voting_classifier_pipeline_combo
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_exploit_200m01_votes_10 = best_solutions_list
# %%
best_solutions_list = prelim_exploit_200m01_votes_10
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
vote_exploit_200m01_mrm_ga_idx_list = best_ens_idx_list
vote_exploit_200m01_mrm_ga_idx_list
# %%
vote_exploit_200m01_mrm_ga = predictive_ability(
    classifiers, vote_exploit_200m01_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = vote_exploit_200m01_mrm_ga
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
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_100m01_vote_10', 'rb') as f:
    wrap_prelim_exploit_100m01_votes_10 = dill.load(
        f)
# %%
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_100m01_votes_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_100m01_votes_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_100m01_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_100m01_votes_10[5], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_votes_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_100m01_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_100m01_votes_10[6], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_votes_10[6], "C2", "Population Averages Mean")
# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_100m01_votes_10[0]:
    print(i)
    pipe = voting_classifier_pipeline_combo
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_exploit_100m01_votes_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_100m01_votes_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
vote_exploit_100m01_mrm_ga_idx_list = best_ens_idx_list
vote_exploit_100m01_mrm_ga_idx_list
# %%
vote_exploit_100m01_mrm_ga = predictive_ability(
    classifiers, vote_exploit_100m01_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = vote_exploit_100m01_mrm_ga
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
old_filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsGAx/'
with open(old_filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_rws200m005_vote_10', 'rb') as f:
    wrap_prelim_exploit_rws200m005_votes_10 = dill.load(
        f)
# %%
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[4], "C2", "Population Averages Mean")
# fig, ax = plt.subplots()
# ci_plot(wrap_prelim_exploit_votes_10[1], "C1", "Best Solutions Mean")
# ci_plot(wrap_prelim_exploit_votes_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[5], "C2", "Population Averages Mean")
# fig, ax = plt.subplots()
# ci_plot(wrap_prelim_exploit_votes_10[2], "C1", "Best Solutions Mean")
# ci_plot(wrap_prelim_exploit_votes_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[6], "C2", "Population Averages Mean")
# fig, ax = plt.subplots()
# ci_plot(wrap_prelim_exploit_votes_10[3], "C1", "Best Solutions Mean")
# ci_plot(wrap_prelim_exploit_votes_10[6], "C2", "Population Averages Mean")
# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_rws200m005_votes_10[0]:
    print(i)
    pipe = voting_classifier_pipeline_combo
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_exploit_rws200m005_votes_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_rws200m005_votes_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
vote_exploit_rws200m005_mrm_ga_idx_list = best_ens_idx_list
vote_exploit_rws200m005_mrm_ga_idx_list
# %%
vote_exploit_rws200m005_mrm_ga = predictive_ability(
    classifiers, vote_exploit_rws200m005_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = vote_exploit_rws200m005_mrm_ga
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
ATI_vote_exploit_rws200m005 = average_tanimoto_index(vote_exploit_rws200m005_mrm_ga_idx_list)
print('Voting Classifier ATI:   ', ATI_vote_exploit_rws200m005)
# %%
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_rws500m005_vote_10', 'rb') as f:
    wrap_prelim_exploit_rws500m005_votes_10 = dill.load(
        f)
# %%
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[5], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[6], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[6], "C2", "Population Averages Mean")
# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_rws500m005_votes_10[0]:
    print(i)
    pipe = voting_classifier_pipeline_combo
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_exploit_rws500m005_votes_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_rws500m005_votes_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
vote_exploit_rws500m005_mrm_ga_idx_list = best_ens_idx_list
vote_exploit_rws500m005_mrm_ga_idx_list
# %%
vote_exploit_rws500m005_mrm_ga = predictive_ability(
    classifiers, vote_exploit_rws500m005_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = vote_exploit_rws500m005_mrm_ga
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
ATI_vote_exploit_rws500m005 = average_tanimoto_index(vote_exploit_rws500m005_mrm_ga_idx_list)
print('Voting Classifier ATI:   ', ATI_vote_exploit_rws500m005)
# %%
with open(old_filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_rws200m01_vote_10', 'rb') as f:
    wrap_prelim_exploit_rws200m01_votes_10 = dill.load(
        f)
# %%
input = wrap_prelim_exploit_rws200m01_votes_10
fig, ax = plt.subplots()
ci_plot(input[1], "C1", "Best Solutions Mean")
ci_plot(input[4], "C2", "Population Averages Mean")
# fig, ax = plt.subplots()
# ci_plot(wrap_prelim_exploit_rws200m01_votes_10[1], "C1", "Best Solutions Mean")
# ci_plot(wrap_prelim_exploit_rws200m01_votes_10[4], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(input[2], "C1", "Best Solutions Mean")
ci_plot(input[5], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(input[3], "C1", "Best Solutions Mean")
ci_plot(input[6], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws200m005_votes_10[6], "C2", "Population Averages Mean")
# fig, ax = plt.subplots()
# ci_plot(wrap_prelim_exploit_votes_10[3], "C1", "Best Solutions Mean")
# ci_plot(wrap_prelim_exploit_votes_10[6], "C2", "Population Averages Mean")
# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_rws200m01_votes_10[0]:
    print(i)
    pipe = voting_classifier_pipeline_combo
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_exploit_rws200m01_votes_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_rws200m01_votes_10_solutions_list
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
vote_exploit_rws200m01_mrm_ga_idx_list = best_ens_idx_list
vote_exploit_rws200m01_mrm_ga_idx_list
# %%
vote_exploit_rws200m01_mrm_ga = predictive_ability(
    classifiers, vote_exploit_rws200m01_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = vote_exploit_rws200m01_mrm_ga
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
ATI_vote_exploit_rws200m01 = average_tanimoto_index(vote_exploit_rws200m01_mrm_ga_idx_list)
print('Voting Classifier ATI:   ', ATI_vote_exploit_rws200m01)
# %%
with open(filter_pickle_directory+'ge_raw_6_ga_wrapper_stage_prelim_exploitive_rws500m01_vote_10', 'rb') as f:
    wrap_prelim_exploit_rws500m01_votes_10 = dill.load(
        f)
# %%
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws500m01_votes_10[1], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws500m01_votes_10[4], "C2", "Population Averages Mean")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws500m01_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws500m01_votes_10[5], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[2], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[5], "C2", "Population Averages Mean")
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws500m01_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws500m01_votes_10[6], "C2", "Population Averages Mean")
fig, ax = plt.subplots()
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[3], "C1", "Best Solutions Mean")
ci_plot(wrap_prelim_exploit_rws500m005_votes_10[6], "C2", "Population Averages Mean")
# %%
best_solutions_list = []
i = 0
for instance in wrap_prelim_exploit_rws500m01_votes_10[0]:
    print(i)
    pipe = voting_classifier_pipeline_combo
    np.seterr(divide='ignore')
    best_solution = instance.best_solution()
    np.seterr(divide='warn')
    best_solutions_list.append(best_solution)
    i += 1
prelim_exploit_rws500m01_votes_10_solutions_list = best_solutions_list
# %%
best_solutions_list = prelim_exploit_rws500m01_votes_10_solutions_list
best_solutions_list[0]
best_ens_idx_list = []
for i in range(0, 50):
    res = [i for i, val in enumerate(best_solutions_list[i][0]) if val == 1]
    best_ens_idx = idx_ensemble_list_10[i][res]
    best_ens_idx_list.append(best_ens_idx)
vote_exploit_rws500m01_mrm_ga_idx_list = best_ens_idx_list
vote_exploit_rws500m01_mrm_ga_idx_list
# %%
vote_exploit_rws500m01_mrm_ga = predictive_ability(
    classifiers, vote_exploit_rws500m01_mrm_ga_idx_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = vote_exploit_rws500m01_mrm_ga
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
ATI_vote_exploit_rws500m01 = average_tanimoto_index(vote_exploit_rws500m01_mrm_ga_idx_list)
print('Voting Classifier ATI:   ', ATI_vote_exploit_rws500m01)
# %%
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Results to be written up
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# %%
'''
# ---------------------------------------------------------------------------------
                                ENSEMBLE HP-combinations
# ---------------------------------------------------------------------------------
'''
# Ensemble with all the hyper-parameter combinations
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsGAx/new/'
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_1_random_0.05_vote_th25', 'rb') as f:
    ga_250_25_u1_r005_vote_th25 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_1_random_0.1_vote_th25', 'rb') as f:
    ga_250_25_u1_r01_vote_th25 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_1_random_0.1_vote_th25', 'rb') as f:
    ga_250_50_u1_r01_vote_th25 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_1_random_0.05_vote_th25', 'rb') as f:
    ga_250_50_u1_r005_vote_th25 = dill.load(f)

with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_0.8_random_0.05_vote_th25', 'rb') as f:
    ga_250_25_u08_r005_vote_th25 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_0.8_random_0.1_vote_th25', 'rb') as f:
    ga_250_25_u08_r01_vote_th25 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_0.8_random_0.1_vote_th25', 'rb') as f:
    ga_250_50_u08_r01_vote_th25 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_0.8_random_0.05_vote_th25', 'rb') as f:
    ga_250_50_u08_r005_vote_th25 = dill.load(f)
# %%
# Extra ensemble hyper-parameter combinations to show effect of more generations
with open(filter_pickle_directory+'ge_raw_6_ga_rws_500_25_uniform_0.8_random_0.05_vote_th25', 'rb') as f:
    ga_1000_25_u08_r005_vote_th25 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_500_50_uniform_0.8_random_0.05_vote_th25', 'rb') as f:
    ga_1000_50_u08_r005_vote_th25 = dill.load(f)
# %%
# to show effect of having a 0.6 omega value in the fitness function
with open('C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsGAx/omega_effect/'+'ge_raw_6_ga_rws_250_50_uniform_0.8_random_0.1_vote_th25_w06', 'rb') as f:
    ga_250_50_u08_r01_vote_th25_w06 = dill.load(f)
with open('C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsGAx/omega_effect/'+'ge_raw_6_ga_rws_250_25_uniform_0.8_random_0.1_vote_th25_w06', 'rb') as f:
    ga_250_25_u08_r01_vote_th25_w06 = dill.load(f)
# %%
# -------------------------------- Comparison of internal results
# %%
'''Fitness'''
fig, ax = plt.subplots()
# ci_plot(ga_250_25_u1_r005_vote_th25[1], "C1", "25-1-005")
# ci_plot(ga_250_25_u1_r01_vote_th25[1], "C2", "25-1-01")
# ci_plot(ga_250_50_u1_r005_vote_th25[1], "C3", "50-1-005")
# ci_plot(ga_250_50_u1_r01_vote_th25[1], "C4", "50-1-01")
ci_plot(ga_250_50_u08_r01_vote_th25_w06[1], "C4", "50-08-01-06")
#
# ci_plot(ga_250_25_u08_r005_vote_th25[1], "C1", "25-08-005")
# ci_plot(ga_250_25_u08_r01_vote_th25[1], "C2", "25-08-01")
# ci_plot(ga_250_50_u08_r005_vote_th25[1], "C3", "50-08-005")
# ci_plot(ga_250_50_u08_r01_vote_th25[1], "C1", "50-08-01")
ci_plot(ga_250_25_u08_r01_vote_th25_w06[1], "C1", "25-08-01-06")
# %%
'''Number of features'''
fig, ax = plt.subplots()
# ci_plot(ga_250_25_u1_r005_vote_th25[2], "C1", "25-1-005")
# ci_plot(ga_250_25_u1_r01_vote_th25[2], "C2", "25-1-01")
# ci_plot(ga_250_50_u1_r005_vote_th25[2], "C3", "50-1-005")
# ci_plot(ga_250_50_u1_r01_vote_th25[2], "C4", "50-1-01")
ci_plot(ga_250_50_u08_r01_vote_th25_w06[2], "C4", "50-08-01-06")

# ci_plot(ga_250_25_u08_r005_vote_th25[2], "C1", "25-08-005")
# ci_plot(ga_250_25_u08_r01_vote_th25[2], "C2", "25-08-01")
# ci_plot(ga_250_50_u08_r005_vote_th25[2], "C3", "50-08-005")
# ci_plot(ga_250_50_u08_r01_vote_th25[2], "C4", "50-08-01")
ci_plot(ga_250_25_u08_r01_vote_th25_w06[2], "C1", "25-08-01-06")
# %%
'''Predictive Performance'''
fig, ax = plt.subplots()
# ci_plot(ga_250_25_u1_r005_vote_th25[3], "C1", "25-1-005")
# ci_plot(ga_250_25_u1_r01_vote_th25[3], "C2", "25-1-01")
# ci_plot(ga_250_50_u1_r005_vote_th25[3], "C3", "50-1-005")
# ci_plot(ga_250_50_u1_r01_vote_th25[3], "C4", "50-1-01")
ci_plot(ga_250_50_u08_r01_vote_th25_w06[3], "C4", "50-08-01-06")

# ci_plot(ga_250_25_u08_r005_vote_th25[3], "C1", "25-08-005")
# ci_plot(ga_250_25_u08_r01_vote_th25[3], "C2", "25-08-01")
# ci_plot(ga_250_50_u08_r005_vote_th25[3], "C4", "50-08-005")
# ci_plot(ga_250_50_u08_r01_vote_th25[3], "C4", "50-08-01")
ci_plot(ga_250_25_u08_r01_vote_th25_w06[3], "C1", "25-08-01-06")
# %%
iter = 1  # from last
# -------------------------------- Predictive performance
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u1_r01_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx[0])
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_25_u1_r01_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u1_r005_vote_th25_best_indices = fold_best_solution_list

# %%
ga_250_25_u1_r005_vote_th25_PP_best = predictive_ability(
    classifiers, fold_best_solution_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u1_r005_vote_th25_PP_best
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
ATI_ga_250_25_u1_r005_vote_th25 = average_tanimoto_index(ga_250_25_u1_r005_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u1_r005_vote_th25)
# --------------------------------------------------------------------------------------------------------
# Extract the `best solutions` per fold
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u1_r005_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx[0])
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_25_u1_r005_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u1_r005_vote_th25_best_indices = fold_best_solution_list

# %%
ga_250_25_u1_r005_vote_th25_PP_best = predictive_ability(
    classifiers, fold_best_solution_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u1_r005_vote_th25_PP_best
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
ATI_ga_250_25_u1_r005_vote_th25 = average_tanimoto_index(ga_250_25_u1_r005_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u1_r005_vote_th25)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u1_r01_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_50_u1_r01_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u1_r01_vote_th25_best_indices = fold_best_solution_list
# %%
ga_250_50_u1_r01_vote_th25_PP_best = predictive_ability(
    classifiers, ga_250_50_u1_r01_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u1_r01_vote_th25_PP_best
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
ATI_ga_250_50_u1_r01_vote_th25 = average_tanimoto_index(ga_250_50_u1_r01_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u1_r01_vote_th25)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u1_r005_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_50_u1_r005_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u1_r005_vote_th25_best_indices = fold_best_solution_list
# %%
ga_250_50_u1_r005_vote_th25_PP_best = predictive_ability(
    classifiers, ga_250_50_u1_r005_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u1_r005_vote_th25_PP_best
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
ATI_ga_250_50_u1_r005_vote_th25 = average_tanimoto_index(ga_250_50_u1_r005_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u1_r005_vote_th25)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u08_r005_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_25_u08_r005_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u08_r005_vote_th25_best_indices = fold_best_solution_list
# %%
ga_250_25_u08_r005_vote_th25_PP_best = predictive_ability(
    classifiers, ga_250_25_u08_r005_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u08_r005_vote_th25_PP_best
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
ATI_ga_250_25_u08_r005_vote_th25 = average_tanimoto_index(ga_250_25_u08_r005_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u08_r005_vote_th25)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u08_r01_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_25_u08_r01_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u08_r01_vote_th25_best_indices = fold_best_solution_list
# %%
ga_250_25_u08_r01_vote_th25_PP_best = predictive_ability(
    classifiers, ga_250_25_u08_r01_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u08_r01_vote_th25_PP_best
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
ATI_ga_250_25_u08_r01_vote_th25 = average_tanimoto_index(ga_250_25_u08_r01_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u08_r01_vote_th25)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u08_r01_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_50_u08_r01_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u08_r01_vote_th25_best_indices = fold_best_solution_list
# %%
ga_250_50_u08_r01_vote_th25_PP_best = predictive_ability(
    classifiers, ga_250_50_u08_r01_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u08_r01_vote_th25_PP_best
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
ATI_ga_250_50_u08_r01_vote_th25 = average_tanimoto_index(ga_250_50_u08_r01_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u08_r01_vote_th25)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u08_r005_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_50_u08_r005_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u08_r005_vote_th25_best_indices = fold_best_solution_list
# %%
ga_250_50_u08_r005_vote_th25_PP_best = predictive_ability(
    classifiers, ga_250_50_u08_r005_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u08_r005_vote_th25_PP_best
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
ATI_ga_250_50_u08_r005_vote_th25 = average_tanimoto_index(ga_250_50_u08_r005_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u08_r005_vote_th25)
'''
    --------------------------------------------------------------------------------------------------------
---------------------------------------------- ENSEMBLE 1000 ---------------------------------------------------
    -------------------------------------------------------------------------------------------------------- '''
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_1000_50_u08_r005_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::10])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 1000)
folds = range(0, 50)
gens_pp = []
for gen in generations[::10]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_1000_50_u08_r005_vote_th25_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_1000_50_u08_r005_vote_th25_best_indices = fold_best_solution_list
# %%
ga_1000_50_u08_r005_vote_th25_PP_best = predictive_ability(
    classifiers, ga_1000_50_u08_r005_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_1000_50_u08_r005_vote_th25_PP_best
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
ATI_ga_1000_50_u08_r005_vote_th25 = average_tanimoto_index(
    ga_1000_50_u08_r005_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_1000_50_u08_r005_vote_th25)
# --------------------------------------------------------------------------------------------------------
# %%
all_best_solutions = ga_1000_25_u08_r005_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1
pd.set_option('max_rows', 1000)
pd.set_option('max_columns', None)
pd.DataFrame(all_best_solutions_idx[0])
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::10])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 1000)
folds = range(0, 50)
gens_pp = []
for gen in generations[::10]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_1000_25_u08_r005_vote_th25_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_1000_25_u08_r005_vote_th25_best_indices = fold_best_solution_list
# %%
ga_1000_25_u08_r005_vote_th25_PP_best = predictive_ability(
    classifiers, ga_1000_25_u08_r005_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_1000_25_u08_r005_vote_th25_PP_best
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
ATI_ga_1000_25_u08_r005_vote_th25 = average_tanimoto_index(
    ga_1000_25_u08_r005_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_1000_25_u08_r005_vote_th25)
# %%
# Ensemble with all the hyper-parameter combinations thresheld @ 8 features per filter
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsGAx/newnew/'
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_1_random_0.05_vote_th8', 'rb') as f:
    ga_250_25_u1_r005_vote_th8 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_1_random_0.1_vote_th8', 'rb') as f:
    ga_250_25_u1_r01_vote_th8 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_1_random_0.1_vote_th8', 'rb') as f:
    ga_250_50_u1_r01_vote_th8 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_1_random_0.05_vote_th8', 'rb') as f:
    ga_250_50_u1_r005_vote_th8 = dill.load(f)

with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_0.8_random_0.05_vote_th8', 'rb') as f:
    ga_250_25_u08_r005_vote_th8 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_0.8_random_0.1_vote_th8', 'rb') as f:
    ga_250_25_u08_r01_vote_th8 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_0.8_random_0.1_vote_th8', 'rb') as f:
    ga_250_50_u08_r01_vote_th8 = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_0.8_random_0.05_vote_th8', 'rb') as f:
    ga_250_50_u08_r005_vote_th8 = dill.load(f)
# %%
# Extra ensemble hyper-parameter combinations to show effect of more generations
# with open(filter_pickle_directory+'ge_raw_6_ga_rws_500_25_uniform_0.8_random_0.05_vote_th25', 'rb') as f:
#     ga_1000_25_u08_r005_vote_th25 = dill.load(f)
# with open(filter_pickle_directory+'ge_raw_6_ga_rws_500_50_uniform_0.8_random_0.05_vote_th25', 'rb') as f:
#     ga_1000_50_u08_r005_vote_th25 = dill.load(f)
# %%
# -------------------------------- Comparison of internal results
# %%
# Fitness
fig, ax = plt.subplots()
# ci_plot(ga_250_25_u1_r005_vote_th8[1], "C1", "25-1-005")
# ci_plot(ga_250_25_u1_r01_vote_th8[1], "C2", "25-1-01")
# ci_plot(ga_250_50_u1_r005_vote_th8[1], "C3", "50-1-005")
ci_plot(ga_250_50_u1_r01_vote_th8[1], "C4", "50-1-01")
#
# ci_plot(ga_250_25_u08_r005_vote_th8[1], "C1", "25-08-005")
# ci_plot(ga_250_25_u08_r01_vote_th8[1], "C2", "25-08-01")
# ci_plot(ga_250_50_u08_r005_vote_th8[1], "C3", "50-08-005")
ci_plot(ga_250_50_u08_r01_vote_th8[1], "C1", "50-08-01")
# %%
# Number of features
fig, ax = plt.subplots()
ci_plot(ga_250_25_u1_r005_vote_th8[2], "C1", "25-1-005")
# ci_plot(ga_250_25_u1_r01_vote_th8[2], "C2", "25-1-01")
# ci_plot(ga_250_50_u1_r005_vote_th8[2], "C3", "50-1-005")
# ci_plot(ga_250_50_u1_r01_vote_th8[2], "C4", "50-1-01")
ci_plot(ga_250_25_u08_r005_vote_th8[2], "C1", "25-08-005")
# ci_plot(ga_250_25_u08_r01_vote_th8[2], "C2", "25-08-01")
# ci_plot(ga_250_50_u08_r005_vote_th8[2], "C3", "50-08-005")
# ci_plot(ga_250_50_u08_r01_vote_th8[2], "C4", "50-08-01")
# %%
# Predictive Performance
fig, ax = plt.subplots()
# ci_plot(ga_250_25_u1_r005_vote_th8[3], "C1", "25-1-005")
# ci_plot(ga_250_25_u1_r01_vote_th8[3], "C2", "25-1-01")
ci_plot(ga_250_50_u1_r005_vote_th8[3], "C3", "50-1-005")
# ci_plot(ga_250_50_u1_r01_vote_th8[3], "C4", "50-1-01")

# ci_plot(ga_250_25_u08_r005_vote_th8[3], "C1", "25-08-005")
# ci_plot(ga_250_25_u08_r01_vote_th8[3], "C2", "25-08-01")
ci_plot(ga_250_50_u08_r005_vote_th8[3], "C4", "50-08-005")
# ci_plot(ga_250_50_u08_r01_vote_th8[3], "C4", "50-08-01")
# %%
# -------------------------------- Predictive performance
# Extract the `best solutions` per fold
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u1_r005_vote_th8[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_8[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx[0])
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u1_r005_vote_th8_best_indices = fold_best_solution_list
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_25_u1_r005_vote_th8_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
ga_250_25_u1_r005_vote_th8_PP_best = predictive_ability(
    classifiers, fold_best_solution_list, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u1_r005_vote_th8_PP_best
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
ATI_ga_250_25_u1_r005_vote_th8 = average_tanimoto_index(ga_250_25_u1_r005_vote_th8_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u1_r005_vote_th8)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u1_r01_vote_th8[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_8[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u1_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u1_r01_vote_th8_best_indices = fold_best_solution_list
# %%
ga_250_25_u1_r01_vote_th8_PP_best = predictive_ability(
    classifiers, ga_250_25_u1_r01_vote_th8_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u1_r01_vote_th8_PP_best
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
ATI_ga_250_25_u1_r01_vote_th8 = average_tanimoto_index(ga_250_25_u1_r01_vote_th8_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u1_r01_vote_th8)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u1_r01_vote_th8[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_8[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u1_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u1_r01_vote_th8_best_indices = fold_best_solution_list
# %%
ga_250_50_u1_r01_vote_th8_PP_best = predictive_ability(
    classifiers, ga_250_50_u1_r01_vote_th8_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u1_r01_vote_th8_PP_best
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
ATI_ga_250_50_u1_r01_vote_th8 = average_tanimoto_index(ga_250_50_u1_r01_vote_th8_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u1_r01_vote_th8)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u1_r005_vote_th8[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_8[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u1_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u1_r005_vote_th8_best_indices = fold_best_solution_list
# %%
ga_250_50_u1_r005_vote_th8_PP_best = predictive_ability(
    classifiers, ga_250_50_u1_r005_vote_th8_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u1_r005_vote_th8_PP_best
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
ATI_ga_250_50_u1_r005_vote_th8 = average_tanimoto_index(ga_250_50_u1_r005_vote_th8_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u1_r005_vote_th8)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u08_r005_vote_th8[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_8[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u1_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u08_r005_vote_th8_best_indices = fold_best_solution_list
# %%
ga_250_25_u08_r005_vote_th8_PP_best = predictive_ability(
    classifiers, ga_250_25_u08_r005_vote_th8_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u08_r005_vote_th8_PP_best
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
ATI_ga_250_25_u08_r005_vote_th8 = average_tanimoto_index(ga_250_25_u08_r005_vote_th8_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u08_r005_vote_th8)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u08_r01_vote_th8[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_8[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u1_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u08_r01_vote_th8_best_indices = fold_best_solution_list
# %%
ga_250_25_u08_r01_vote_th8_PP_best = predictive_ability(
    classifiers, ga_250_25_u08_r01_vote_th8_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u08_r01_vote_th8_PP_best
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
ATI_ga_250_25_u08_r01_vote_th8 = average_tanimoto_index(ga_250_25_u08_r01_vote_th8_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u08_r01_vote_th8)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u08_r01_vote_th8[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_8[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u1_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u08_r01_vote_th8_best_indices = fold_best_solution_list
# %%
ga_250_50_u08_r01_vote_th8_PP_best = predictive_ability(
    classifiers, ga_250_50_u08_r01_vote_th8_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u08_r01_vote_th8_PP_best
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
ATI_ga_250_50_u08_r01_vote_th8 = average_tanimoto_index(ga_250_50_u08_r01_vote_th8_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u08_r01_vote_th8)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u08_r005_vote_th8[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_8[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u1_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u08_r005_vote_th8_best_indices = fold_best_solution_list
# %%
ga_250_50_u08_r005_vote_th8_PP_best = predictive_ability(
    classifiers, ga_250_50_u08_r005_vote_th8_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u08_r005_vote_th8_PP_best
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
ATI_ga_250_50_u08_r005_vote_th8 = average_tanimoto_index(ga_250_50_u08_r005_vote_th8_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u08_r005_vote_th8)
# --------------------------------------------------------------------------------------------------------
# %%
# -------------------------------------------------------------------------------------------------
# Boruta
# -------------------------------------------------------------------------------------------------
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xwrapper outputsGAx/new/'
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_1_random_0.05_vote_bor', 'rb') as f:
    ga_250_25_u1_r005_vote_bor = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_1_random_0.1_vote_bor', 'rb') as f:
    ga_250_25_u1_r01_vote_bor = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_1_random_0.1_vote_bor', 'rb') as f:
    ga_250_50_u1_r01_vote_bor = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_1_random_0.05_vote_bor', 'rb') as f:
    ga_250_50_u1_r005_vote_bor = dill.load(f)

with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_0.8_random_0.05_vote_bor', 'rb') as f:
    ga_250_25_u08_r005_vote_bor = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_25_uniform_0.8_random_0.1_vote_bor', 'rb') as f:
    ga_250_25_u08_r01_vote_bor = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_0.8_random_0.1_vote_bor', 'rb') as f:
    ga_250_50_u08_r01_vote_bor = dill.load(f)
with open(filter_pickle_directory+'ge_raw_6_ga_rws_250_50_uniform_0.8_random_0.05_vote_bor', 'rb') as f:
    ga_250_50_u08_r005_vote_bor = dill.load(f)

# %%
# -------------------------------- Comparison of internal results
# %%
# Fitness
# Boruta
fig, ax = plt.subplots()
ci_plot_out(ga_250_25_u1_r005_vote_bor[1], "C1", "25-1-005", ax, True)
ci_plot_out(ga_250_25_u1_r01_vote_bor[1], "C2", "25-1-01", ax, True)
ci_plot_out(ga_250_50_u1_r005_vote_bor[1], "C3", "50-1-005", ax, True)
ci_plot_out(ga_250_50_u1_r01_vote_bor[1], "C4", "50-1-01", ax, True)
#
ci_plot_out(ga_250_25_u08_r005_vote_bor[1], "C5", "25-08-005", ax, True)
ci_plot_out(ga_250_25_u08_r01_vote_bor[1], "C6", "25-08-01", ax, True)
ci_plot_out(ga_250_50_u08_r005_vote_bor[1], "C7", "50-08-005", ax, True)
ci_plot_out(ga_250_50_u08_r01_vote_bor[1], "C8", "50-08-01", ax, True)
# %%
# Number of features
set_style()
fig, ax = plt.subplots()
ci_plot_out(ga_250_25_u1_r005_vote_bor[2], "C1", "25-1-005",  ax, True)
ci_plot_out(ga_250_25_u1_r01_vote_bor[2], "C2", "25-1-01", ax, True)
ci_plot_out(ga_250_50_u1_r005_vote_bor[2], "C3", "50-1-005", ax, True)
ci_plot_out(ga_250_50_u1_r01_vote_bor[2], "C4", "50-1-01", ax, True)
ci_plot_out(ga_250_25_u08_r005_vote_bor[2], "C5", "25-08-005",  ax, True)
ci_plot_out(ga_250_25_u08_r01_vote_bor[2], "C6", "25-08-01", ax, True)
ci_plot_out(ga_250_50_u08_r005_vote_bor[2], "C7", "50-08-005",  ax, True)
ci_plot_out(ga_250_50_u08_r01_vote_bor[2], "C8", "50-08-01", ax, True)
# %%
# Predictive Performance
fig, ax = plt.subplots()
ci_plot_out(ga_250_25_u1_r005_vote_bor[3], "C1", "25-1-005", ax, True)
ci_plot_out(ga_250_25_u1_r01_vote_bor[3], "C2", "25-1-01", ax, True)
ci_plot_out(ga_250_50_u1_r005_vote_bor[3], "C3", "50-1-005", ax, True)
ci_plot_out(ga_250_50_u1_r01_vote_bor[3], "C4", "50-1-01", ax, True)
# fig, ax = plt.subplots()
ci_plot_out(ga_250_25_u08_r005_vote_bor[3], "C5", "25-08-005", ax, True)
ci_plot_out(ga_250_25_u08_r01_vote_bor[3], "C6", "25-08-01", ax, True)
ci_plot_out(ga_250_50_u08_r005_vote_bor[3], "C7", "50-08-005", ax, True)
ci_plot_out(ga_250_50_u08_r01_vote_bor[3], "C8", "50-08-01", ax, True)
ax.grid()
# --------------------------------------------------------------------------------------------------------
# %%
# -------------------------------------------- Predictive Performance
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u1_r01_vote_bor[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, np.array(idx_boruta_list[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
pd.DataFrame(all_best_solutions_idx[0])
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_25_u1_r01_vote_bor_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u1_r01_vote_bor_best_indices = fold_best_solution_list
# %%
ga_250_25_u1_r01_vote_bor_PP_best = predictive_ability(
    classifiers, ga_250_25_u1_r01_vote_bor_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u1_r01_vote_bor_PP_best
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
ATI_ga_250_25_u1_r01_vote_bor = average_tanimoto_index(ga_250_25_u1_r01_vote_bor_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u1_r01_vote_bor)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u1_r005_vote_bor[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, np.array(idx_boruta_list[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u1_r005_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u1_r005_vote_bor_best_indices = fold_best_solution_list
# %%
ga_250_25_u1_r005_vote_bor_PP_best = predictive_ability(
    classifiers, ga_250_25_u1_r005_vote_bor_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u1_r005_vote_bor_PP_best
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
ATI_ga_250_25_u1_r005_vote_bor = average_tanimoto_index(ga_250_25_u1_r005_vote_bor_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u1_r005_vote_bor)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_25_u08_r01_vote_bor[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, np.array(idx_boruta_list[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u08_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u08_r01_vote_bor_best_indices = fold_best_solution_list
# %%
ga_250_25_u08_r01_vote_bor_PP_best = predictive_ability(
    classifiers, ga_250_25_u08_r01_vote_bor_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u08_r01_vote_bor_PP_best
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
ATI_ga_250_25_u08_r01_vote_bor = average_tanimoto_index(ga_250_25_u08_r01_vote_bor_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u08_r01_vote_bor)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices

all_best_solutions = ga_250_25_u08_r005_vote_bor[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, np.array(idx_boruta_list[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_25_u08_r005_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%

# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u08_r005_vote_bor_best_indices = fold_best_solution_list
# %%
ga_250_25_u08_r005_vote_bor_PP_best = predictive_ability(
    classifiers, ga_250_25_u08_r005_vote_bor_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u08_r005_vote_bor_PP_best
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
ATI_ga_250_25_u08_r005_vote_bor = average_tanimoto_index(ga_250_25_u08_r005_vote_bor_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u08_r005_vote_bor)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices

all_best_solutions = ga_250_50_u1_r01_vote_bor[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, np.array(idx_boruta_list[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_50_u1_r01_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u1_r01_vote_bor_best_indices = fold_best_solution_list
# %%
ga_250_50_u1_r01_vote_bor_PP_best = predictive_ability(
    classifiers, ga_250_50_u1_r01_vote_bor_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u1_r01_vote_bor_PP_best
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
ATI_ga_250_50_u1_r01_vote_bor = average_tanimoto_index(ga_250_50_u1_r01_vote_bor_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u1_r01_vote_bor)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices

all_best_solutions = ga_250_50_u1_r005_vote_bor[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, np.array(idx_boruta_list[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
# with open('ga_250_50_u1_r005_vote_bor_fit_results', 'wb') as f:
#     pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u1_r005_vote_bor_best_indices = fold_best_solution_list
# %%
ga_250_50_u1_r005_vote_bor_PP_best = predictive_ability(
    classifiers, ga_250_50_u1_r005_vote_bor_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u1_r005_vote_bor_PP_best
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
ATI_ga_250_50_u1_r005_vote_bor = average_tanimoto_index(ga_250_50_u1_r005_vote_bor_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u1_r005_vote_bor)
# --------------------------------------------------------------------------------------------------------
# %%
# Binary feature position to feature indices

all_best_solutions = ga_250_50_u08_r01_vote_bor[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, np.array(idx_boruta_list[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

all_best_solutions_idx[2]
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_50_u08_r01_vote_bor_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u08_r01_vote_bor_best_indices = fold_best_solution_list
# %%
ga_250_50_u08_r01_vote_bor_PP_best = predictive_ability(
    classifiers, ga_250_50_u08_r01_vote_bor_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u08_r01_vote_bor_PP_best
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
ATI_ga_250_50_u08_r01_vote_bor = average_tanimoto_index(ga_250_50_u08_r01_vote_bor_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u08_r01_vote_bor)
# %%
# Binary feature position to feature indices
all_best_solutions = ga_250_50_u08_r005_vote_bor[7]
len(all_best_solutions[1])
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen,  np.array(idx_boruta_list[i]))
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

all_best_solutions_idx[2]
# %%
# generate external predictive performance for best feature set from each generation
# all best solutions indices
len(all_best_solutions_idx[49][0])
len(all_best_solutions_idx[::4])
input = all_best_solutions_idx
preproc = 'ens'

generations = range(0, 250)
folds = range(0, 50)
gens_pp = []
for gen in generations[::4]:
    gen_features = []
    for fold in folds:
        features = input[fold][gen]
        gen_features.append(features)
    gen_pp = predictive_ability(
        classifiers, gen_features, X_train, y_train, num_repeats, num_splits, preproc)
    gens_pp.append(gen_pp)
len(gens_pp)
geomean_gens_pp = []
for i in range(0, len(gens_pp)):
    sensitivity = pd.DataFrame(gens_pp[i][2], columns=classifier_names.keys())
    specificity = pd.DataFrame(gens_pp[i][3], columns=classifier_names.keys())
    geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
    geomean_gens_pp.append(geomean)
geomean_gens_pp
pd.DataFrame(geomean_gens_pp, columns=classifier_names)
# %%
# Pickle dump feature subset score and index lists
with open('ga_250_50_u08_r005_vote_bor_fit_results', 'wb') as f:
    pickle.dump(gens_pp, f)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - iter]
    fold_best_solution_list.append(fold_best_solution)
ga_250_50_u08_r005_vote_bor_best_indices = fold_best_solution_list
# %%
ga_250_50_u08_r005_vote_bor_PP_best = predictive_ability(
    classifiers, ga_250_50_u08_r005_vote_bor_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_50_u08_r005_vote_bor_PP_best
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
ATI_ga_250_50_u08_r005_vote_bor = average_tanimoto_index(ga_250_50_u08_r005_vote_bor_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_50_u08_r005_vote_bor)
# %%
'''
The above results produce the internal and external predictive performance results, as well as stability
results for the ensemble and boruta first phase approaches in combination with a genetic algorithm with
a varied set of hyper-parameters. The ensemble-ga approach is tested with both a equivalent first phase
cardinality as the boruta implementation and with a set of 25 features.
'''
# %%
'''
################################################################################################
#                    GA EXTERNAL RESULT GENEREATION @ MULTIPLE THRESHOLDS
################################################################################################
'''
# %%
classifier_names = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1),
    'Voting Ensemble': voting_classifier_pipeline_combo
}


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

classifier_names = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1),
    # 'Voting Ensemble': voting_classifier_pipeline_combo
}
selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF']
# %%
# Pickle upload
# Bor
# ga_250_25_u1_r01_vote_bor_fit_results
# ga_250_25_u1_r005_vote_bor_fit_results
# ga_250_25_u08_r01_vote_bor_fit_results
# ga_250_25_u08_r005_vote_bor_fit_results

# ga_250_50_u1_r01_vote_bor_fit_results
# ga_250_50_u08_r01_vote_bor_fit_results
# ga_250_50_u1_r005_vote_bor_fit_results

# ensemble
# ga_250_25_u1_r01_vote_th25_fit_results
# ga_250_25_u1_r005_vote_th25_fit_results
# ga_250_25_u08_r005_vote_th25_fit_results
# ga_250_25_u08_r01_vote_th25_fit_results
#
# ga_250_50_u08_r005_vote_th25_fit_results
# ga_250_50_u08_r01_vote_th25_fit_results
# ga_250_50_u1_r005_vote_th25_fit_results
# ga_250_50_u1_r01_vote_th25_fit_results

ens_predperf_250 = {
    "ga_250_25_u1_r005_vote_th25_fit_results": "250, 25, 100%, 5%",
    "ga_250_25_u1_r01_vote_th25_fit_results": "250, 25, 100%, 10%",
    "ga_250_25_u08_r005_vote_th25_fit_results": "250, 25, 80%, 5%",
    "ga_250_25_u08_r01_vote_th25_fit_results": "250, 25, 80%, 10%",
    "ga_250_50_u1_r005_vote_th25_fit_results": "250, 50, 100%, 5%",
    "ga_250_50_u1_r01_vote_th25_fit_results": "250, 50, 100%, 10%",
    "ga_250_50_u08_r005_vote_th25_fit_results": "250, 50, 80%, 5%",
    "ga_250_50_u08_r01_vote_th25_fit_results": "250, 50, 80%, 10%",
}

bor_predperf_250 = {
    "ga_250_25_u1_r005_vote_bor_fit_results": "250, 25, 100%, 5%",
    "ga_250_25_u1_r01_vote_bor_fit_results": "250, 25, 100%, 10%",
    "ga_250_25_u08_r005_vote_bor_fit_results": "250, 25, 80%, 5%",
    "ga_250_25_u08_r01_vote_bor_fit_results": "250, 25, 80%, 10%",
    "ga_250_50_u1_r005_vote_bor_fit_results": "250, 50, 100%, 5%",
    "ga_250_50_u1_r01_vote_bor_fit_results": "250, 50, 100%, 10%",
    "ga_250_50_u08_r005_vote_bor_fit_results": "250, 50, 80%, 5%",
    "ga_250_50_u08_r01_vote_bor_fit_results": "250, 50, 80%, 10%",
}

ensemble_fit_results_list = ['ga_250_25_u1_r005_vote_th25_fit_results', 'ga_250_50_u1_r005_vote_th25_fit_results', 'ga_250_50_u1_r005_vote_th25_fit_results', 'ga_250_50_u1_r01_vote_th25_fit_results',
                             'ga_250_25_u08_r005_vote_th25_fit_results', 'ga_250_50_u08_r005_vote_th25_fit_results', 'ga_250_25_u08_r01_vote_th25_fit_results', 'ga_250_50_u08_r01_vote_th25_fit_results']

ensemble_fit_results_omega_effect_list = [
    'ga_250_25_u1_r01_vote_th25_fit_results with voting ensemble results']

boruta_fit_results_list = ['ga_250_50_u08_r005_vote_bor_fit_results', 'ga_250_50_u08_r01_vote_bor_fit_results', 'ga_250_50_u1_r005_vote_bor_fit_results', 'ga_250_50_u1_r01_vote_bor_fit_results',
                           'ga_250_25_u08_r005_vote_bor_fit_results', 'ga_250_25_u08_r01_vote_bor_fit_results', 'ga_250_25_u1_r005_vote_bor_fit_results', 'ga_250_25_u1_r01_vote_bor_fit_results']

# input_list = ['ga_1000_50_u08_r005_vote_th25_fit_results']

input_dict = bor_predperf_250


ga_output_results_dict = {}
for results_input_name, value in input_dict.items():
    # results_input_name = 'ga_250_50_u1_r01_vote_bor_fit_results'# + '_fit_results'
    with open(results_input_name, 'rb') as f:
        ga_output_results = pickle.load(
            f)
    geomean_list = []
    geomean_list.append(np.zeros(len(classifier_names)))
    for i in range(0, len(ga_output_results)):
        sensitivity = pd.DataFrame(ga_output_results[i][2], columns=classifier_names.keys())
        specificity = pd.DataFrame(ga_output_results[i][3], columns=classifier_names.keys())
        geomean = np.nanmedian(np.sqrt(sensitivity*specificity), axis=0)
        geomean_list.append(geomean)
    ga_output_results_dict[value] = geomean_list
# %%
selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF']
set_style()
range_start = 1*4
range_end = 250
fig_width = 5.8
fig_height_scale = 2.5


fig, axs = plt.subplots(4, 2, figsize=(fig_width, gr*fig_width*fig_height_scale))
fig.subplots_adjust(hspace=.25)

axs = axs.ravel()
i = 0
for results_input_name, value in input_dict.items():
    result = pd.DataFrame(ga_output_results_dict[value], columns=classifier_names)
    print(result.index*4)
    result.index = result.index*4
    for key in selected_classifiers:
        axs[i].plot(range(range_start, range_end, 4),
                    result.loc[range_start:range_end, key], label=key)
    axs[i].set_ylim(0.45, 0.7)
    axs[i].set_title(value)
    i += 1
# set common x axis label
fig.text(0.5, 0.03, 'Number of Generations', ha='center', va='center')
# set common y axis label
fig.text(-0.01, 0.5, 'External Cross-Validation \nPredictive Performance',
         ha='center', va='center', rotation='vertical')
# Put the legend out of the figure
handles, labels = axs[0].get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.55, -0.002),
           loc="lower center", ncol=len(selected_classifiers))
fig.tight_layout()
fig.subplots_adjust(bottom=0.08, left=0.08)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/BorutaGAExternalPPResults.png",
#             bbox_inches="tight", dpi=1000)
# %%
# individual graph
ga_output_results_list
selected_classifiers = ['Voting Ensemble']
set_style()
range_start = 1
range_end = len(ga_output_results)+1
fig_width = 5.8
fig_height_scale = 0.8

fig, ax = plt.subplots(1, figsize=(fig_width, gr*fig_width*fig_height_scale))
for key in selected_classifiers:
    ax.plot(range(range_start, range_end), pd.DataFrame(ga_output_results_list[0],
                                                        columns=classifier_names).loc[range_start:range_end, key], label=key)
ax.legend()
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/GAExternalVotingPredResults.png",
#             bbox_inches="tight", dpi=1000)
# %%
# --------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# Final GA Results Output
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# Set output styles


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
# --------------------------------------------------------------------------------------------------------
# Internal results: Hyper-parameter evaluation
# --------------------------------------------------------------------------------------------------------
# Ensemble
# Predictive Performance (plot type = 3) and Selected num features (plot type = 2)
set_style()
plot_type = 3
fig_width = 5.8
fig_height_scale = 0.8
fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(fig_width, gr*fig_width*fig_height_scale))
ci_plot_out(ga_250_25_u1_r005_vote_th25[plot_type], "C1", "25, 100%, 5%", only_mean=True, axis=ax1)
ci_plot_out(ga_250_25_u1_r01_vote_th25[plot_type], "C2", "25, 100%, 10%", only_mean=True, axis=ax1)
ci_plot_out(ga_250_25_u08_r005_vote_th25[plot_type], "C3", "25, 80%, 5%", only_mean=True, axis=ax1)
ci_plot_out(ga_250_25_u08_r01_vote_th25[plot_type], "C4", "25, 80%, 10%", only_mean=True, axis=ax1)
# ci_plot_out(ga_1000_25_u08_r005_vote_th25[plot_type], "C5", "25,08,01", only_mean=True, axis=ax1)
# ci_plot_out(ga_1000_50_u08_r005_vote_th25[plot_type], "C6", "50,08,01", only_mean=True, axis=ax1)
ax1.grid()
ax1.set_ylabel("Internal Cross-validation\n# Selected Features")
# ax1.set_ylim(0.68, 0.8)
# ci_plot(ga_250_50_u1_r005_vote_th25[plot_type], "C3", "50-1-005", True)
# ci_plot(ga_250_50_u1_r01_vote_th25[plot_type], "C4", "50-1-01", True)
# ax.grid()
# fig, ax2 = plt.subplots()
ci_plot_out(ga_250_50_u1_r005_vote_th25[plot_type], "C1", "50, 100%, 5%", only_mean=True, axis=ax2)
ci_plot_out(ga_250_50_u1_r01_vote_th25[plot_type], "C2", "50, 100%, 10%", only_mean=True, axis=ax2)
ci_plot_out(ga_250_50_u08_r005_vote_th25[plot_type], "C3", "50, 80%, 5%", only_mean=True, axis=ax2)
ci_plot_out(ga_250_50_u08_r01_vote_th25[plot_type], "C4", "50, 80%, 10%", only_mean=True, axis=ax2)
# ci_plot_out(ga_1000_25_u08_r005_vote_th25[plot_type], "C5", "25,08,01", only_mean=True, axis=ax2)
# ci_plot_out(ga_1000_50_u08_r005_vote_th25[plot_type], "C6", "50,08,01", only_mean=True, axis=ax2)

ax2.grid()
ax2.set_ylabel("")
# ax2.set_ylim(0.68, 0.8)
# fig.subplots_adjust(wspace=0.34)

# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/GA8EnsInternalPredPerf.png",
#             bbox_inches="tight", dpi=1000)
# %%
# Predictive Performance (plot type = 3) and Selected num features (plot type = 2) for 1000 generations
set_style()
# Predictive Performance
plot_type = 3
fig_width = 5.8
fig_height_scale = 0.8
fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(fig_width, gr*fig_width*fig_height_scale))
# ci_plot_out(ga_250_25_u1_r005_vote_th25[plot_type], "C1", "25,1,005", only_mean=True, axis=ax1)
# ci_plot_out(ga_250_25_u1_r01_vote_th25[plot_type], "C2", "25,1,01", only_mean=True, axis=ax1)
# ci_plot_out(ga_250_25_u08_r005_vote_th25[plot_type], "C3", "25,08,005", only_mean=True, axis=ax1)
# ci_plot_out(ga_250_25_u08_r01_vote_th25[plot_type], "C4", "25,08,01", only_mean=True, axis=ax1)
ci_plot_out(ga_1000_25_u08_r005_vote_th25[plot_type],
            "C5", "25, 80%, 10%", only_mean=True, axis=ax1)
ci_plot_out(ga_1000_50_u08_r005_vote_th25[plot_type],
            "C6", "50, 80%, 10%", only_mean=True, axis=ax1)
ax1.grid()
ax1.set_ylabel("Internal Cross-validation\nPredictive Performance")
# ax1.set_ylim(0.68, 0.8)
# ci_plot(ga_250_50_u1_r005_vote_th25[plot_type], "C3", "50-1-005", True)
# ci_plot(ga_250_50_u1_r01_vote_th25[plot_type], "C4", "50-1-01", True)
# ax.grid()
# fig, ax = plt.subplots()
# Selected num features
plot_type = 2
# ci_plot(ga_250_25_u08_r005_vote_th25[plot_type], "C5", "25-08-005", True)
# ci_plot(ga_250_25_u08_r01_vote_th25[plot_type], "C6", "25-08-01", True)
# ci_plot_out(ga_250_50_u1_r005_vote_th25[plot_type], "C1", "50,1,005", only_mean=True, axis=ax2)
# ci_plot_out(ga_250_50_u1_r01_vote_th25[plot_type], "C2", "50,1,01", only_mean=True, axis=ax2)
# ci_plot_out(ga_250_50_u08_r005_vote_th25[plot_type], "C3", "50,08,005", only_mean=True, axis=ax2)
# ci_plot_out(ga_250_50_u08_r01_vote_th25[plot_type], "C4", "50,08,01", only_mean=True, axis=ax2)
ci_plot_out(ga_1000_25_u08_r005_vote_th25[plot_type],
            "C5", "25, 80%, 10%", only_mean=True, axis=ax2)
ci_plot_out(ga_1000_50_u08_r005_vote_th25[plot_type],
            "C6", "50, 80%, 10%", only_mean=True, axis=ax2)

ax2.grid()
ax2.set_ylabel("Internal Cross-validation\n# Selected Features")
fig.subplots_adjust(wspace=0.34)
# ax2.set_ylim(0.68, 0.8)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/GAEnsInternalPredPerf&NumSelFeat_1000.png",
#             bbox_inches="tight", dpi=1000)

# %%
# Boruta
# Predictive Performance (plot type = 3) and Selected num features (plot type = 2)
plot_type = 3
fig_width = 5.8
fig_height_scale = 0.8
fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(fig_width, gr*fig_width*fig_height_scale))
ci_plot_out(ga_250_25_u1_r005_vote_bor[plot_type], "C1",
            "25,100%, 5%", only_mean=True, axis=ax1)
ci_plot_out(ga_250_25_u1_r01_vote_bor[plot_type], "C2",
            "25, 100%, 10%", only_mean=True, axis=ax1)
ci_plot_out(ga_250_25_u08_r005_vote_bor[plot_type], "C3",
            "25, 80%, 5%", only_mean=True, axis=ax1)
ci_plot_out(ga_250_25_u08_r01_vote_bor[plot_type], "C4",
            "25, 80%, 10%", only_mean=True, axis=ax1)
ax1.grid()
ax1.set_ylabel("Internal Cross-validation\n# Selected Features")
ax1.set_ylim(0.705, 0.793)
# ci_plot(ga_250_50_u1_r005_vote_bor[plot_type], "C3", "50-1-005", True)
# ci_plot(ga_250_50_u1_r01_vote_bor[plot_type], "C4", "50-1-01", True)
# ax.grid()
# fig, ax = plt.subplots()
# ci_plot(ga_250_25_u08_r005_vote_bor[plot_type], "C5", "25-08-005", True)
# ci_plot(ga_250_25_u08_r01_vote_bor[plot_type], "C6", "25-08-01", True)
ci_plot_out(ga_250_50_u1_r005_vote_bor[plot_type], "C1",
            "50, 100%, 5%", only_mean=True, axis=ax2)
ci_plot_out(ga_250_50_u1_r01_vote_bor[plot_type], "C2",
            "50, 100%, 10%", only_mean=True, axis=ax2)
ci_plot_out(ga_250_50_u08_r005_vote_bor[plot_type], "C3",
            "50, 80%, 5%", only_mean=True, axis=ax2)
ci_plot_out(ga_250_50_u08_r01_vote_bor[plot_type], "C4",
            "50, 80%, 10%", only_mean=True, axis=ax2)
ax2.grid()
ax2.set_ylabel("")
ax2.set_ylim(0.705, 0.793)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/GABorInternalPredPerf.png",
#             bbox_inches="tight", dpi=1000)

# %%
# Final internal results num selected features
# Ensemble
pp_249_list = []
for pp_fold in ga_250_25_u1_r005_vote_th25[2]:
    pp_249 = pp_fold[249]
    pp_249_list.append(pp_249)

input = pp_249_list
print('max: ' + str(np.max(input)))
print('mean: ' + str(np.mean(input)))
print('10: ' + str(np.sort(input)[4]))
print('50: ' + str(np.sort(input)[24]))
print('90: ' + str(np.sort(input)[44]))
print('min: ' + str(np.min(input)))
# %%
# Boruta
pp_249_list = []
for pp_fold in ga_250_25_u1_r005_vote_bor[2]:
    pp_249 = pp_fold[249]
    pp_249_list.append(pp_249)

input = pp_249_list
print('max: ' + str(np.max(input)))
print('mean: ' + str(np.mean(input)))
print('10: ' + str(np.sort(input)[4]))
print('50: ' + str(np.sort(input)[24]))
print('90: ' + str(np.sort(input)[44]))
print('min: ' + str(np.min(input)))
# %%
# --------------------------------------------------------------------------------------------------------
# External results: Hyper-parameter evaluation
# --------------------------------------------------------------------------------------------------------
# Results import
# Function to change key names


def rename_keys(d, keys):
    return dict([(keys.get(k), v) for k, v in d.items()])


# New key names
new_key_names_ens_250 = {
    "250, 25, 0.05, 1": "250, 25, 100%, 5%",
    "250, 25, 0.1, 1": "250, 25, 100%, 10%",
    "250, 25, 0.05, 0.8": "250, 25, 80%, 5%",
    "250, 25, 0.1, 0.8": "250, 25, 80%, 10%",
    "250, 50, 0.05, 1": "250, 50, 100%, 5%",
    "250, 50, 0.1, 1": "250, 50, 100%, 10%",
    "250, 50, 0.05, 0.8": "250, 50, 80%, 5%",
    "250, 50, 0.1, 0.8": "250, 50, 80%, 10%",
    "1000, 50, 0.05, 0.8": "1000, 50, 80%, 5%",
    "1000, 25, 0.05, 0.8": "1000, 25, 80%, 5%"
}

new_key_names_bor_250 = {
    "250, 25, 0.05, 1": "250, 25, 100%, 5%",
    "250, 25, 0.1, 1": "250, 25, 100%, 10%",
    "250, 25, 0.05, 0.8": "250, 25, 80%, 5%",
    "250, 25, 0.1, 0.8": "250, 25, 80%, 10%",
    "250, 50, 0.05, 1": "250, 50, 100%, 5%",
    "250, 50, 0.1, 1": "250, 50, 100%, 10%",
    "250, 50, 0.05, 0.8": "250, 50, 80%, 5%",
    "250, 50, 0.1, 0.8": "250, 50, 80%, 10%"
}

new_key_names_ens_100 = {
    "100, 25, 0.05, 1": "100, 25, 100%, 5%",
    "100, 25, 0.1, 1": "100, 25, 100%, 10%",
    "100, 25, 0.05, 0.8": "100, 25, 80%, 5%",
    "100, 25, 0.1, 0.8": "100, 25, 80%, 10%",
    "100, 50, 0.05, 1": "100, 50, 100%, 5%",
    "100, 50, 0.1, 1": "100, 50, 100%, 10%",
    "100, 50, 0.05, 0.8": "100, 50, 80%, 5%",
    "100, 50, 0.1, 0.8": "100, 50, 80%, 10%"
}

new_key_names_bor_100 = {
    "100, 25, 0.05, 1": "100, 25, 100%, 5%",
    "100, 25, 0.1, 1": "100, 25, 100%, 10%",
    "100, 25, 0.05, 0.8": "100, 25, 80%, 5%",
    "100, 25, 0.1, 0.8": "100, 25, 80%, 10%",
    "100, 50, 0.05, 1": "100, 50, 100%, 5%",
    "100, 50, 0.1, 1": "100, 50, 100%, 10%",
    "100, 50, 0.05, 0.8": "100, 50, 80%, 5%",
    "100, 50, 0.1, 0.8": "100, 50, 80%, 10%"
}
# %%
# results_ens = {
#     "100, 25, 0.05, 1": ga_250_25_u1_r005_vote_th8_PP_best,
#     "100, 25, 0.1, 1": ga_250_25_u1_r01_vote_th8_PP_best,
#     "100, 25, 0.05, 0.8": ga_250_25_u08_r005_vote_th8_PP_best,
#     "100, 25, 0.1, 0.8": ga_250_25_u08_r01_vote_th8_PP_best,
#     "100, 50, 0.05, 1": ga_250_50_u1_r005_vote_th8_PP_best,
#     "100, 50, 0.1, 1": ga_250_50_u1_r01_vote_th8_PP_best,
#     "100, 50, 0.05, 0.8": ga_250_50_u08_r005_vote_th8_PP_best,
#     "100, 50, 0.1, 0.8": ga_250_50_u08_r01_vote_th8_PP_best
# }
# Pickle dump ensemble results
# -------------------------------
# with open(filename + '_ga_wrapper_stage_8_ens_vote_' + str(250), 'wb') as f:
#     pickle.dump([results_ens, ga_ens_overlap], f)
with open(filename+'_ga_wrapper_stage__ens_vote_250', 'rb') as f:
    results_ens_250, ga_ens_overlap_250 = pickle.load(f)
with open(filename+'_ga_wrapper_stage__ens_vote_100', 'rb') as f:
    results_ens_100, ga_ens_overlap_100 = pickle.load(f)
with open(filename+'_ga_wrapper_stage_8_ens_vote_250', 'rb') as f:
    results_ens8_250, ga_ens8_overlap_250 = pickle.load(f)
results_ens_250.keys()
# results_ens_250['1000, 50, 0.05, 0.8'] = ga_1000_50_u08_r005_vote_th25_PP_best
# results_ens_250['1000, 25, 0.05, 0.8'] = ga_1000_25_u08_r005_vote_th25_PP_best
# Change key names
results_ens_250 = rename_keys(results_ens_250, new_key_names_ens_250)
results_ens_100 = rename_keys(results_ens_100, new_key_names_ens_100)
results_ens8_250 = rename_keys(results_ens8_250, new_key_names_ens_100)
# results_bor = {
#     "100, 25, 0.05, 1": ga_250_25_u1_r005_vote_bor_PP_best,
#     "100, 25, 0.1, 1": ga_250_25_u1_r01_vote_bor_PP_best,
#     "100, 25, 0.05, 0.8": ga_250_25_u08_r005_vote_bor_PP_best,
#     "100, 25, 0.1, 0.8": ga_250_25_u08_r01_vote_bor_PP_best,
#     "100, 50, 0.05, 1": ga_250_50_u1_r005_vote_bor_PP_best,
#     "100, 50, 0.1, 1": ga_250_50_u1_r01_vote_bor_PP_best,
#     "100, 50, 0.05, 0.8": ga_250_50_u08_r005_vote_bor_PP_best,
#     "100, 50, 0.1, 0.8": ga_250_50_u08_r01_vote_bor_PP_best
# }
# %%
# Pickle dump Boruta results
# -------------------------------
# with open(filename + '_ga_wrapper_stage__bor_vote_' + str(100) + "_new", 'wb') as f:
#     pickle.dump([results_bor,ga_bor_overlap], f)
with open(filename+'_ga_wrapper_stage__bor_vote_250', 'rb') as f:
    results_bor_250, ga_bor_overlap_250 = pickle.load(f)
with open(filename+'_ga_wrapper_stage__bor_vote_250_new', 'rb') as f:
    results_bor_250_new, ga_bor_overlap_250_new = pickle.load(f)
with open(filename+'_ga_wrapper_stage__bor_vote_100', 'rb') as f:
    results_bor_100, ga_bor_overlap_100 = pickle.load(f)
with open(filename+'_ga_wrapper_stage__bor_vote_100_new', 'rb') as f:
    results_bor_100_new, ga_bor_overlap_100_new = pickle.load(f)
results_bor_100_new.keys()
# Change key names
results_bor_250 = rename_keys(results_bor_250, new_key_names_bor_250)
results_bor_100 = rename_keys(results_bor_100, new_key_names_bor_100)
results_bor_250_new = rename_keys(results_bor_250_new, new_key_names_bor_250)
results_bor_100_new = rename_keys(results_bor_100_new, new_key_names_bor_100)
results_bor_100_new.keys()
# --------------------------------------------------------------------------------------------
# %%
# Results Visualizations
fig_height_scale = 1.8
set_style()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(
    fig_width, gr*fig_width*fig_height_scale))  # for two figure
# fig, ax1 = plt.subplots(figsize=(fig_width, gr*fig_width*fig_height_scale))  # for one figure
ax1 = boxplot_filter(results_bor_250_new, classifier_names, "Geomean", selected_classifiers, ax1)
ax1.legend_.remove()
ax1.set_xlabel("GA hyper-parameter combinations with Ensemble selected features")
ax1.set_ylabel("External Cross-validation\n Predictive Performance")
ax1.set_ylim(0.2, 1)
# for two figure
# ----------------------
ax1.set_xticklabels([])
ax1.get_xaxis().get_label().set_visible(False)
ax2 = boxplot_filter(results_bor_100_new, classifier_names, "Geomean", selected_classifiers, ax2)
ax2.legend_.remove()
ax2.set_xlabel("GA hyper-parameter combinations with ensemble selected features")
ax2.set_ylabel("External Cross-validation\n Predictive Performance")
ax2.set_ylim(0.2, 1)
ax2.get_xaxis().get_label().set_visible(False)
# ----------------------

# Put the legend out of the figure
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.545, 0), loc="lower center", ncol=len(classifiers))
fig.tight_layout()
fig.subplots_adjust(bottom=0.33)

# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/GAEnsExternalPredPerfGeo.png",
#             bbox_inches="tight", dpi=1000)
# %%
mean_tables(results_ens_250, classifier_names, "Geo", selected_classifiers)
mean_tables(results_ens8_250, classifier_names, "Geo", selected_classifiers)
# %%
print("Ensemble features")
num_gen = str(100)
input_set = ga_ens_overlap
print(num_gen + ', 25, 0.05, 1 - ' +
      str(average_tanimoto_index(input_set[num_gen + ', 25, 0.05, 1'])))
print(num_gen + ', 25, 0.1, 1 - ' +
      str(average_tanimoto_index(input_set[num_gen + ', 25, 0.1, 1'])))
print(num_gen + ', 25, 0.05, 0.8 - ' +
      str(average_tanimoto_index(input_set[num_gen + ', 25, 0.05, 0.8'])))
print('\n'+num_gen + ', 25, 0.1, 0.8 - ' +
      str(average_tanimoto_index(input_set[num_gen + ', 25, 0.1, 0.8'])))
print(num_gen + ', 50, 0.05, 1 - ' +
      str(average_tanimoto_index(input_set[num_gen + ', 50, 0.05, 1'])))
print(num_gen + ', 50, 0.1, 1 - ' +
      str(average_tanimoto_index(input_set[num_gen + ', 50, 0.1, 1'])))
print(num_gen + ', 50, 0.05, 0.8 - ' +
      str(average_tanimoto_index(input_set[num_gen + ', 50, 0.05, 0.8'])))

print(num_gen + ',50, 0.1, 0.8 - ' +
      str(average_tanimoto_index(input_set[num_gen + ', 50, 0.1, 0.8'])))
# %%
print("250, 50, 0.1, 0.8 - " + str(average_tanimoto_index(input_set['100, 50, 0.1, 0.8'])))

# %%
print('Boruta features')
print("250, 25, 0.05, 1 - " + str(average_tanimoto_index(ga_250_25_u1_r005_vote_bor_best_indices)))
print("250, 25, 0.1, 1 - " + str(average_tanimoto_index(ga_250_25_u1_r01_vote_bor_best_indices)))
print("250, 25, 0.05, 0.8 - " + str(average_tanimoto_index(ga_250_25_u08_r005_vote_bor_best_indices)))
print("250, 25, 0.1, 0.8 - " + str(average_tanimoto_index(ga_250_25_u08_r01_vote_bor_best_indices)))
print("\n250, 50, 0.05, 1 - " + str(average_tanimoto_index(ga_250_50_u1_r005_vote_bor_best_indices)))
print("250, 50, 0.1, 1 - " + str(average_tanimoto_index(ga_250_50_u1_r01_vote_bor_best_indices)))
print("250, 50, 0.05, 0.8 - " + str(average_tanimoto_index(ga_250_50_u08_r005_vote_bor_best_indices)))
print("250, 50, 0.1, 0.8 - " + str(average_tanimoto_index(ga_250_50_u08_r01_vote_bor_best_indices)))
# %%
ga_250_25_u1_r005_vote_bor_best_indices

res = list(set.intersection(*map(set, ga_250_25_u1_r005_vote_bor_best_indices)))
res
# %%
# Output overlap
# ga_ens_overlap={
#     "100, 25, 0.05, 1": ga_250_25_u1_r005_vote_th8_best_indices,
#     "100, 25, 0.1, 1": ga_250_25_u1_r01_vote_th8_best_indices,
#     "100, 25, 0.05, 0.8": ga_250_25_u08_r005_vote_th8_best_indices,
#     "100, 25, 0.1, 0.8": ga_250_25_u08_r01_vote_th8_best_indices,
#     "100, 50, 0.05, 1": ga_250_50_u1_r005_vote_th8_best_indices,
#     "100, 50, 0.1, 1": ga_250_50_u1_r01_vote_th8_best_indices,
#     "100, 50, 0.05, 0.8": ga_250_50_u08_r005_vote_th8_best_indices,
#     "100, 50, 0.1, 0.8": ga_250_50_u08_r01_vote_th8_best_indices
# }
#
# ga_bor_overlap = {
#     "250, 25, 0.05, 1": ga_250_25_u1_r005_vote_bor_best_indices,
#     "250, 25, 0.1, 1": ga_250_25_u1_r01_vote_bor_best_indices,
#     "250, 25, 0.05, 0.8": ga_250_25_u08_r005_vote_bor_best_indices,
#     "250, 25, 0.1, 0.8": ga_250_25_u08_r01_vote_bor_best_indices,
#     "250, 50, 0.05, 1": ga_250_50_u1_r005_vote_bor_best_indices,
#     "250, 50, 0.1, 1": ga_250_50_u1_r01_vote_bor_best_indices,
#     "250, 50, 0.05, 0.8": ga_250_50_u08_r005_vote_bor_best_indices,
#     "250, 50, 0.1, 0.8": ga_250_50_u08_r01_vote_bor_best_indices
# }
# %%
set_list_th = ga_bor_overlap_250
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
# Number of features SeLected
print("Boruta Size Confidence Interval")

# wrap_boruta_mrm_RF_050bal_gmean
input = ga_250_50_u1_r005_vote_th8_best_indices
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
# Kyk na die ander ensemble predictive performance resultate

# __________________________________________________________________________________________________


# Binary feature position to feature indices
all_best_solutions = ga_250_25_u1_r01_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

len(all_best_solutions_idx)
# %%
# Extract best features per fold (last iteration)
fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - 1]
    fold_best_solution_list.append(fold_best_solution)
ga_250_25_u1_r01_vote_th25_best_indices = fold_best_solution_list
# %%
ga_250_25_u1_r01_vote_th25_PP_best = predictive_ability(
    classifiers, ga_250_25_u1_r01_vote_th25_best_indices, X_train, y_train, num_repeats, num_splits, preproc)
# %%
input = ga_250_25_u1_r01_vote_th25_PP_best
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
ATI_ga_250_25_u1_r01_vote_th25 = average_tanimoto_index(ga_250_25_u1_r01_vote_th25_best_indices)
print('Voting Classifier ATI:   ', ATI_ga_250_25_u1_r01_vote_th25)
# %%
idx_ensemble_list_25[1]
with open('idx_boruta_list', 'rb') as f:
    idx_boruta_list = dill.load(
        f)

with open('idx_ensemble_list_25', 'rb') as f:
    idx_ensemble_list_25 = dill.load(
        f)
with open('idx_ensemble_list_8', 'rb') as f:
    idx_ensemble_list_8 = dill.load(
        f)
preproc = 'ens'

with open('idx_ensemble_list_2)', 'rb') as f:
    idx_ensemble_list_25 = dill.load(
        f)
preproc = 'ens'


all_best_solutions = ga_250_25_u1_r005_vote_th25[7]
i = 0
all_best_solutions_idx = []
for fold_solutions in all_best_solutions:
    fold_solutions_idx = []
    for solutions_per_gen in fold_solutions:
        reduced_features_ind = reduce_features(solutions_per_gen, idx_ensemble_list_25[i])
        fold_solutions_idx.append(reduced_features_ind)
    all_best_solutions_idx.append(fold_solutions_idx)
    i = i+1

fold_best_solution_list = []
for i in range(0, 50):
    fold_best_solution = all_best_solutions_idx[i][len(all_best_solutions_idx[0]) - 1]
    fold_best_solution_list.append(fold_best_solution)
fold_best_solution_list
# doen basics dan probeer overfitting kode skryf vir almal
# %%
len(all_best_solutions_idx[0])
ex = []
for i in range(0, 250):
    lah = []
    for x in all_best_solutions_idx:
        row = x[i]
        lah.append(row)
    ex.append(lah)

len(ex[249])
ex[249]
# %%
ga_len_external_results = []
for i in range(0, 250):
    for j in range(0, 50):
        num_feat = len(ex[i][j])

# %%

# %%
result_list = []
for i in range(0, 250):
    ga_250_25_u1_r005_vote_th25_PP = predictive_ability(
        classifiers, ex[i], X_train, y_train, num_repeats, num_splits, preproc)
    result_list.append(ga_250_25_u1_r005_vote_th25_PP)
# %%
pd.DataFrame(result_list)
fig, ax = plt.subplots()
ci_plot(result_list[3], "C1", "Best Solutions Mean")
ci_plot(ga_250_25_u1_r005_vote_th25[6], "C2", "Population Averages Mean")
ci_plot(ga_250_50_u1_r005_vote_th25[3], "C3", "Best Solutions Mean 2 ")
ci_plot(ga_250_50_u1_r005_vote_th25[6], "C4", "Population Averages Mean 2")
# %%
ga_gmean_external_results = []
for i in range(0, 250):
    input = result_list[i]
    m_auc = pd.DataFrame(auc_clf_compiler(
        classifiers, input[4], input[5]), columns=classifiers.keys()).mean(axis=0)
    s_auc = pd.DataFrame(auc_clf_compiler(
        classifiers, input[4], input[5]), columns=classifiers.keys()).std(axis=0)
    m_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).mean(axis=0)
    s_sens = pd.DataFrame(input[2][0:-2, :], columns=classifiers.keys()).std(axis=0)
    m_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys())
    for i in range(0, 5):
        s_sens
    s_spec = pd.DataFrame(input[3][0:-2, :], columns=classifiers.keys())
    m_gmean = np.sqrt(m_spec*m_sens)
    s_gmean = np.sqrt(s_sens*s_spec)
    ga_gmean_external_results.append(m_gmean)
    ms = pd.concat([m_auc, s_auc, m_sens, s_sens, m_spec, s_spec, m_gmean, s_gmean], axis=1)
    ms.columns = ["Mean AUC", "Std AUC", "Mean Sens", "Std Sens",
                  "Mean Spec", "Std Spec", "Mean gmean", "Std gmean"]
