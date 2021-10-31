'''
The following script runs the Boruta-RFE feature selection model to generate
outputs for the final test set evaluation.


This script is setup for the interchangeable evaluation of the ACS and GC6-74 datasets.
'''
# imports
############################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import roc_curve, auc
import time
import pickle
from matplotlib import pyplot as plt

# local
from boruta_rfe_fs import BorutaRFE
from boruta_rfecv_fs import BorutaRFECV

# Standardization
from median_ratio_method import geo_mean, median_ratio_standardization

# Machine Learning Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# functions
def single_eval(classifiers, subset_list, X_train, y_train, X_test, y_test, preprocessing):

    # apply deseq
    print("Applying: " + preprocessing)
    X_train = np.round(median_ratio_standardization(X_train), 0)
    X_test = np.round(median_ratio_standardization(X_test), 0)

    # Extract only selected features from training/testing fold
    X_train_sel = X_train[:, subset_list]
    X_test_sel = X_test[:, subset_list]
    # classifier loop

    # initialize CV output variables
    clfs_predict_list = []
    clfs_acc_list = []
    clfs_sensitivity_list = []
    clfs_specificity_list = []
    clfs_precision_list = []
    clfs_fpr_list = []
    clfs_tpr_list = []

    # classifier loop
    for clf_key, clf in classifiers.items():
        print(clf_key)
        # application of standardization
        if clf_key in ('SVM_linear', 'SVM_rbf', 'KNN'):
            print("Standardizing")
            scaler = StandardScaler().fit(X_train_sel)
            X_train_sel_sc = scaler.transform(X_train_sel)
            X_test_sel_sc = scaler.transform(X_test_sel)
            # class imbalance rectification
            sm = SMOTE(random_state=42)
            # application of SMOTE, BorderlineSMOTE
            X_train_sel_sm, y_train_sm = sm.fit_resample(
                X_train_sel_sc, y_train)  # (if smote)

            X_train_in = X_train_sel_sm
            y_train_in = y_train_sm
            X_test_in = X_test_sel_sc
        else:
            print('Not Standardizing')
            # class imbalance rectification
            sm = SMOTE(random_state=42)
            # application of SMOTE, BorderlineSMOTE
            X_train_sel_sm, y_train_sm = sm.fit_resample(
                X_train_sel, y_train)  # (if smote)

            X_train_in = X_train_sel_sm
            y_train_in = y_train_sm
            X_test_in = X_test_sel
        if preprocessing == "ens" and clf_key in ('GaussianNB'):
            # apply log normalization
            print("Normalizing for NB")
            X_train_in = np.log2(X_train_in+1)
            X_test_in = np.log2(X_test_in+1)

        # classifier input training data and labels

        # train classifier
        clf.fit(X_train_in, y_train_in)

        # predict
        y_test_predict = clf.predict(X_test_in)
        clfs_predict_list.append(y_test_predict)
        # accuracy
        acc = accuracy_score(y_test, y_test_predict)
        clfs_acc_list.append(acc)
        # recall/sensitivity
        # with class 1 as representing a positive
        sensitivity = recall_score(y_test, y_test_predict)
        clfs_sensitivity_list.append(sensitivity)
        # specificity
        # with class 0 as representing a negative
        specificity = recall_score(y_test, y_test_predict, pos_label=0)
        clfs_specificity_list.append(specificity)
        # ROC variables generation
        # for classifiers which produce confidence scores
        if clf_key in ('SVM_linear', 'SVM_rbf'):
            y_score = clf.decision_function(X_test_in)  # confidence score
            # False positive rate & True positive rate
            fpr, tpr, threshold = roc_curve(y_test, y_score)
            clfs_fpr_list.append(fpr)
            clfs_tpr_list.append(tpr)
        # for classifiers which produce probability estimations
        else:
            y_probas = clf.predict_proba(X_test_in)  # probability estimations
            y_score = y_probas[:, 1]  # confidence score
            # False positive rate & True positive rate
            fpr, tpr, threshold = roc_curve(y_test, y_score)
            clfs_fpr_list.append(fpr)
            clfs_tpr_list.append(tpr)
    return clfs_predict_list, clfs_acc_list, clfs_sensitivity_list, clfs_specificity_list, clfs_fpr_list, clfs_tpr_list


# %%
# import data
############################################################
# training
filename = 'ge_raw_12'
_data_training = pd.read_csv(filename+'.csv', sep=',')

# testing
filename = 'gc6_test'
_data_testing = pd.read_csv(filename+'.csv', sep=',')

# validation
# original
# training
filename = 'ge_raw_acs_train'
_data_acs_training = pd.read_csv(filename+'.csv', sep=',')
# testing
filename = 'ge_raw_acs_test'
_data_acs_testing = pd.read_csv(filename+'.csv', sep=',')

# adapted
# training
filename = 'ge_raw_acs_adapt_train'
_data_acs_adapt_training = pd.read_csv(filename+'.csv', sep=',')
_data_acs_adapt_training
# testing
filename = 'ge_raw_acs_adapt_test'
_data_acs_adapt_testing = pd.read_csv(filename+'.csv', sep=',')

# %%
''' GC6-74 '''
# prepare data
############################################################
# training
labels_train = _data_training.loc[:, 'label']
# First 8 columns are sample information
sample_info_train = _data_training.loc[:, :"before_diagnosis_group"]
count_data_train = _data_training.loc[:, "7SK":]

X_train = count_data_train.to_numpy()

Label_Encoder = LabelEncoder()
y_categorical_train = labels_train.to_numpy().reshape(len(labels_train),)
y_train = np.abs(Label_Encoder.fit_transform(y_categorical_train) - 1)

# testing
labels_test = _data_testing.loc[:, 'label']
# First 8 columns are sample information
sample_info_test = _data_testing.loc[:, :"before_diagnosis_group"]
count_data_test = _data_testing.loc[:, "7SK":]

X_test = count_data_test.to_numpy()

Label_Encoder = LabelEncoder()
y_categorical_test = labels_test.to_numpy().reshape(len(labels_test),)
y_test = np.abs(Label_Encoder.fit_transform(y_categorical_test) - 1)

# %%
# Generate features
selector_output, final_feat, selected_feat, X_train_check = BorutaRFE(X_train, y_train)
# %%
selector_output_cv, final_feat_cv, selected_feat_cv, X_train_check_cv = BorutaRFECV(
    X_train, y_train)

# %%
# Predictive Performance
# ------------------------------------------------------------
# Internal results
fig, ax = plt.subplots()

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")

ax.plot(range(0, len(np.insert(selector_output_cv.grid_scores_, 0, 0))),
        np.insert(selector_output_cv.grid_scores_, 0, 0))
# %%
fig, ax = plt.subplots()
selector_output_cv.grid_scores_
ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")
ax.plot(range(1, len(selector_output_cv.grid_scores_) + 1), selector_output_cv.grid_scores_)
# %%
# External results

# Generation

# select feature set cut-off

''' Based on internal predictive performance results, the cut-off would be probably be
around 20 features, however based on the developmental procedure results the procedure is
capable of strong predictive performance reduction up to the final 5 features even, thus
this will be used for the test set results. '''

# calculate selected feature set predictive performance

# parameters
number_features = 10
selected_features = selected_feat
ranking = selector_output.ranking_
subset_list = selected_feat[ranking <= number_features]

preprocessing = "mrm"

classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=1),
    'SVM_linear': LinearSVC(dual=False),
    'SVM_rbf': SVC(kernel="rbf"),
    'GaussianNB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=1),
    'XGBoost': XGBClassifier(n_jobs=1)
}

classifier_names = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF', 'XGBoost']
# %%
output = single_eval(classifiers, subset_list, X_train, y_train, X_test, y_test, preprocessing)
# %%
# sensitivity
np.array(output[2])
# %%
# specificity

np.array(output[3])
# %%
# geometric mean of sensitivity and specificity
np.sqrt(np.multiply(np.array(output[2]), np.array(output[3])))
# %%
# roc curve
plt.figure
for i in range(len(classifiers)):
    fpr = output[4][i]
    tpr = output[5][i]
    # auc
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.8,
             label='{} AUC'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')

# %%
# Temporal group analysis
total_temp = {"0-6": 0, "6-12": 0, "12-18": 0, "18-24": 0, "(Missing)": 0}
num_identified_as = {"0-6": 0, "6-12": 0, "12-18": 0, "18-24": 0, "(Missing)": 0}

#
for classifier_num in range(0, 5):
    print('Results for classifier ' + classifier_names[classifier_num])
    pp_dict_per_temp_group = {"0-6": 0, "6-12": 0, "12-18": 0, "18-24": 0, "(Missing)": 0}
    for temp in ["0-6", "6-12", "12-18", "18-24", "(Missing)"]:
        # totals
        temporal_group = sample_info_test[sample_info_test['before_diagnosis_group'] == temp]
        total_temp[temp] += len(temporal_group)
        # correctly identified
        identified_as_cases = sample_info_test.loc[output[0][classifier_num] == 1]
        # split identified as cases
        num_identified_as[temp] += len(
            identified_as_cases[identified_as_cases['before_diagnosis_group'] == temp])
        pp_dict_per_temp_group[temp] = (
            len(identified_as_cases[identified_as_cases['before_diagnosis_group'] == temp])/len(temporal_group))
    print(pp_dict_per_temp_group)
# %%
# Selected feature names
gc6_selected_features = count_data_train.iloc[:, subset_list].columns
gc6_selected_features
gc6_first_phase_selected_features = count_data_train.iloc[:, selected_feat].columns
"C1QC" in gc6_first_phase_selected_features
gc6_first_phase_selected_features
# %%
''' ------------------------------------------------------------------------------------------------------- '''
''' ACS Original '''
# prepare data
############################################################
# training
_data_acs_training["time_to_diagnosis_group"].value_counts()
labels_acs_train = _data_acs_training.loc[:, 'label']
# First 8 columns are sample information
sample_info_acs_train = _data_acs_training.loc[:, :"time_to_diagnosis_group"]
count_data_acs_train = _data_acs_training.loc[:, "A1BG":]

X_acs_train = count_data_acs_train.to_numpy()

Label_Encoder = LabelEncoder()
y_categorical_acs_train = labels_acs_train.to_numpy().reshape(len(labels_acs_train),)
y_acs_train = np.abs(Label_Encoder.fit_transform(y_categorical_acs_train) - 1)
y_acs_train
# testing
labels_acs_test = _data_acs_testing.loc[:, 'label']
# First 8 columns are sample information
sample_info_acs_test = _data_acs_testing.loc[:, :"time_to_diagnosis_group"]
count_data_acs_test = _data_acs_testing.loc[:, "A1BG":]

X_acs_test = count_data_acs_test.to_numpy()

Label_Encoder = LabelEncoder()
y_categorical_acs_test = labels_acs_test.to_numpy().reshape(len(labels_acs_test),)
y_acs_test = np.abs(Label_Encoder.fit_transform(y_categorical_acs_test) - 1)
# %%
# Generate features
selector_output_acs, final_feat_acs, selected_feat_acs, X_train_check_acs = BorutaRFE(
    X_acs_train, y_acs_train)
# %%
selector_output_acs_cv, final_feat_acs_cv, selected_feat_acs_cv, X_train_check_acs_cv = BorutaRFECV(
    X_acs_train, y_acs_train)
# %%
# Internal results
fig, ax = plt.subplots()

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")

ax.plot(range(0, len(np.insert(selector_output_acs_cv.grid_scores_, 0, 0))),
        np.insert(selector_output_acs_cv.grid_scores_, 0, 0))
# %%
fig, ax = plt.subplots()

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")
ax.plot(range(1, len(selector_output_acs_cv.grid_scores_) + 1), selector_output_acs_cv.grid_scores_)
# %%
# External results

# Generation

# select feature set cut-off

''' Based on internal predictive performance results, the cut-off would be probably be
around 50 features, however based on the developmental procedure results the procedure is
capable of strong predictive performance reduction up to the final 5 features even, thus
this will be used for the test set results. '''

# calculate selected feature set predictive performance

# parameters
number_features = 5
selected_features = selected_feat_acs
ranking = selector_output_acs.ranking_
subset_list = selected_features[ranking <= number_features]


preprocessing = "mrm"

classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=1),
    'SVM_linear': LinearSVC(dual=False),
    'SVM_rbf': SVC(kernel="rbf"),
    'GaussianNB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=1),
    'XGBoost': XGBClassifier(n_jobs=1)
}
# %%
output = single_eval(classifiers, subset_list, X_acs_train,
                     y_acs_train, X_acs_test, y_acs_test, preprocessing)
# %%
# sensitivity
np.array(output[2])
# %%
# specificity
np.array(output[3])
# %%
# geometric mean of sensitivity and specificity
np.sqrt(np.array(output[2])*np.array(output[3]))
# %%
# Temporal group analysis
total_temp = {"0-180": 0, "181-360": 0, "361-540": 0, "541-720": 0, "(Missing)": 0}
num_identified_as = {"0-180": 0, "181-360": 0, "361-540": 0, "541-720": 0, "(Missing)": 0}

sample_info_acs_test['time_to_diagnosis_group'] = sample_info_acs_test['time_to_diagnosis_group'].fillna(
    '(Missing)')
len(output[0][3])
len(sample_info_acs_test)
for temp in "0-180", "181-360", "361-540", "541-720", "(Missing)":
    # totals
    temporal_group = sample_info_acs_test[sample_info_acs_test['time_to_diagnosis_group'] == temp]
    total_temp[temp] += len(temporal_group)
    # correctly identified
    identified_as_cases = sample_info_acs_test.loc[output[0][2] == 1]
    # split identified as cases
    num_identified_as[temp] += len(
        identified_as_cases[identified_as_cases['time_to_diagnosis_group'] == temp])

for classifier_num in range(0, 5):
    print('Results for classifier ' + classifier_names[classifier_num])
    pp_dict_per_temp_group = {"0-6": 0, "6-12": 0, "12-18": 0, "18-24": 0, "(Missing)": 0}
    for temp in ["0-6", "6-12", "12-18", "18-24", "(Missing)"]:
        # totals
        temporal_group = sample_info_test[sample_info_test['before_diagnosis_group'] == temp]
        total_temp[temp] += len(temporal_group)
        # correctly identified
        identified_as_cases = sample_info_test.loc[output[0][classifier_num] == 1]
        # split identified as cases
        num_identified_as[temp] += len(
            identified_as_cases[identified_as_cases['before_diagnosis_group'] == temp])
        pp_dict_per_temp_group[temp] = (
            len(identified_as_cases[identified_as_cases['before_diagnosis_group'] == temp])/len(temporal_group))
    print(pp_dict_per_temp_group)


total_temp
num_identified_as
# %%
# Selected feature names
count_data_acs_train.iloc[:, subset_list].columns
# %%
''' ACS adapted '''
# prepare data
############################################################

# split into temporal groups
_data_acs_adapt_training['time_to_diagnosis_group'].unique()
# get temporal cases
temp_case_indices = np.where(_data_acs_adapt_training['time_to_diagnosis_group'].fillna('(Missing)').str.contains(
    '(post Rx|0-180|181-360)'))[0]  # |post Rx|0-180|181-360|361-540|541-720|> 720
# get controls
control_indices = np.where(_data_acs_adapt_training['label'] == 'control')[0]
temp_control_indices = np.random.choice(control_indices, len(temp_case_indices)*2, replace=False)

# temporal train indices
temp_train_indices = np.concatenate([temp_case_indices, temp_control_indices])
# %%
# training
labels_acs_train = _data_acs_adapt_training.loc[:, 'label']
# First 8 columns are sample information
sample_info_acs_train = _data_acs_adapt_training.loc[:, :"time_to_diagnosis_group"]
count_data_acs_train = _data_acs_adapt_training.loc[:, "A1BG":]

X_acs_train = count_data_acs_train.to_numpy()

Label_Encoder = LabelEncoder()
y_categorical_acs_train = labels_acs_train.to_numpy().reshape(len(labels_acs_train),)
y_acs_train = np.abs(Label_Encoder.fit_transform(y_categorical_acs_train) - 1)
y_acs_train
# testing
labels_acs_test = _data_acs_adapt_testing.loc[:, 'label']
# First 8 columns are sample information
sample_info_acs_test = _data_acs_adapt_testing.loc[:, :"time_to_diagnosis_group"]
ungrouped_case_ind = (sample_info_acs_test['time_to_diagnosis_group'].isna()) & (
    sample_info_acs_test['label'] == 'case')
sample_info_acs_test.loc[ungrouped_case_ind, 'time_to_diagnosis_group'] = 'Ungrouped'
count_data_acs_test = _data_acs_adapt_testing.loc[:, "A1BG":]

X_acs_test = count_data_acs_test.to_numpy()

Label_Encoder = LabelEncoder()
y_categorical_acs_test = labels_acs_test.to_numpy().reshape(len(labels_acs_test),)
y_acs_test = np.abs(Label_Encoder.fit_transform(y_categorical_acs_test) - 1)
# %%

X_acs_train = X_acs_train[temp_train_indices]
y_acs_train = y_acs_train[temp_train_indices]
# %%
# Generate features
selector_output_acs, final_feat_acs, selected_feat_acs, X_train_check_acs = BorutaRFE(
    X_acs_train, y_acs_train)
# %%
selector_output_acs_cv, final_feat_acs_cv, selected_feat_acs_cv, X_train_check_acs_cv = BorutaRFECV(
    X_acs_train, y_acs_train)
# %%
# Internal results
fig, ax = plt.subplots()

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")

ax.plot(range(0, len(np.insert(selector_output_acs_cv.grid_scores_, 0, 0))),
        np.insert(selector_output_acs_cv.grid_scores_, 0, 0))
# %%
fig, ax = plt.subplots()

ax.set_xlabel("Number of features")
ax.set_ylabel("Internal Cross-validation\nPredictive Performance")
ax.plot(range(1, len(selector_output_acs_cv.grid_scores_) + 1), selector_output_acs_cv.grid_scores_)

# %%
# External results

# Generation

# select feature set cut-off

''' Based on internal predictive performance results, the cut-off would be probably be
around 20 features, however based on the developmental procedure results the procedure is
capable of strong predictive performance reduction up to the final 5 features even, thus
this will be used for the test set results. '''

# calculate selected feature set predictive performance

# parameters
number_features = 6
selected_features = selected_feat_acs
ranking = selector_output_acs.ranking_
subset_list = selected_features[ranking <= number_features]


preprocessing = "mrm"

classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=1),
    'SVM_linear': LinearSVC(dual=False),
    'SVM_rbf': SVC(kernel="rbf"),
    'GaussianNB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=1),
    'XGBoost': XGBClassifier(n_jobs=1)
}
# %%
output = single_eval(classifiers, subset_list, X_acs_train,
                     y_acs_train, X_acs_test, y_acs_test, preprocessing)
# %%
# sensitivity
np.array(output[2])
# %%
# specificity
np.array(output[3])
# %%
# geometric mean of sensitivity and specificity
np.sqrt(np.array(output[2])*np.array(output[3]))
# %%
# roc curve
plt.figure
for i in range(len(classifiers)):
    fpr = output[4][i]
    tpr = output[5][i]
    # auc
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.8,
             label='{} - {} AUC'.format(classifier_names[i], round(roc_auc, 2)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.ylabel('Sensitivity')
    plt.xlabel('1- Specificity')
# %%
# Temporal group analysis
total_temp = {"0-180": 0, "181-360": 0, "361-540": 0,
              "541-720": 0, "> 720": 0, "post Rx": 0,  "Ungrouped": 0, "(Missing)": 0}
num_identified_as = {"0-180": 0, "181-360": 0, "361-540": 0,
                     "541-720": 0, "> 720": 0, "post Rx": 0,  "Ungrouped": 0, "(Missing)": 0}

sample_info_acs_test['time_to_diagnosis_group'] = sample_info_acs_test['time_to_diagnosis_group'].fillna(
    '(Missing)')
for classifier_num in range(0, 5):
    print('Results for classifier ' + classifier_names[classifier_num])
    pp_dict_per_temp_group = {"0-180": 0, "181-360": 0, "361-540": 0,
                              "541-720": 0, "> 720": 0, "post Rx": 0,  "Ungrouped": 0, "(Missing)": 0}
    for temp in "0-180", "181-360", "361-540", "541-720", "> 720", "post Rx", "Ungrouped", "(Missing)":
        # totals
        temporal_group = sample_info_acs_test[sample_info_acs_test['time_to_diagnosis_group'] == temp]
        total_temp[temp] += len(temporal_group)
        # correctly identified
        identified_as_cases = sample_info_acs_test.loc[output[0][4] == 1]
        # split identified as cases
        num_identified_as[temp] += len(
            identified_as_cases[identified_as_cases['time_to_diagnosis_group'] == temp])
        pp_dict_per_temp_group[temp] = (
            len(identified_as_cases[identified_as_cases['time_to_diagnosis_group'] == temp])/len(temporal_group))
    print(pp_dict_per_temp_group)
    print("")
    print(num_identified_as)
    print(total_temp)


# total_temp = {"0-180": 0, "181-360": 0, "361-540": 0, "541-720": 0, "(Missing)": 0}
# num_identified_as = {"0-180": 0, "181-360": 0, "361-540": 0, "541-720": 0, "(Missing)": 0}
#
# sample_info_acs_test['time_to_diagnosis_group'] = sample_info_acs_test['time_to_diagnosis_group'].fillna(
#     '(Missing)')
# for classifier_num in range(0,5):
#     print('Results for classifier ' + classifier_names[classifier_num])
#     pp_dict_per_temp_group = {"0-180": 0, "181-360": 0, "361-540": 0, "541-720": 0, "(Missing)": 0}
#     for temp in "0-180", "181-360", "361-540", "541-720", "(Missing)":
#         # totals
#         temporal_group = sample_info_acs_test[sample_info_acs_test['time_to_diagnosis_group'] == temp]
#         total_temp[temp] += len(temporal_group)
#         # correctly identified
#         identified_as_cases = sample_info_acs_test.loc[output[0][2] == 1]
#         # split identified as cases
#         num_identified_as[temp] += len(
#             identified_as_cases[identified_as_cases['time_to_diagnosis_group'] == temp])
#         pp_dict_per_temp_group[temp] = (len(identified_as_cases[identified_as_cases['time_to_diagnosis_group'] == temp])/len(temporal_group))
#     print(pp_dict_per_temp_group)
#     print("")
#     print(num_identified_as)
#     print(total_temp)
# %%
# Selected feature names
count_data_acs_train.iloc[:, subset_list].columns
# %%
''' GC6 features on ACS test set '''
gc6_selected_features = ['HAUS5', 'PRDX3', 'SOCS1', 'CMTM3', 'C1QC']
gc6_subset_list = [count_data_acs_train.columns.get_loc(i) for i in gc6_selected_features]
gc6_subset_list
count_data_acs_train[gc6_selected_features]
# count_data_acs_train.columns.get_loc(gc6_selected_features[5])
# gc6_subset_list = [1726, 2687, 4870,  8320, 10624]
# %%
output = single_eval(classifiers, gc6_subset_list, X_acs_train,
                     y_acs_train, X_acs_test, y_acs_test, preprocessing)
# %%
# sensitivity
np.array(output[2])
# specificity
np.array(output[3])
# geometric mean of sensitivity and specificity
np.sqrt(np.array(output[2])*np.array(output[3]))
# %%
# roc curve
plt.figure
for i in range(len(classifiers)):
    fpr = output[4][i]
    tpr = output[5][i]
    # auc
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.8,
             label='{} - {} AUC'.format(classifier_names[i], round(roc_auc, 2)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.ylabel('Sensitivity')
    plt.xlabel('1- Specificity')
