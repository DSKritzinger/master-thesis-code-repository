'''
The following code aims to provide the evaluation results of the Boruta-RFE
feature selection method on the different temporal groupings.
'''
# %%
# Imports
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
# Evaluation functions
from eval_functions import predictive_ability, average_tanimoto_index
from sklearn.metrics import auc
# Data Prep functions
from sklearn.preprocessing import LabelEncoder
# Machine Learning Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Graphing
import seaborn as sns
from matplotlib import pyplot as plt
# Evaluation
from sklearn.model_selection import RepeatedStratifiedKFold

# classes
class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

# Classifier details
classifier_names = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM (lin)': LinearSVC(dual=False),
    'SVM (rbf)': SVC(kernel="rbf"),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1)
}

classifiers = {
    'KNN': KNeighborsClassifier(n_jobs=1),
    'SVM_linear': LinearSVC(dual=False),
    'SVM_rbf': SVC(kernel="rbf"),
    'GaussianNB': GaussianNB(),
    'RF': RandomForestClassifier(n_jobs=1),
    'XGBoost': XGBClassifier(n_jobs=1)
}
# %%
# Import datasets
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
y_categorical = labels.to_numpy().reshape(len(labels),)  # labels numpy array
# Change categorical labels to binary (controls - 0 and cases - 1)
Label_Encoder = LabelEncoder()
y = np.abs(Label_Encoder.fit_transform(y_categorical) - 1)
# Initialize variables
X_train = X
y_train = y
# %%
################################################################################################
# CV procedure variables
################################################################################################
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

for kf_train_index, kf_test_index in rskf.split(X_train, y_train):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)
 # %%
 ################################################################################################
 #   Load generated feature sets
 ################################################################################################
# %%
pickle_directory = ""
with open(pickle_directory+'ge_raw_24_BorutaRFE_data_characteristic_results_6', 'rb') as f:
  boruta_rfe_6 = pickle.load(
      f)
with open(pickle_directory+'ge_raw_24_BorutaRFE_data_characteristic_results_12', 'rb') as f:
 boruta_rfe_12 = pickle.load(
     f)
with open(pickle_directory+'ge_raw_24_BorutaRFE_data_characteristic_results_18', 'rb') as f:
 boruta_rfe_18 = pickle.load(
     f)
with open(pickle_directory+'ge_raw_24_BorutaRFE_data_characteristic_results_24', 'rb') as f:
 boruta_rfe_24 = pickle.load(
     f)
# %%
pickle_directory = ""
with open(pickle_directory+'ge_raw_24_BorutaRFECV_data_characteristic_results_6', 'rb') as f:
  boruta_rfe_CV_6 = pickle.load(
      f)
with open(pickle_directory+'ge_raw_24_BorutaRFECV_data_characteristic_results_12', 'rb') as f:
 boruta_rfe_CV_12 = pickle.load(
     f)
with open(pickle_directory+'ge_raw_24_BorutaRFECV_data_characteristic_results_18', 'rb') as f:
 boruta_rfe_CV_18 = pickle.load(
     f)
with open(pickle_directory+'ge_raw_24_BorutaRFECV_data_characteristic_results_24', 'rb') as f:
 boruta_rfe_CV_24 = pickle.load(
    f)
# %%
################################################################################################
# Predictive Performance
# ------------------------------------------------------------
# Internal results
fig, ax = plt.subplots()

data_effects_internal = {
    '0-6':boruta_rfe_CV_6[0],
    '0-12':boruta_rfe_CV_12[0],
    '0-18':boruta_rfe_CV_18[0],
    '0-24':boruta_rfe_CV_24[0],
}

for input_name, input in data_effects_internal.items():
    scores_list = []
    for i in range(0, 50):
        scores = input[i].grid_scores_
        scores = np.insert(scores, 0, 0)
        scores_list.append(scores)

    cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
        scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

    scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)
    scores_25 = np.nanmedian(cut_off_dataframe, axis=0)  # cut_off_dataframe.iloc[24]
    scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)


    ax.set_xlabel("Number of features")
    ax.set_ylabel("Internal Cross-validation\nPredictive Performance")
    # ax.plot(range(1, len(scores_5) + 1), scores_5, linestyle=':')
    # ax.plot(range(1, len(scores_45) + 1), scores_45, linestyle=':')
    ax.plot(range(1, len(scores_25) + 1), scores_25, label=input_name)
    # ax.fill_between(range(1, len(scores_5) + 1), scores_5, scores_45, alpha=0.3)
ax.grid()
sns.despine()
ax.legend()
# %%
# External results

data_effects_selected_features = {
    '0-6': {"one":boruta_rfe_6[2],"two":boruta_rfe_6[0]},
    '0-12':{"one":boruta_rfe_12[2],"two":boruta_rfe_12[0]},
    '0-18':{"one":boruta_rfe_18[2],"two":boruta_rfe_18[0]},
    '0-24':{"one":boruta_rfe_24[2],"two":boruta_rfe_24[0]},
}

# Generate External Predictive Performance Results
################################################################################################
# %%
data_effects_external = {}
for k, v in data_effects_selected_features.items():
    print(k)
    data_effects_external[k] = k
    '''Stage 1'''
    input_1 = data_effects_selected_features[k]['one']

    '''Stage 2'''
    input_2 = data_effects_selected_features[k]['two']

    results_list = []
    for j in range(1,len(input_1[0])):
        ''' Grab top x number of features '''
        number_features = j
        top_list = []
        for i in range(0, 50):
            # identify the top {{number_features}} features for each fold
            ranking = input_2[i].ranking_
            top = input_1[i][ranking <=number_features]
            top_list.append(top)
        top_list[0]
        preproc = 'ens'
        input = top_list
        result = predictive_ability(
            classifiers, top_list, X_train, y_train, num_repeats, num_splits, preproc)
        results_list.append(result)
    data_effects_external[k] = results_list

# %%
# Save External Predictive Performance Results
# with open('temporal_predictive_performance_per_feature_results', 'wb') as f:
#     pickle.dump(data_effects_external, f)
# Grab External Predictive Performance Results
with open('temporal_predictive_performance_per_feature_results', 'rb') as f:
    temp_pp_per_feature = pickle.load(
        f)
# %%
''' ----------------------------- Comparison Tests ----------------------------- '''

# Set figure styles
def set_style():
    sns.set(context="paper", font='serif', style="white", rc={"xtick.bottom": True,
                                                              "xtick.labelsize": "x-small",
                                                              "ytick.left": True,
                                                              "ytick.labelsize": "x-small",
                                                              "legend.fontsize": "x-small",
                                                              "ytick.major.size": 2,
                                                              "xtick.major.size": 2})


fig_width = 10
fig_height_scale = 1.2
# set golden ratio values
gr = (np.sqrt(5)-1)/2

selected_classifiers = ['KNN', 'SVM (lin)', 'SVM (rbf)', 'NB', 'RF']

# Combine internal and external predictive performance results
per_feature_comparison = {
    '0-6': {"external":temp_pp_per_feature["0-6"],"internal":boruta_rfe_CV_6[0]},
    '0-12':{"external":temp_pp_per_feature["0-12"],"internal":boruta_rfe_CV_12[0]},
    '0-18':{"external":temp_pp_per_feature["0-18"],"internal":boruta_rfe_CV_18[0]},
    '0-24':{"external":temp_pp_per_feature["0-24"],"internal":boruta_rfe_CV_24[0]},
}
# %%
zoom1 = 26
zoom2 = 0
set_style()
fig_width = 5.8
fig_height_scale = 3

# label
labels = {'0-6':'0-180','0-12':'0-360','0-18':'0-540','0-24':'0-720'}
fig, ax = plt.subplots(4,1, figsize=(fig_width, gr*fig_width*fig_height_scale))

ax = ax.ravel()
temp_auc = {}
temp_sens = {}
temp_spes = {}
j = 0
for k, v in per_feature_comparison.items():

    # Internal
    input = v['internal']
    scores_list = []
    for i in range(0, 50):
        scores = input[i].grid_scores_
        scores = np.insert(scores, 0, 0)
        scores_list.append(scores)

    # crop results to include at least 4 results per number of features
    cut_off_dataframe = pd.DataFrame(scores_list).transform(np.sort).loc[:, pd.DataFrame(
        scores_list).transform(np.sort).isnull().sum() <= (pd.DataFrame(scores_list).shape[0] - 3)]

    scores_5 = np.nanpercentile(cut_off_dataframe, q=10, axis=0)[zoom2:zoom1]
    scores_25 = np.nanmedian(cut_off_dataframe, axis=0)[zoom2:zoom1]
    scores_45 = np.nanpercentile(cut_off_dataframe, q=90, axis=0)[zoom2:zoom1]

    ax[j].plot(range(0, len(scores_5)), scores_5, linestyle=':', color="C9")
    ax[j].plot(range(0, len(scores_45)), scores_45, linestyle=':', color="C9")
    ax[j].plot(range(0, len(scores_25)), scores_25, color="C9", label='Internal CV')
    ax[j].fill_between(range(0, len(scores_5)), scores_5, scores_45, alpha=0.3, color="C9")

    # External
    input_ext = v['external']
    geomean_list = []
    sensitivity_list = []
    specificity_list = []
    auc_list= []
    main_auc_list = []
    tpr_list = []
    fpr_list = []
    geomean_list.append(np.zeros(len(classifier_names)))
    auc_list.append(np.zeros(len(classifier_names)))
    sensitivity_list.append(np.zeros(len(classifier_names)))
    specificity_list.append(np.zeros(len(classifier_names)))
    for i in range(0, len(input_ext)):
        sensitivity = pd.DataFrame(input_ext[i][2], columns=classifier_names.keys())
        sensitivity_list.append(np.nanmedian(sensitivity, axis=0))
        specificity = pd.DataFrame(input_ext[i][3], columns=classifier_names.keys())
        specificity_list.append(np.nanmedian(specificity,  axis=0))
        geomean = np.nanmean(np.sqrt(sensitivity*specificity), axis=0)
        geomean_list.append(geomean)
        fpr = pd.DataFrame(input_ext[i][4], columns=classifier_names.keys())
        fpr_list.append(fpr)
        tpr = pd.DataFrame(input_ext[i][5], columns=classifier_names.keys())
        tpr_list.append(tpr)
        auc_dict = {}
        for clas in range(len(classifiers)):
            classifier_name = list(classifier_names.keys())[clas]
            classifier_ = list(classifiers.keys())[clas]
            auc_list = []
            for q in range(0,50):
                auc_list.append(auc(fpr[classifier_name][q], tpr[classifier_name][q]))
            auc_dict[classifier_name] = auc_list
        main_auc_list.append(auc_dict)
    temp_auc[k] = main_auc_list
    temp_sens[k] = sensitivity_list
    temp_spes[k] = specificity_list
    for key in selected_classifiers:
        ax[j].plot(range(0, len(input_ext)+1),
                 pd.DataFrame(geomean_list, columns=classifier_names)[key], label=key)
    ax[j].grid()
    ax[j].set_ylim(0, 0.8)
    ax[j].set_xlim(-1, 25)
    # ax[j].set_ylabel("Cross Validation\nPredictive Performance")
    sns.despine()
    ax[j].set_title(labels[k] + ' days')
    j+=1

ax[j-1].set_xlabel("Number of Features")

# set common y axis label
fig.text(-0.01, 0.5, 'Cross-Validation Predictive Performance',
         ha='center', va='center', rotation='vertical')
# Put the legend out of the figure
handles, labels = ax[0].get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.525, 0), loc="lower center", ncol=6)
fig.tight_layout()
fig.subplots_adjust(bottom=0.06)
# plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/DataTemporalityEffects.png",
#             bbox_inches="tight", dpi=1000)
# %%
# For AUC results
temp_auc.keys()
np.mean(pd.DataFrame(temp_auc['0-6'][4]), axis=0)
np.std(pd.DataFrame(temp_auc['0-6'][4]), axis=0)
# %%
sens_spes = pd.DataFrame([])
pd.DataFrame(temp_spes['0-24'][5], index=classifier_names.keys())
pd.DataFrame(temp_sens['0-12'][5], index=classifier_names.keys())
# %%
'''
All the above approaches
'''
# %%
# Per temporal group comparison
# ------------------------------------------------------------

sample_info
kf_test_idxcs
test_sample_info_list = []
for test_ind in kf_test_idxcs:
    test_sample_info = sample_info[['sample_id','subject_id','site','gender','label','before_diagnosis','age_group','before_diagnosis_group']].iloc[test_ind]
    test_sample_info_list.append(test_sample_info)

len(kf_test_idxcs[0])
len(test_sample_info_list)
# temporal grouping | num features | predictiver performance metric | fold | learning algorithm
temp_pp_per_feature["0-6"][0][0][0][3]
len(temp_pp_per_feature["0-6"][0][0][0][3])
test_sample_info_list[0]
len(test_sample_info_list[0])

test_sample_info_list[0].loc[temp_pp_per_feature["0-6"][5][0][0][5] ==1]


# %%
# Prediction results target information split
# ------------------------------------------------------------
# Determine total cases for each temporal grouping
for classifier_num in range(0,5):
    print('Results for classifier ' + list(classifier_names.keys())[classifier_num])
    for temp_1 in ["0-6","0-12","0-18","0-24"]:
        print('Results for ' + temp_1)
        tester = temp_pp_per_feature[temp_1]
        total_temp = {"0-6":0,"6-12":0,"12-18":0,"18-24":0, "(Missing)":0}
        num_identified_as = {"0-6":0,"6-12":0,"12-18":0,"18-24":0, "(Missing)":0}
        pp_list_per_temp_group = []

        total_case = 0
        total_control = 0
        for i in range(50):
            pp_dict_per_temp_group = {"0-6":0,"6-12":0,"12-18":0,"18-24":0, "(Missing)":0}
            for temp in ["0-6","6-12","12-18","18-24","(Missing)"]:
                # totals
                temp_fold = test_sample_info_list[i][test_sample_info_list[i]['before_diagnosis_group'] == temp]
                total_temp[temp] += len(temp_fold)
                # correctly identified
                # temporal grouping | num features | predictive performance metric | fold | learning algorithm
                identified_as_cases = test_sample_info_list[i].loc[tester[5][0][i][classifier_num] == 1] # identified cases
                # split identified as cases
                num_identified_as[temp] += len(identified_as_cases[identified_as_cases['before_diagnosis_group'] == temp])
                if len(temp_fold) != 0:
                    pp_dict_per_temp_group[temp] = (len(identified_as_cases[identified_as_cases['before_diagnosis_group'] == temp])/len(temp_fold))
            pp_list_per_temp_group.append(pp_dict_per_temp_group)
        print(pd.DataFrame.from_dict(pp_list_per_temp_group, orient='columns').mean())
        print('')
# %%
# -- 3 20

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

# -- 3 10

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}


# -- 3 5

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

# -- 2

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

# -- 4

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

# -- 5

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

# -- 1

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}

total_temp
num_identified_as
{x:float(num_identified_as[x])/total_temp[x] for x in ["0-6","6-12","12-18","18-24","(Missing)"]}
# %%
# Similarity
# ------------------------------------------------------------
# First boruta
stage = 'one'
set_list = data_effects_selected_features

num_overlap_list2 = []
for key_1, filter_output_1 in set_list.items():
    one = pd.unique(pd.DataFrame(set_list[key_1][stage]).values.ravel())
    num_overlap_list1 = []
    for key_2, filter_output_2 in set_list.items():
        two = pd.unique(pd.DataFrame(set_list[key_2][stage]).values.ravel())
        overlapping_features = np.intersect1d(one, two)
        num_overlap = len(overlapping_features)
        num_overlap_list1.append(num_overlap)
    num_overlap_list2.append(num_overlap_list1)

pd.DataFrame(num_overlap_list2)
# %%
# Second Boruta-RFE

# threshold sets
thres = 5
boruta_rfe_6th_list = []
boruta_rfe_12th_list = []
boruta_rfe_18th_list = []
boruta_rfe_24th_list = []
for i in range(50):
    boruta_rfe_6th_list.append(boruta_rfe_6[2][i][boruta_rfe_6[0][i].ranking_ <= thres])
    boruta_rfe_12th_list.append(boruta_rfe_12[2][i][boruta_rfe_12[0][i].ranking_ <= thres])
    boruta_rfe_18th_list.append(boruta_rfe_18[2][i][boruta_rfe_18[0][i].ranking_ <= thres])
    boruta_rfe_24th_list.append(boruta_rfe_24[2][i][boruta_rfe_24[0][i].ranking_ <= thres])

set_list = {
    '0-6': boruta_rfe_6th_list,
    '0-12': boruta_rfe_12th_list,
    '0-18': boruta_rfe_18th_list,
    '0-24': boruta_rfe_24th_list,
}
num_overlap_list2 = []
for key_1, filter_output_1 in set_list.items():
    one = pd.unique(pd.DataFrame(set_list[key_1]).values.ravel())
    num_overlap_list1 = []
    for key_2, filter_output_2 in set_list.items():
        two = pd.unique(pd.DataFrame(set_list[key_2]).values.ravel())
        overlapping_features = np.intersect1d(one, two)
        print(overlapping_features)
        num_overlap = len(overlapping_features)
        num_overlap_list1.append(num_overlap)
    num_overlap_list2.append(num_overlap_list1)
    print("_______")

pd.DataFrame(num_overlap_list2)
# %%
# Stability
# ------------------------------------------------------------

# First Boruta
print(average_tanimoto_index(boruta_rfe_6[2]))
print(average_tanimoto_index(boruta_rfe_12[2]))
print(average_tanimoto_index(boruta_rfe_18[2]))
print(average_tanimoto_index(boruta_rfe_24[2]))
# %%
# Second Boruta-RFE

# threshold sets
thres = 5
boruta_rfe_6th_list = []
boruta_rfe_12th_list = []
boruta_rfe_18th_list = []
boruta_rfe_24th_list = []
for i in range(50):
    boruta_rfe_6th_list.append(boruta_rfe_6[2][i][boruta_rfe_6[0][i].ranking_ <= thres])
    boruta_rfe_12th_list.append(boruta_rfe_12[2][i][boruta_rfe_12[0][i].ranking_ <= thres])
    boruta_rfe_18th_list.append(boruta_rfe_18[2][i][boruta_rfe_18[0][i].ranking_ <= thres])
    boruta_rfe_24th_list.append(boruta_rfe_24[2][i][boruta_rfe_24[0][i].ranking_ <= thres])

set_list = {
    '0-6': boruta_rfe_6th_list,
    '0-12': boruta_rfe_12th_list,
    '0-18': boruta_rfe_18th_list,
    '0-24': boruta_rfe_24th_list,
}

print(average_tanimoto_index(set_list['0-6']))
print(average_tanimoto_index(set_list['0-12']))
print(average_tanimoto_index(set_list['0-18']))
print(average_tanimoto_index(set_list['0-24']))

# %%
len(pd.DataFrame(boruta_rfe_6[1]).values.ravel())
one =pd.unique(pd.DataFrame(boruta_rfe_6[1]).values.ravel())
len(one)
len(pd.DataFrame(boruta_rfe_12[1]).values.ravel())
two = pd.unique(pd.DataFrame(boruta_rfe_12[1]).values.ravel())
len(two)
len(np.intersect1d(one, two))

inter_list = []
one_len_list = []
two_len_list = []
for i in range(50):
    one = pd.DataFrame(boruta_rfe_6[2][i])
    one_len = len(one)
    one_len_list.append(one_len)
    two = pd.DataFrame(boruta_rfe_24[2][i])
    two_len = len(two)
    two_len_list.append(two_len)
    intersect = len(np.intersect1d(one, two))
    inter_list.append(intersect)

np.mean(inter_list)
np.max(inter_list)
np.min(inter_list)
np.median(inter_list)

np.mean(one_len_list)

np.mean(two_len_list)

boruta_rfe_6[2][0][boruta_rfe_6[0][0].ranking_ <= 20]
boruta_rfe_12[2][0][boruta_rfe_12[0][0].ranking_ <= 20]
pd.unique(pd.DataFrame(set_list_th[filter1]).values.ravel())
pd.DataFrame(boruta_rfe_6[2])

boruta_rfe_6[1][0]
average_tanimoto_index(boruta_rfe_6[2])
average_tanimoto_index(boruta_rfe_12[2])
average_tanimoto_index(boruta_rfe_18[2])
average_tanimoto_index(boruta_rfe_24[2])
