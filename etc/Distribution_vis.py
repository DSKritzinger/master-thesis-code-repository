# %%
# Imports
# Basics
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from median_ratio_method import geo_mean, median_ratio_standardization
import pandas as pd
import numpy as np
import pickle
import time
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
    #idx_fisher_score_list, idx_chi_score_list, idx_reliefF_score_list, idx_mim_score_list, idx_gini_score_list, idx_mrmr_list
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
# %%
pd.DataFrame(X).describe()
pd.DataFrame(np.round(median_ratio_standardization(X), 0)).describe()

X.shape

plt.hist(X[:, 11])

X[1, :]
plt.hist(np.log2(X[:, 11]+1))
plt.hist(np.log2(np.round(median_ratio_standardization(X), 0)[:, 11]+1))
np.round(median_ratio_standardization(X), 0)[1, :] + 1
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
plt.hist(X_sc[:, 11])

scaler = StandardScaler()
X_norm_sc = scaler.fit_transform(np.round(median_ratio_standardization(X), 0))
plt.hist(X_norm_sc[:, 11])
