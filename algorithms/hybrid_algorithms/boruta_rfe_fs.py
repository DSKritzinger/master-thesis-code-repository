'''
The following script is the Boruta-RFE feature selection algorithm.

Input
    X_train - training data (numpy array)
    y_train - training labels (numpy array)
Output
    selector_output - final fitted RFE model
    final_feat - final feature in the ranking process
    selected_feat - boruta phase selected features output
    X_train - to confirm input and outputs are correct

'''
# Imports
from utils.median_ratio_method import geo_mean, median_ratio_standardization, median_ratio_standardization_, median_ratio_standardization_log
from utils.boruta.boruta import BorutaPy  # forked master boruta_py
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE

from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score

import numpy as np
import pandas as pd
import time

# Functions


def gmean(y_true, y_predicted):
    sensitivity = recall_score(y_true, y_predicted)
    specificity = recall_score(y_true, y_predicted, pos_label=0)
    error = np.sqrt(sensitivity*specificity)
    return error


geometric_mean = make_scorer(gmean, greater_is_better=True)


def BorutaRFE(X_train, y_train):
    start = time.perf_counter()
    # Prepare data
    # --------------------------------------------------------
    # Standardization
    X_train = np.round(median_ratio_standardization_(X_train), 0)
    # First phase
    # --------------------------------------------------------
    print('Start first phase')
    # Boruta
    # initialize
    rf_1 = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7)
    boruta = BorutaPy(rf_1, n_estimators='auto', verbose=2, random_state=1, max_iter=250)
    # fit
    boruta.fit(X_train, y_train)
    # first phase selected feature
    confirmed = pd.DataFrame(X_train).columns[boruta.support_].to_list()
    tentative = pd.DataFrame(X_train).columns[boruta.support_weak_].to_list()
    selected = confirmed.copy()
    selected.extend(tentative)
    selected_feat = np.array(selected)

    # Second phase
    # --------------------------------------------------------
    print('Start second phase')
    # RFE
    # initialize
    rf_2 = RandomForestClassifier(n_jobs=1, class_weight='balanced', n_estimators=500)
    selector = RFE(rf_2, step=1, n_features_to_select=1)  # full ranking
    # fit
    selector_output = selector.fit(X_train[:, selected_feat], y_train)

    # second phase selected features
    # will output final selected feature set based on CV loop
    final_feat = selected_feat[selector_output.support_]

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
    print('Run complete')
    return selector_output, final_feat, selected_feat, X_train
