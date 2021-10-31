'''
The following script combines all the important functions used for results evaluations.
'''
import pandas as pd
import numpy as np


def gmean(y_true, y_predicted):
    sensitivity = recall_score(y_true, y_predicted)
    specificity = recall_score(y_true, y_predicted, pos_label=0)
    error = np.sqrt(sensitivity*specificity)
    return error


def extract_boruta_list(boruta_output, kf_train_idxcs, X):
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
