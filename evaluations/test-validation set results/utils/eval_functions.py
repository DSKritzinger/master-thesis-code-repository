'''
The following module consists of a set of functions aimed at the evaluation
of the developmental procedure (10 fold x 5 cross-validation) generated 
feature sets.

Functions:
- Tanimoto Index: Determines similarity between two feature subsets.
- Average Tanimoto Index: Determines stability of a feature selection 
                          method across k-folds.
- Intersystem Average Tanimoto Index: Determines similarity between different methods
- Predictive ability: Determine the predictive abilities of the selected features for a varity of classifiers in terms of sensitivity and specificity

This specific code is setup for the preprocessing of the real gc6-74 matched datasets.
'''
# Imports
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_curve
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
# Standardization
from median_ratio_method import geo_mean, median_ratio_standardization

# Functions

# Tanimoto Index

def tanimoto_index(idx_1, idx_2):
    '''
    Calculates tanimoto index similarity measure
    between two feature sets.
    Input:
        idx_1 - list of identified gene indices
        idx_2 - list of identified gene indices
    Output:
          - tanimoto_index (similarity between feature sets)
    '''
    n_1 = len(idx_1)    # the number of elements in feature set 1
    n_2 = len(idx_2)    # the number of elements in feature set 2
    # the number of elements that is in both feature sets 1 and 2
    n_12 = sum(np.isin(idx_1, idx_2))
    # tanimoto distance (according to P.Somol, et al. 2010)
    tanimoto_distance = (n_1 + n_2 - 2*n_12)/(n_1 + n_2 - n_12)
    # tanimoto index (according to P.Somol, et al. 2010)
    tanimoto_index = 1 - tanimoto_distance
    return tanimoto_index

# Average Tanimoto Index


def average_tanimoto_index(selected_set_list):
    '''
    Calculates average tanimoto index, a measure of stability between
    repeated list of generated feature sets.
    Input:
        selected_set_list - list of generated feature sets from repeated
                            feature set generatinon procedure.
    Output:
          - average_tanimoto_index (stability)
    '''
    num_subsets = len(selected_set_list)
    sum_2 = 0
    for i in range(0, num_subsets-1):
        sum_1 = 0
        for j in range(i+1, num_subsets):
            sum_1 += tanimoto_index(selected_set_list[i], selected_set_list[j])
        sum_2 += sum_1
    # average tanimoto index (according to P.Somol, et al. 2010)
    ATI = (sum_2*2)/(num_subsets*(num_subsets-1))
    return ATI

# Predictive performance


def predictive_ability(classifiers, subset_list, X, y, repeats, splits, preprocessing):
    '''
    Evaluates predictive performance of generated feature sets over k-folds,
    through multiple metrics.
    Input:
        classifiers - dictionary of classifiers to be used for predictive 
                      performance result generation.
        subset_list - list of k-fold generated feature sets.
        X - numpy array (shape - n_samples, n_features) of gene count data
        y - numpy array (shape - n_samples, 1) of sample target outcomes
        repeats - cross-validation number of repeats
        splits - cross-valiation number of folds
        preprocessing - string representing preprocessing method to be implemented, 
                        mrm - median ratio method
                        mrm_log - median ratio method with log normalization pre
                                  feature selection
                        mrm_loglog - median ratio method with log normalization 
                                     pre and post feature selection 
    Output:
    predict_list, acc_list, sensitivity_list, specificity_list, fpr_list, tpr_list,
          predict_list - list of list with each sample predicted outcome
          acc_list - list of accuracies generated from each cv fold loop's feature sets
          sensitivity_list - list of sensitivities generated from each cv fold loop's 
                             feature sets
          specifivity_list - list of specificities generated from each cv fold loop's 
                             feature sets      
          fpr_list - list of false positive rates generated from each cv fold loop's 
                     feature sets  
          tpr_list - list of true positive rates generated from each cv fold loop's
                     feature sets              
    '''

    print("Preprocessing procedure employed: " + preprocessing)
    i = 0    # CV fold counter
    # initialize function output variables
    acc_list = []
    sensitivity_list = []
    specificity_list = []
    fpr_list = []
    tpr_list = []
    predict_list = []

    # generate CV indices
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=0)
    # CV procedure
    for kf_train_index, kf_test_index in rskf.split(X, y):
        # k-fold indices
        X_train_f, X_test_f = X[kf_train_index], X[kf_test_index]
        y_train_f, y_test_f = y[kf_train_index], y[kf_test_index]
        # apply relevant preprocessing
        if preprocessing == "mrm":
            # apply deseq median-ratio-method standardization
            print("Applying: " + preprocessing)
            X_train_f = np.round(median_ratio_standardization(X_train_f), 0)
            X_test_f = np.round(median_ratio_standardization(X_test_f), 0)
        elif preprocessing == "mrm_log" or preprocessing == "ens":
            print("Applying: " + preprocessing)
            # apply deseq median-ratio-method standardization
            X_train_f = np.round(median_ratio_standardization(X_train_f), 0)
            X_test_f = np.round(median_ratio_standardization(X_test_f), 0)
        elif preprocessing == "mrm_log_log":
            print("Applying: " + preprocessing)
            # apply deseq median-ratio-method standardization
            X_train_f_ = np.round(median_ratio_standardization(X_train_f), 0)
            X_test_f_ = np.round(median_ratio_standardization(X_test_f), 0)
            # apply log normalization
            X_train_f = np.log2(X_train_f_+1)
            X_test_f = np.log2(X_test_f_+1)

        # Extract only selected features from training/testing fold
        X_train_sel = X_train_f[:, subset_list[i]]
        X_test_sel = X_test_f[:, subset_list[i]]

        # print function progress
        print("Fold ", i, " of 50")

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
                X_train_sel_sm, y_train_f_sm = sm.fit_resample(
                    X_train_sel_sc, y_train_f)  # (if smote)
                # X_train_sel_sm, y_train_f_sm = X_train_sel_sc, y_train_f  # (if no smote)

                X_train_in = X_train_sel_sm
                y_train_in = y_train_f_sm
                X_test_in = X_test_sel_sc
            else:
                print('Not Standardizing')
                # class imbalance rectification
                sm = SMOTE(random_state=42)
                # application of SMOTE, BorderlineSMOTE
                X_train_sel_sm, y_train_f_sm = sm.fit_resample(
                    X_train_sel, y_train_f)  # (if smote)
                # X_train_sel_sm, y_train_f_sm = X_train_sel, y_train_f  # (if no smote)

                X_train_in = X_train_sel_sm
                y_train_in = y_train_f_sm
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
            acc = accuracy_score(y_test_f, y_test_predict)
            clfs_acc_list.append(acc)
            # recall/sensitivity
            # with class 1 as representing a positive
            sensitivity = recall_score(y_test_f, y_test_predict)
            clfs_sensitivity_list.append(sensitivity)
            # specificity
            # with class 0 as representing a negative
            specificity = recall_score(y_test_f, y_test_predict, pos_label=0)
            clfs_specificity_list.append(specificity)
            # ROC variables generation
            # for classifiers which produce confidence scores
            if clf_key in ('SVM_linear', 'SVM_rbf'):
                y_score = clf.decision_function(X_test_in)  # confidence score
                # False positive rate & True positive rate
                fpr, tpr, threshold = roc_curve(y_test_f, y_score)
                clfs_fpr_list.append(fpr)
                clfs_tpr_list.append(tpr)
            # for classifiers which produce probability estimations
            else:
                y_probas = clf.predict_proba(X_test_in)  # probability estimations
                y_score = y_probas[:, 1]  # confidence score
                # False positive rate & True positive rate
                fpr, tpr, threshold = roc_curve(y_test_f, y_score)
                clfs_fpr_list.append(fpr)
                clfs_tpr_list.append(tpr)

        # Output lists for each fold
        # Predictions
        predict_list.append(clfs_predict_list)
        # Accuracy
        acc_list.append(clfs_acc_list)
        # Sensitivity
        sensitivity_list.append(clfs_sensitivity_list)
        # Specificity
        specificity_list.append(clfs_specificity_list)
        # ROC
        fpr_list.append(clfs_fpr_list)
        tpr_list.append(clfs_tpr_list)

        # update counter
        i += 1

    # Accuruacy, sensitivity, specificity and precision classifier mean and standard deviation
    # Convert to numpy arrays
    acc_list = np.array(acc_list)
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)
    # Result averages
    acc_average = np.mean(acc_list, axis=0)
    sensitivity_average = np.mean(sensitivity_list, axis=0)
    specificity_average = np.mean(specificity_list, axis=0)

    # Result standard deviations
    acc_std = np.std(acc_list, axis=0)
    sensitivity_std = np.std(sensitivity_list, axis=0)
    specificity_std = np.std(specificity_list, axis=0)

    # Append lists
    acc_list = np.append(acc_list, [acc_average, acc_std], axis=0)
    sensitivity_list = np.append(sensitivity_list, [sensitivity_average, sensitivity_std], axis=0)
    specificity_list = np.append(specificity_list, [specificity_average, specificity_std], axis=0)

    return predict_list, acc_list, sensitivity_list, specificity_list, fpr_list, tpr_list,
