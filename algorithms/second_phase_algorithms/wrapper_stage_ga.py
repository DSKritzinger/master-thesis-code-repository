'''
The following script implements the second phase GA feature selection method.The 
implementation is specifically focussed on the generation of feature sets for the 
hybrid method developmental procedure (10 fold x 5 cross-validation), thus various 
variables can be tested.

As the cross-validation procedure is computationally intensive, a multiprocessing 
approach was implemented for use on a high performance compute cluster (many core 
system for ideal performance).
'''
# Imports
import pandas as pd
import numpy as np
import time
import dill
from pathos.multiprocessing import ProcessPool
import pickle
import random
# Feature Selection methods
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.similarity_based import reliefF
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.statistical_based import gini_index
import pygad
# Estimators
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
# Standardization
from utils.median_ratio_method import geo_mean, median_ratio_standardization, median_ratio_standardization_, median_ratio_standardization_log
# Scaling
from sklearn.preprocessing import StandardScaler
# Metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_curve, precision_recall_curve

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# %%
################################################################################################
# Functions
# rank filter method output score indices
'''
input:  ranker_score_lists = ranker filter scores (order: fisher, chi, reliefF, mim, gini) and fold sample indices
output: ordered by score, ranker filter method indices (order: fisher, chi, reliefF, mim, gini)
'''


def rank_rank_dict(ranker_score_dict):
    '''
    Function for the rank sorting of the first phase ranker
    generated feature sets.
    '''
    # extract features from rank_rank() output
    fisher_score_list = ranker_score_dict['Fisher-Score']
    chi_score_list = ranker_score_dict['Chi-Square']
    reliefF_score_list = ranker_score_dict['ReliefF']
    mim_score_list = ranker_score_dict['Info Gain']
    gini_score_list = ranker_score_dict['Gini Index']
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

    return idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_mim_list, idx_gini_list


# apply threshold to ranker method outputs ('top-k')
'''
input:  ranked_ranker_lists = ranked ranker filter indices (order: fisher, chi, reliefF, mim, gini, mrmr)
        treshold = # of genes to select
ouptput: 'top-k' ranker filter method indices (order: fisher, chi, reliefF, mim, gini, mrmr)
'''


def rank_thres(ranked_ranker_lists, threshold):
    '''
    Function for the thresholding of he rank sorted first phase 
    ranker generated feature sets.
    '''
    list_th_out = []
    for list in ranked_ranker_lists:
        list_th = [item[0:threshold] for item in list]
        list_th_out.append(list_th)
    return list_th_out

# Preprocessing class initialization

class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

############################################Import Data#########################################
# %%
directory = "C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Method Development/Developmental sets/"
filename = 'ge_raw_6'
# Import dataset
_data = pd.read_csv(directory+filename+'.csv', sep=',')
_data
# Extract labels, sample id's and count data from imported data
labels = _data.loc[:, 'label']
# For GC6-74
sample_info = _data.loc[:, :"before_diagnosis_group"]  # First 8 columns are sample information
count_data = _data.loc[:, "7SK":]
sum(labels == "case")
################################################################################################
# Initialize data for evaluation
# %%
# Initialize data for input into feature selection and classification
X_train = count_data.to_numpy()  # count matrix numpy array
X = X_train
y_categorical = labels.to_numpy().reshape(len(labels),)  # labels numpy array
# Change categorical labels to binary (controls - 0 and cases - 1)
Label_Encoder = LabelEncoder()
y_train = np.abs(Label_Encoder.fit_transform(y_categorical) - 1)
y_train
############################################Split Data##########################################
# %%
# Thereafter for Validation: apply stratified K-fold data splits
num_splits = 10
num_repeats = 5
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
#                                 Import Ensemble Selected Features
################################################################################################
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xfilter outputsx/'

# import filter ensemble output
with open(filter_pickle_directory+filename+'_filter_stage_105', 'rb') as f:
    filter_set_105 = pickle.load(f)
# extract scores and save to dict
fisher_score_list, chi_score_list, reliefF_score_list, mim_score_list, gini_score_list, _ = filter_set_105
fs_filter_set_scores = {
    'ReliefF': reliefF_score_list,
    'Chi-Square': chi_score_list,
    'Fisher-Score': fisher_score_list,
    'Info Gain': mim_score_list,
    'Gini Index': gini_score_list,
}
# Rank feature indices based on scores
ranked_filter_set = rank_rank_dict(fs_filter_set_scores)
# %%
# Select filter threshold
# -----------------
threshold_feats = 8
# -----------------
# %%
# Apply thresholding for ensemble
idx_fisher_list_th_e, idx_chi_list_th_e, idx_reliefF_list_th_e, idx_mim_list_th_e, idx_gini_list_th_e = rank_thres(
    ranked_filter_set, threshold_feats)

# Initialize feature list
idx_ensemble_list = []
# append features from different methods together
for i in range(0, (num_repeats*num_splits)):
    ensembled_features = np.append(idx_fisher_list_th_e[i], [
                                   idx_chi_list_th_e[i], idx_reliefF_list_th_e[i], idx_mim_list_th_e[i], idx_gini_list_th_e[i]])
    ensembled_features
    # remove features which are duplicated in the list
    ensembled_features = np.array(list(dict.fromkeys(ensembled_features)))
    # make list of every folds selected features
    idx_ensemble_list.append(ensembled_features)

# %%
################################################################################################
#                                 Import Boruta Selected Features
################################################################################################
# boruta
boruta_pickle_directory = 'D:/Thesis_to_big_file/xboruta outputsx/'

# n_est| iter | perc | depth | alpha
# 'auto', 250, 100, 7, 0.01
with open(boruta_pickle_directory+filename+'_boruta_filter_stage_105_16', 'rb') as f:
    boruta_out16 = pickle.load(f)


def extract_boruta_list(boruta_output):
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

print('# of selected estimators: "auto"')
confirmed_list, tentative_list, selected_list = extract_boruta_list(boruta_out16)

# %%
################################################################################################
#                                        Evaluation Parameters
################################################################################################
# Evaluation Measure

def gmean(y_true, y_predicted):
    sensitivity = recall_score(y_true, y_predicted)
    specificity = recall_score(y_true, y_predicted, pos_label=0)
    error = np.sqrt(sensitivity*specificity)
    return error

geometric_mean = make_scorer(gmean, greater_is_better=True)
eval_measure = geometric_mean

# Pipeline combinations
# Initialize
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
# Pipelines
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
voting_classifier_pipeline_combo = VotingClassifier(estimators=[('SVM_rbf', pipeline_SVMrbf), ('NB', pipeline_NB), ('KNN', pipeline_KNN)],
                                                    voting='soft')
pipe = voting_classifier_pipeline_combo

# First phase selected features
input_set = selected_list  # idx_ensemble_list
# %%
################################################################################################
#                                          Main function
################################################################################################

def ga_wrapper_stage(train_idx, selected_features):
    # create train and test data folds
    X_train_f = X_train[train_idx]
    y_train_f = y_train[train_idx]

    # Genetic Algorithm Functions

    def reduce_features(solution, features):
        selected_elements_indices = np.where(solution == 1)[0]
        reduced_features_ind = features[selected_elements_indices]
        return reduced_features_ind

    def on_start(ga_instance):
        print("\nStarting Genetic Algorithm")

    def on_fitness(ga_instance, population_fitness):
        print("\nPopulation Fitness Evaluated")
        print('Population fitness - ')
        print(population_fitness)
        check1 = np.array(population_fitness)
        print("this is the population np array and its max")
        print(check1)
        check2 = np.max(population_fitness)
        print(check2)
        try:
            max_fit_idx = np.where(check1 == check2)[0][0]
        except:
            print(population_fitness)
            print(ga_instance.generations_completed)
            print(ga_instance.population)
            exit()

            print("An exception occurred")
        print(max_fit_idx)
        print("Population number of selected features - ")
        print(num_selected_features_list)
        print("Population predictive performance - ")
        print(pred_score_mean_list)
        max_fit_pop = ga_instance.population[max_fit_idx]
        max_fit_pop_list.append(max_fit_pop)
        max_fit_n_feat = num_selected_features_list[max_fit_idx]
        max_fit_n_feat_list.append(max_fit_n_feat)
        max_fit_mean_pred = pred_score_mean_list[max_fit_idx]
        max_fit_mean_pred_list.append(max_fit_mean_pred)
        max_fitness = np.max(population_fitness)
        max_fitness_list.append(max_fitness)
        print("\nMax:")
        if ga_instance.generations_completed >= 1:
            gen_best_sol = ga_instance.best_solution()
            print("Best Solution and Index: " + str([index for index, value in enumerate(
                gen_best_sol[0]) if value == 1]) + " - " + str(gen_best_sol[2]))
        print("Fitness: " + str(np.max(population_fitness)))
        print("Predictive Performance: " + str(max_fit_mean_pred))
        print("Number of features: " + str(max_fit_n_feat))
        avg_pop_n_feat = np.mean(num_selected_features_list)
        avg_pop_n_feat_list.append(avg_pop_n_feat)
        avg_pop_mean_pred = np.mean(pred_score_mean_list)
        avg_pop_mean_pred_list.append(avg_pop_mean_pred)
        avg_pop_fitness = np.mean(population_fitness)
        avg_pop_fitness_list.append(avg_pop_fitness)
        print("\nAverage:")
        print("Fitness: " + str(avg_pop_fitness))
        print("Predictive Performance: " + str(avg_pop_mean_pred))
        print("Number of features: " + str(avg_pop_n_feat))
        num_selected_features_list.clear()
        pred_score_mean_list.clear()

    def on_parents(ga_instance, selected_parents):
        print("\nParents Selected by " + parent_selection_type + " - ")
        print("Number of selected parents: " + str(len(selected_parents)))
        for parent in selected_parents:
            print([index for index, value in enumerate(parent) if value == 1])

    def on_crossover(ga_instance, offspring_crossover):
        print("\nCrossover Applied by " + crossover_type + " - ")
        print("Number of offspring generated: " + str(len(offspring_crossover)))
        # print(offspring_crossover)

    def on_mutation(ga_instance, offspring_mutation):
        print("\nMutation Applied by " + mutation_type + " - ")
        #print("Offspring Mutations: " + str(len(offspring_mutation)))
        # print(offspring_mutation)

    def on_generation(ga_instance):
        print("\nGeneration: " + str(ga_instance.generations_completed))
        print("#######################################################")

    def on_stop(ga_instance, last_population_fitness):
        print("---------------- Genetic Algorithm Complete ----------------")
        finish = time.perf_counter()
        print(f'\nFinished in {round(finish-start, 2)} second(s)')

    # set fitness function (with training fold)
    def fitness_func(solution, solution_idx):
        # number of selected features
        print("FITNESS evaluation running")
        num_selected_features = sum(solution)
        num_selected_features_list.append(num_selected_features)
        # extract features from first phase to be tested
        reduced_features_ind = reduce_features(solution, selected_features)
        X_train_f_sel = X_train_f[:, reduced_features_ind]
        # evalate predictive performance of solution
        pred_scores = cross_val_score(pipe, X_train_f_sel,
                                      y_train_f, cv=5, scoring=eval_measure)
        # calculate mean cross-validation results
        print(pred_scores)
        if np.isnan(pred_scores).any():
            pred_score_mean = 0
        else:
            pred_score_mean = np.mean(pred_scores)
        pred_score_mean_list.append(pred_score_mean)
        # print(num_selected_features)
        # calculate solution fitness based on predictive performance and solution size
        fitness = (1-imp_weight)*pred_score_mean + imp_weight * \
            (1-num_selected_features/len(selected_features))

        print("FITNESS evaluation finished")
        return fitness


    # parameters
    imp_weight = 0.3
    num_generations = 5
    sol_per_pop = 50
    num_parents_mating = np.uint8(sol_per_pop/2)
    num_genes = len(selected_features)
    parent_selection_type = "rws"  # "sus","rank"
    keep_parents = 1
    crossover_type = "uniform"  # "single_point","two_points"
    #crossover_probability = 0.8
    mutation_type = "random"
    mutation_probability = 0.1
    #mutation_percent_genes = 10
    gene_space = [0, 1]

    # set ga_instance variables with updated fitness function
    ga_instance = pygad.GA(num_generations=num_generations,
                        sol_per_pop=sol_per_pop,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_func,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        # crossover_probability=crossover_probability,
                        mutation_type=mutation_type,
                        mutation_probability=mutation_probability,
                        # mutation_percent_genes=mutation_percent_genes,
                        gene_space=gene_space,
                        on_start=on_start,
                        on_fitness=on_fitness,
                        on_parents=on_parents,
                        on_crossover=on_crossover,
                        on_mutation=on_mutation,
                        on_generation=on_generation,
                        on_stop=on_stop)
    # initialize output lists
    max_fit_pop_list = []
    max_fitness_list = []
    max_fit_n_feat_list = []
    max_fit_mean_pred_list = []
    avg_pop_n_feat_list = []
    avg_pop_mean_pred_list = []
    avg_pop_fitness_list = []
    num_selected_features_list = []
    pred_score_mean_list = []

    # run ga instance
    start = time.perf_counter()
    np.seterr(divide='ignore')
    random.seed(1)
    ga_instance.run()
    np.seterr(divide='warn')

    return ga_instance, max_fitness_list, max_fit_n_feat_list, max_fit_mean_pred_list, avg_pop_fitness_list, avg_pop_n_feat_list, avg_pop_mean_pred_list, max_fit_pop_list, train_idx

# %%
################################################################################################
#                                  Parallelization Main function
################################################################################################


def main():
    # initialize
    i = 0

    # initializing empty score lists
    ga_instance_list = []
    max_fitness_lists = []
    max_fit_n_feat_lists = []
    max_fit_mean_pred_lists = []
    avg_pop_fitness_lists = []
    avg_pop_n_feat_lists = []
    avg_pop_mean_pred_lists = []
    max_fit_pop_lists = []

    train_idx_list = []

    start = time.perf_counter()

    with ProcessPool(max_workers=24) as executor:
        print(filename)
        results = executor.map(ga_wrapper_stage, kf_train_idxcs, input_set)
        for result in results:
            ga_instance, max_fitness_list, max_fit_n_feat_list, max_fit_mean_pred_list, avg_pop_fitness_list, avg_pop_n_feat_list, avg_pop_mean_pred_list, max_fit_pop_list, train_idx = result  # extract output

            i += 1
            print("This is fold: ", i, "of", (num_splits*num_repeats))
            # Stage 2 output and selected features

            ga_instance_list.append(ga_instance)

            max_fit_pop_lists.append(max_fit_pop_list)
            max_fitness_lists.append(max_fitness_list)
            max_fit_n_feat_lists.append(max_fit_n_feat_list)
            max_fit_mean_pred_lists.append(max_fit_mean_pred_list)

            avg_pop_fitness_lists.append(avg_pop_fitness_list)
            avg_pop_n_feat_lists.append(avg_pop_n_feat_list)
            avg_pop_mean_pred_lists.append(avg_pop_mean_pred_list)

            train_idx_list.append(train_idx)

        finish = time.perf_counter()

        print(f'Finished in {round(finish-start, 2)} second(s)')

        # Pickle dump feature subset score and index lists
        with open(filename + '_ga_wrapper_stage_prelim_exploitive_rws500m005_vote_10', 'wb') as f:
            dill.dump([
                ga_instance_list,
                max_fitness_lists,
                max_fit_n_feat_lists,
                max_fit_mean_pred_lists,
                avg_pop_fitness_lists,
                avg_pop_n_feat_lists,
                avg_pop_mean_pred_lists,
                max_fit_pop_lists,
                train_idx_list], f)

        print("Done!Done!")


if __name__ == '__main__':
    main()

