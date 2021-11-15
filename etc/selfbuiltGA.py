'''
The following code implements a genetic algorithm (GA) for the second phase of the feature selection process.

The GA has the goal of optimizing the learning algorithms performance by selecting the most 'informative'
features, while reducing the feature subset cardinality.

The GA, consist of various parts, namely,
-
-
-
-
-

This specific code is setup for the preprocessig of the real gc6-74 matched datasets.
'''
# Imports
import numpy as np
import pandas as pd
import pickle
import matplotlib
from matplotlib import pyplot
from random import choices
from scipy.stats import rankdata
# Data Prep
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
# Feature Selection Methods
# Ranker
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.similarity_based import reliefF
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.statistical_based import gini_index
# Subset
from skfeature.function.statistical_based import CFS
from fcbf_func import fcbf
from skfeature.utility.mutual_information import su_calculation
# Machine Learning Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Model Evaluation
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
# %%
# General functions
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
# Import
################################################################################################
'''##############################################Choose#########################################'''
filename = 'ge_raw_6'
'''#############################################################################################'''
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
#   Load filter method outputs
################################################################################################
filter_pickle_directory = 'C:/Users/Daniel/Documents/Thesis/Python Code/xfilter ouputx/'
# Pickle load feature subset lists RANKERS
with open(filter_pickle_directory+filename+'_mrm_rank_105_625', 'rb') as f:
    mrm_rank_105_625 = pickle.load(
        f)
with open(filter_pickle_directory+filename+'_mrm_log_rank_105_625', 'rb') as f:
    mrm_log_rank_105_625 = pickle.load(
        f)
# %%
################################################################################################
#   Create Ensemble
################################################################################################
# ----------------- Create Feature Set -----------------
# A ranked set of features from the filter algorithms with necessary preprocessing step to be used for the ensemble
# Standardization applied to: mRMR, Fisher-Score, Info Gain, Gini-Index
idx_fisher_list, n, n, idx_mim_list, idx_gini_list, idx_mrmr_list = rank_rank(
    mrm_rank_105_625)
# Standardization + Normalization applied to: ReliefF, Chi-Square
n, idx_chi_list, idx_reliefF_list, n, n, n = rank_rank(
    mrm_log_rank_105_625)
# Create filter set
filter_set_105_625 = idx_fisher_list, idx_chi_list, idx_reliefF_list, idx_mim_list, idx_gini_list, idx_mrmr_list
# %%
# ----------------- Create Ensemble -----------------
'''###########Choose###########'''
ensemble_thresholds = [5,10,25,50,125]
'''###########Choose###########'''

for threshold in ensemble_thresholds:
    print("Now making: " + str(threshold))
    ensemble_threshold = threshold  # caps the number of features of each algorithm to put into ensemble

    # Apply thresholding for ensemble
    idx_fisher_list_th_e, idx_chi_list_th_e, idx_reliefF_list_th_e, idx_mim_list_th_e, idx_gini_list_th_e, idx_mrmr_list_th_e = rank_thres(
        filter_set_105_625, ensemble_threshold)

    # Initialize feature list
    idx_ensemble_list = []
    # append features from different methods together
    for i in range(0, (5*10)):
        ensembled_features = np.append(idx_fisher_list_th_e[i], [idx_chi_list_th_e[i], idx_reliefF_list_th_e[i], idx_mim_list_th_e[i], idx_gini_list_th_e[i]])
        ensembled_features
        # remove features which are duplicated in the list
        ensembled_features = np.array(list(dict.fromkeys(ensembled_features)))
        # make list of every folds selected features
        idx_ensemble_list.append(ensembled_features)

    if ensemble_threshold == 5:
        idx_ensemble_list_5 = idx_ensemble_list
    elif ensemble_threshold == 10:
        idx_ensemble_list_10 = idx_ensemble_list
    elif ensemble_threshold == 25:
        idx_ensemble_list_25 = idx_ensemble_list
    elif ensemble_threshold == 50:
        idx_ensemble_list_50 = idx_ensemble_list
    elif ensemble_threshold == 125:
        idx_ensemble_list_125 = idx_ensemble_list
# %%
################################################################################################
# Cross-validation splitting
################################################################################################
# ----------------- Variables -----------------
# Data input variables
X_train = X
y_train = y
sum(y_train == 1)
num_splits = 10  # number of folds
num_repeats = 5  # number of repeats

# Apply stratified cross-validation
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=0)

'''
Important to note the random_state of the train_test_split function as well as the random_state and splitting criteria of the RepeatedStratifiedKFold
function for future use.

These criteria are essentially the data splitting criteria.
'''
# initialize cv indices lists
kf_train_idxcs = []
kf_test_idxcs = []
# generate cv indices lists
for kf_train_index, kf_test_index in rskf.split(X_train, y_train):
    kf_train_idxcs.append(kf_train_index)
    kf_test_idxcs.append(kf_test_index)
y_train[kf_test_idxcs[1]]
len(y_train[kf_test_idxcs[1]])
# %%
################################################################################################
# Genetic Algorithm
################################################################################################
# Functions
################################################################################################

# Fitness function


def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features_ind = features[selected_elements_indices]
    return reduced_features_ind


def calc_pop_fitness(pop, features, labels, data, classifier, weight_factor, train_indices, test_indices):

    pop = population
    pop
    features = selected_feature_indices  # from phase 1
    features.shape
    labels = y_train
    data = X_train
    classifier = classifiers
    weight_factor = fitness_weight
    train_indices = fold_train_indices
    test_indices = fold_test_indices

    # Initialize
    # output arrays
    accuracies = np.zeros(pop.shape[0])
    sensitivity = np.zeros(pop.shape[0])
    specificity = np.zeros(pop.shape[0])
    gmean = np.zeros(pop.shape[0])
    weighted_fitness = np.zeros(pop.shape[0])
    # population index
    i = 0
    # data
    X_train = data[train_indices, :]
    y_train = labels[train_indices]
    # Standardize features
    # First check if the learning algorithm to be used requires standardization
    ''' This has to be appended for the other learning algorithms that perform
    better with standardization '''
    if any(req_standard in classifiers.keys() for req_standard in ('SVM_linear', 'SVM_rbf', 'KNN')):
        print("Standardizing")
        scaler = StandardScaler().fit(X_train_sel)
        #standardization = StandardScaler()
        #X_train = standardization.fit_transform(X_train)

    # application of SMOTE, BorderlineSMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    # evaluate each solution's/chromoome's fitness
    for curr_solution in pop:
        # extract chosen feature indices
        reduced_features_ind = reduce_features(
            curr_solution, features)

        # create a 'solution' training set (training set with only chosen features)
        X_train_sol = X_train[:, reduced_features_ind]

        # Determine number of selected features
        num_selected_features = np.sum(curr_solution)
        print(num_selected_features)
        # calculate solution/chromosome fitness
        for clf_key, clf in classifier.items():
            # apply standardization
            if clf_key in ('SVM_linear', 'SVM_rbf', 'KNN'):
                print("Standardizing")
                scaler = StandardScaler().fit(X_train_sel)

            # train classifier
            clf.fit(X_train_sol, y_train)
            # predict
            y_train_predict = clf.predict(X_train_sol)
            # accuracy
            accuracies[i] = accuracy_score(y_train, y_train_predict)
            # recall/sensitivity
            # with class 0 as representing a positive
            sensitivity[i] = recall_score(y_train, y_train_predict, pos_label=0)
            # specificity
            # with class 1 as representing a positive
            specificity[i] = recall_score(y_train, y_train_predict, pos_label=1)
            # geometric mean
            gmean[i] = np.sqrt(sensitivity[i]*specificity[i])
            # fitness assignment
            weighted_fitness[i] = (1-weight_factor)*gmean[i] + weight_factor * \
                (1 - num_selected_features/chromosome_size)
        # Increment population index
        i = i + 1
    return weighted_fitness, gmean, sensitivity, specificity

# Parent selection


def select_mating_pool_trunc(pop, fitnesses, num_parents):
    # Selecting the best feature sets in the current generation as parents for producing the offspring of the next generation
    parents = np.empty((num_parents, pop.shape[1]))
    # rank population indices according to its fitness
    # populations indices sorted in descending order according to fitness
    sorted_pop_indices = (-fitnesses).argsort()
    # rank populations according indices
    ranked_populations = pop[sorted_pop_indices]
    parents = ranked_populations[0:num_parents, :]
    return parents

    # for parent_num in range(num_parents):
    #     # identify max fitness populations
    #     max_fitness_idx = np.where(fitnesses == np.max(fitnesses))
    #     max_fitness_idx = max_fitness_idx[0][0]  # select only the first max fitness population
    #     # set the selected population as a parent
    #     parents[parent_num, :] = pop[max_fitness_idx, :]
    #     # set the selected population fitness to low number to ensure the next round it is not chosen
    #     fitnesses[max_fitness_idx] = -999999
    # return parents


def select_mating_pool_roulette(pop, fitnesses, num_parents):
    parents = choices(pop, weights=fitnesses, k=num_parents)  # roulette wheel selection
    return np.array(parents)


def select_mating_pool_rank(pop, fitnesses, num_parents):
    ranked_fitness = rankdata(fitnesses)  # populations ranked based on their fitness
    # roulette wheel selection with population rank as weights
    parents = choices(pop, weights=ranked_fitness, k=num_parents)
    return np.array(parents)


def select_mating_pool_anneal(pop, fitnesses, num_parents, current_gen):
    rank = rankdata(fitnesses)  # populations ranked based on their fitness
    avg_rank = rank/(sum(rank))
    avg_fitness = fitness/(sum(fitnesses))
    annealed_fitness = avg_rank*(1-1/(current_gen+1))+avg_fitness*(0+1/(current_gen+1))
    parents = choices(pop, weights=annealed_fitness, k=num_parents)
    return np.array(parents)

# Crossover


def crossover_single_point(parents, pop_shape):
    offspring = np.empty(pop_shape)
    for k in range(pop_shape[0]):
        # random single crossover point
        crossover_point = np.uint8(pop_shape[1]*np.random.uniform(0, 1))
        # Select parents initialy in order that they were selected by parent selection operator until all have mated
        if k < parents.shape[0]:
            # Index of the first parent to mate
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate
            parent2_idx = (k+1) % parents.shape[0]
        # Thereafter mate randomly selected parents from parent group
        else:
            parent1_idx = np.random.randint(parents.shape[0])
            parent2_idx = np.random.randint(parents.shape[0])
        # The new offspring will have its first half of its genes taken from the first parent
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def crossover_uniform(parents, pop_shape, prob):
    offspring = np.empty(pop_shape)
    for k in range(pop_shape[0]):
        # Select parents initialy in order that they were selected by parent selection operator until all have mated
        if k < parents.shape[0]:
            # Index of the first parent to mate
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate
            parent2_idx = (k+1) % parents.shape[0]
        # Thereafter mate randomly selected parents from parent group
        else:
            parent1_idx = np.random.randint(parents.shape[0])
            parent2_idx = np.random.randint(parents.shape[0])
        # parents
        parent1, parent2 = parents[parent1_idx], parents[parent2_idx]
        # Create child by first initializing with parent 1 and then adding randomly chosen features from parent 2
        child = parent1
        uniform_mask = np.random.rand(parents.shape[1]) > prob
        child[uniform_mask] = parent2[uniform_mask]
        offspring[k] = child
    return offspring

# Mutation


def mutation_1(offspring, perc_mutation, num_features):
    mutated_offspring = offspring
    for l in range(offspring.shape[0]):
        mutation_idx = np.random.randint(
            low=0, high=offspring.shape[1], size=np.uint8(perc_mutation*num_features))
        # The random value to be added to the gene.
        mutated_offspring[l, mutation_idx] = 1 - offspring[l, mutation_idx]
    return mutated_offspring


def mutation_2(offspring, perc_mutation, mutation_rate, num_features):
    mutated_offspring = offspring
    for l in range(offspring.shape[0]):
        mutation_idx = np.random.randint(
            low=0, high=offspring.shape[1], size=np.uint8(perc_mutation*num_features))
        if np.random.random() < mutation_rate:
            # The random value to be added to the gene.
            mutated_offspring[l, mutation_idx] = 1 - offspring[l, mutation_idx]
    return mutated_offspring

# Survivor selection


def survivor_selection_elitism(pop, offspring, fitnesses, percentage_elite):
    num_elite = np.uint8(percentage_elite*pop.shape[0])
    new_pop = np.empty(pop.shape)
    # rank population indices according to its fitness
    # populations indices sorted in descending order according to fitness
    sorted_pop_indices = (-fitnesses).argsort()
    # rank populations according indices
    ranked_populations = pop[sorted_pop_indices]
    elite_solutions = ranked_populations[0:num_elite, :]
    new_pop[0:num_elite, :] = elite_solutions
    new_pop[num_elite:, :] = offspring[0:(pop.shape[0]-num_elite)]
    return new_pop


################################################################################################
# %%
# Set parameters
################################################################################################
# Initialize classifiers to be used
classifiers = {
    # 'KNN': KNeighborsClassifier(n_jobs=-1),
    # 'SVM_linear': LinearSVC(loss='hinge'),
     'SVM_rbf': SVC(kernel="rbf"),
    # 'GaussianNB': GaussianNB(),
    # 'RF': RandomForestClassifier(n_jobs=-1),
    # 'XGBoost': XGBClassifier(n_jobs=-1)
}
# Select fold to evaluate
fold_num = 0

# idx_ensemble_list_5
# idx_ensemble_list_10
# idx_ensemble_list_25
# idx_ensemble_list_50
# idx_ensemble_list_125

fold_train_indices = kf_train_idxcs[fold_num]  # training indices
fold_test_indices = kf_test_idxcs[fold_num]  # test indices

selected_feature_indices = idx_ensemble_list_25[fold_num]  # selected feature indices
# %%
# Set GA parameters
pop_size = 30  # Population size (number of solutions/chromosomes to be tested)
n_parents_mating = np.uint8(pop_size/2)  # Number of parents inside the mating pool
perc_mutation = 0.1  # Number of elements to mutate in a chromosome
mutation_rate = 0.05
chromosome_size = len(selected_feature_indices)  # Number of features in a chromosome/solution
fitness_weight = 0.3  # weight assigned to number of selected features
perc_elite = 0.1  # for elitism survival selection, percentage of population to be selected as elite solutions
# Initial population parameters
mask_size = 0.6  # size of initialization mask
# Select number of generations
n_generations = 35
################################################################################################
# %%
# Initialize output lists
best_outputs_pop = []
best_outputs_n_pop = []
best_outputs_fitness = []
best_outputs_pa = []
best_outputs_num_features = []
best_outputs_sensitivity = []
best_outputs_specificity = []
# %%
################################################################################################
# RUN GENETIC ALGORITHM
################################################################################################
# 1. Create initial population
pop_shape = (pop_size, chromosome_size)  # initialize population array shape
population = np.empty(pop_shape)  # create empty population array
for i in range(pop_size):
    # create chromosome/solution with all features selected (=1)
    chromosome = np.ones(chromosome_size)
    mask = np.random.rand(len(chromosome)) < mask_size  # make mask to randomly 'remove' features
    chromosome[mask] = False  # remove the masked features
    population[i] = chromosome
# 2. INITIAL Measuring the fitness of each chromosome in the population
fitness, predictive_ability, sensitivity, specificity = calc_pop_fitness(
    population, selected_feature_indices, y_train, X_train, classifiers, fitness_weight, fold_train_indices, fold_test_indices)

for generation in range(n_generations):
    print("Generation: ", generation)
    # Save best outputs for each generation
    best_outputs_idx = np.where(fitness == np.max(fitness))[0]  # best output index
    # Number of best outputs
    best_outputs_n_pop.append(best_outputs_idx)
    # Population    ?
    best_outputs_pop.append(population[best_outputs_idx, :])  # ?
    # Fitness ?
    best_outputs_fitness.append(np.max(fitness))  # best output fitness
    # Predictive ability ?
    best_outputs_pa.append(predictive_ability[best_outputs_idx[0]])  # ? Hoekom net 0
    # Number of features in best output
    best_outputs_num_features.append(sum(population[best_outputs_idx[0], :] == 1))  # ? Hoekom net 0
    # sensitivity & specificity
    best_outputs_sensitivity.append(sensitivity[best_outputs_idx[0]])  # best output sensitivity
    best_outputs_specificity.append(specificity[best_outputs_idx[0]])  # best output specificity
    # Show best result in the current iteration
    print("Best output : \nFitness - ", best_outputs_fitness[-1], ", # Selected Features - ",
          best_outputs_num_features[-1], ", Predictive Ability - ", best_outputs_pa[-1])

    # 3. Selecting the best parents in the population for mating
    # 3.1 Truncation
    # mating_parents = select_mating_pool_trunc(population, fitness, n_parents_mating)
    # 3.2 Roulette Wheel selection
    # mating_parents = select_mating_pool_roulette(population, fitness, n_parents_mating)
    # 3.3 Rank selection
    mating_parents = select_mating_pool_rank(population, fitness, n_parents_mating)
    # 3.4 Anneal selection
    # mating_parents = select_mating_pool_anneal(population, fitness, n_parents_mating, generation)

    # 4. Generating next generation using crossover
    # 4.1 single point
    # offspring_crossover = crossover_single_point(
    #     mating_parents, pop_shape= pop_shape)
    # 4.2 uniform
    offspring_crossover = crossover_uniform(mating_parents, pop_shape=pop_shape, prob=0.5)

    # 5. Adding some variations to the offspring using mutation
    # 5.1 mutation method without mutation rate (increased exploration)
    offspring_mutation = mutation_1(offspring_crossover, perc_mutation, chromosome_size)
    # 5.2 mutation method with mutation rate (dampens exploration rate)
    # offspring_mutation = mutation_2(offspring_crossover, perc_mutation, mutation_rate, chromosome_size)

    # 6. Creating the new population based on the parents and offspring (Survival selection)
    # 6.1 Elitism
    population = survivor_selection_elitism(population, offspring_mutation, fitness, perc_elite)

    # 2. Measuring the fitness of each chromosome in the population
    fitness, predictive_ability, sensitivity, specificity = calc_pop_fitness(
        population, selected_feature_indices, y_train, X_train, classifiers, fitness_weight, fold_train_indices, fold_test_indices)

# Getting the best solution after iterating finishing all generations
print("Generation: ", generation + 1)
# Save outputs & best outputs
best_outputs_idx = np.where(fitness == np.max(fitness))[0]  # best output index
# Number of best outputs
best_outputs_n_pop.append(best_outputs_idx)
# Population
best_outputs_pop.append(population[best_outputs_idx, :])
# Fitness
best_outputs_fitness.append(np.max(fitness))  # best output fitness
# Predictive ability
best_outputs_pa.append(predictive_ability[best_outputs_idx[0]])
# Number of features
best_outputs_num_features.append(sum(population[best_outputs_idx[0], :] == 1))
# sensitivity & specificity
best_outputs_sensitivity.append(sensitivity[best_outputs_idx[0]])  # best output sensitivity
best_outputs_specificity.append(specificity[best_outputs_idx[0]])  # best output specificity
# Show best result in the current iteration
print("Best output : \nFitness - ", best_outputs_fitness[-1], ", # Selected Features - ",
      best_outputs_num_features[-1], ", Predictive Ability - ", best_outputs_pa[-1])
# %%
# Then return the index of the generations corresponding to the best fitnesses
best_match_gen = np.where(best_outputs_fitness == np.max(best_outputs_fitness))[0]
best_match_gen
# %%
for match in best_match_gen:
    best_solution = best_outputs_pop[match][0]
    best_solution
    best_solution_feature_indices = np.where(best_solution == 1)[0]
    best_solution_num_elements = best_solution_feature_indices.shape[0]
    best_solution_fitness = best_outputs_fitness[match]
    best_solution_pa = best_outputs_pa[match]
    best_solution_sensitivity = best_outputs_sensitivity[match]
    best_solution_specificity = best_outputs_specificity[match]

    print("best_match_idx : ", best_match_gen)
    print("\nBest population : ", best_solution)
    print("\nSelected features : ", selected_feature_indices[best_solution_feature_indices])
    print("\nNumber of selected elements : ", best_solution_num_elements)
    print("\nBest solution fitness : ", best_solution_fitness)
    print("\nBest solution predictive ability : ", best_solution_pa, "\n")

matplotlib.pyplot.plot(best_outputs_fitness)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()

matplotlib.pyplot.plot(best_outputs_num_features)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("# Selected Features")
matplotlib.pyplot.show()

matplotlib.pyplot.plot(best_outputs_pa)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Predictive Ability")
matplotlib.pyplot.show()

matplotlib.pyplot.plot(best_outputs_sensitivity)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Sensitivity")
matplotlib.pyplot.show()

matplotlib.pyplot.plot(best_outputs_specificity)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Specificity")
matplotlib.pyplot.show()
# %%
################################################################################################
# Test
################################################################################################
# selected features
final_selected_features = selected_feature_indices[best_solution_feature_indices]

# training and testing data folds
X_train_fold = X_train[fold_train_indices, :]
X_train_fold.shape
X_test_fold = X_train[fold_test_indices, :]
X_test_fold.shape

# training and testing label folds
y_train_fold = y_train[fold_train_indices]
y_train_fold.shape
y_test_fold = y_train[fold_test_indices]
y_test_fold.shape

# Train and Test set with only selected feature_selection
X_train_fold_red = X_train_fold[:, final_selected_features]
X_train_fold_red.shape
X_test_fold_red = X_test_fold[:, final_selected_features]
X_test_fold_red.shape

# Standardize datasets
''' This has to be appended for the other learning algorithms that perform
better with standardization '''
if any(req_standard in classifiers.keys() for req_standard in ('SVM_linear', 'SVM_rbf')):
    standardization = StandardScaler()
    X_train_fold_red = standardization.fit_transform(X_train_fold_red)
    X_test_fold_red = standardization.fit_transform(X_test_fold_red)

sm = BorderlineSMOTE(random_state=42)
X_train_fold_redS, y_train_foldS = sm.fit_resample(X_train_fold_red, y_train_fold)
sum(y_train_foldS == 0)
# Initialize classifier
clf = classifiers['ComplementNB']

# fit
clf.fit(X_train_fold_redS, y_train_foldS)
# predict
predict = clf.predict(X_test_fold_red)
# confusion matrix
conf_mat = confusion_matrix(y_test_fold, predict)
print('Confusion matrix:\n', conf_mat)
# Number of controls in test
sum(y_test_fold == 1)
# Number of cases in test
sum(y_test_fold == 0)
# accuracy
accuracy_score(y_test_fold, predict)
# recall/sensitivity
recall_score(y_test_fold, predict, pos_label=0)
# specificity
recall_score(y_test_fold, predict, pos_label=1)
# GMean
np.sqrt(recall_score(y_test_fold, predict, pos_label=0)
        * recall_score(y_test_fold, predict, pos_label=1))
'''
To improve
- Address the class imbalance:
    - Preprocessing: SMOTE
    - Algorithmic: Adjust the way in which the fitness function determines importance

Note:
- It is interesting the way in which the GA acquires good senitivity and specificity, but on the test set
the sensitivity nearly always lags considerably.
- It is necessary to take a look at the probabilities
https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models/312787#312787
- (After application of SMOTE) It is necessary to note that the GA has to major concerns
    * Firstly, its stability is abismal, and I don't know if its because of a lack of data (training
      and testing) or due to the algorithm itself.
    * Secondly, it is maybe necessary to specifically bias the fitness function in order to Identify
      the minority class.
- Maybe use of CV is hurting model due to small data?

BIG NOTE:
Each aspect of the feature selection process must be tested under different conditions. Thus,
this entire fs method should not be tailored to only this dataset. An easier dataset, must
maybe first be tested in order to determine the methods suitability, and thereafter this more
difficult dataset can be tested. (BUT TIME)

TO DO:
- Need to automate the way in which results are shown
    * Automatically calculate for all folds and produce averages in a neat table (like phase 1)
    * Haal die indices goed uit die fitness function uit. Jy wil net die data vir hom gee en hy
      moet go!
    * Vergemaklik die toets prosedure! Easy options to test different implementations and inputs


Question:
- Should class imbalance be addressed prior to or after feature selection?
    * It is worried that applying it before could bias the fs method and its implication of applying it
      later has thus far shown no substainal negatives. In actual fact it improves the sensitivity
'''
