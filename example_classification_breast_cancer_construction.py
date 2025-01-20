# -*- coding: utf-8 -*- 
import grape
import algorithms
import pandas as pd
import numpy as np
from deap import creator, base, tools
import random
import csv
from functions import add, sub, mul, pdiv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
import sys
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# Additional imports to handle saving of predicted probabilities
import logging
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

# Suppressing Warnings:
import warnings
warnings.filterwarnings("ignore")
problem = 'breast_cancer_construction'
scenario = 0
run = 1
N_RUNS = 1

def setDataSet(problem, RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    if problem == 'breast_cancer_construction':
        if scenario == 0:
            data_train = pd.read_csv(r"./dataset_v2_three_files/ML_dataset/STEM_DATASET/cc_mlo_segments/stem_cc_mlo_train_50.csv", sep=",")
            data_val = pd.read_csv(r"./dataset_v2_three_files/ML_dataset/STEM_DATASET/cc_mlo_segments/cc_mlo_segments_holdout.csv", sep=",")
            data_test = pd.read_csv(r"./dataset_v2_three_files/ML_dataset/STEM_DATASET/cc_mlo_segments/cc_mlo_segments_val_30.csv", sep=",")     

        if scenario >= 36 and scenario <= 44:
            GRAMMAR_FILE = 'breast_cancer_construction_30featuresGeneral.bnf'
        else:
            GRAMMAR_FILE = 'breast_cancer_construction_52featuresGeneral.bnf'
            
        l = data_train.shape[0]
        Y_train = np.zeros([l,], dtype=int)
        for i in range(l):
            Y_train[i] = data_train['diagnosis'].iloc[i]
        data_train.pop('diagnosis')
        
        l = data_val.shape[0]
        Y_val = np.zeros([l,], dtype=int)
        for i in range(l):
            Y_val[i] = data_val['diagnosis'].iloc[i]
        data_val.pop('diagnosis')
        
        l = data_test.shape[0]
        Y_test = np.zeros([l,], dtype=int)
        for i in range(l):
            Y_test[i] = data_test['diagnosis'].iloc[i]
        data_test.pop('diagnosis')
        
        X_train = data_train.to_numpy().transpose()
        X_val = data_val.to_numpy().transpose()
        X_test = data_test.to_numpy().transpose()
        
    BNF_GRAMMAR = grape.Grammar(r"grammars/" + GRAMMAR_FILE)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, BNF_GRAMMAR

def fitness_eval1(individual, points_train, points_val=None, random_seed=None):
    global x
    global c
    
    if points_val:
        x_train = points_train[0]
        Y_train = points_train[1]
        x_val = points_val[0]
        Y_val = points_val[1]
        
        x = np.hstack((x_train, x_val))
        Y = np.hstack((Y_train, Y_val))
        
        _, c = np.shape(x)
        _, c_train = np.shape(x_train)
    else:
        _, c = np.shape(x)
        x = points_train[0]
        Y = points_train[1]
        
    global classifier_name
    global buildFeatures
    
    if individual.invalid == True:
        return np.NaN,
    else:
        try:
            exec(individual.phenotype, globals()) # buildFeatures will be filled
            
            if classifier_name == 'lda':
                classifier = LinearDiscriminantAnalysis()
            elif classifier_name == 'xgboost':
                classifier = XGBClassifier(n_jobs=1, verbosity=0, random_state=random_seed)
            elif classifier_name == 'lightgbm':
                classifier = LGBMClassifier(n_jobs=1, verbose=-1, random_state=random_seed)
            elif classifier_name == 'et':
                classifier = ExtraTreesClassifier(n_jobs=1, random_state=random_seed)
            elif classifier_name == 'rf':
                classifier = RandomForestClassifier(n_jobs=1, random_state=random_seed)
            elif classifier_name == 'lr':
                classifier = LogisticRegression(n_jobs=1, max_iter=200, random_state=random_seed)
            elif classifier_name == 'adaboost':
                classifier = AdaBoostClassifier(random_state=random_seed)
            elif classifier_name == 'dt':
                classifier = DecisionTreeClassifier(random_state=random_seed)
            elif classifier_name == 'nb':
                classifier = GaussianNB()
            elif classifier_name == 'knn':
                classifier = KNeighborsClassifier()
            elif classifier_name == 'svm':
                classifier = SVC(probability=True, random_state=random_seed)
            
            # Train Linear Discriminant Analysis model
            #lda_model = LinearDiscriminantAnalysis()
            classifier.fit(buildFeatures[:c_train,:], Y[:c_train])

            # Make predictions on the training with crossvalidation and validation set            
            scores_cv = cross_val_score(classifier, buildFeatures[:c_train,:], Y[:c_train], cv=5, scoring='roc_auc')
            
            val_predic_proba = classifier.predict_proba(buildFeatures[c_train:,:])
            
            # Calculate the crossvalidated training and the validation AUC
            auc_cv_train = np.mean(scores_cv)
            auc_val = roc_auc_score(Y[c_train:], val_predic_proba[:,1])
            
            fitness = 1 - (auc_cv_train + auc_val) / 2
            
        except (FloatingPointError, ZeroDivisionError, OverflowError,
                MemoryError, ValueError, TypeError):
            return np.NaN,
        
        if fitness == 0:
            print(individual.phenotype)
        return fitness,
    
def eval_test(individual, points_train, points_test, random_seed):
    global x
    
    x = points_train[0]
    Y = points_train[1]
    
    global c
    
    _, c = np.shape(x)
    
    global classifier_name
    global buildFeatures
    
    if individual.invalid == True:
        return np.NaN,
    else:
        try:
            exec(individual.phenotype, globals()) # buildFeatures will be filled
            
            if classifier_name == 'lda':
                classifier = LinearDiscriminantAnalysis()
            elif classifier_name == 'xgboost':
                classifier = XGBClassifier(n_jobs=1, verbosity=0, random_state=random_seed)
            elif classifier_name == 'lightgbm':
                classifier = LGBMClassifier(n_jobs=1, verbose=-1, random_state=random_seed)
            elif classifier_name == 'et':
                classifier = ExtraTreesClassifier(n_jobs=1, random_state=random_seed)
            elif classifier_name == 'rf':
                classifier = RandomForestClassifier(n_jobs=1, random_state=random_seed)
            elif classifier_name == 'lr':
                classifier = LogisticRegression(n_jobs=1, max_iter=200, random_state=random_seed)
            elif classifier_name == 'adaboost':
                classifier = AdaBoostClassifier(random_state=random_seed)
            elif classifier_name == 'dt':
                classifier = DecisionTreeClassifier(random_state=random_seed)
            elif classifier_name == 'nb':
                classifier = GaussianNB()
            elif classifier_name == 'knn':
                classifier = KNeighborsClassifier()
            elif classifier_name == 'svm':
                classifier = SVC(probability=True, random_state=random_seed)
            
            # Train Linear Discriminant Analysis model
            #lda_model = LinearDiscriminantAnalysis()
            classifier.fit(buildFeatures, Y)

            
            # Make predictions on the training set
            train_predic_proba = classifier.predict_proba(buildFeatures)
            
            
            # Calculate and print the training AUC
            fitness = 1 - roc_auc_score(Y, train_predic_proba[:,1])
            
        except (FloatingPointError, ZeroDivisionError, OverflowError,
                MemoryError, ValueError, TypeError):
            return np.NaN,
        
        x = points_test[0]
        global buildFeaturesTest
        phenotype = individual.phenotype.replace("buildFeatures", "buildFeaturesTest")
        _, c = np.shape(x)
   #     buildFeatures = np.zeros([c, 3], dtype=float)
        exec(phenotype, globals()) # buildFeatures will be filled for test    
        
        # Make predictions on the training set
        test_predictions = classifier.predict(buildFeaturesTest)
        test_predict_proba = classifier.predict_proba(buildFeaturesTest)
        # Save the predicted probabilities and true labels into a CSV file
        save_predicted_probabilities(Y_test, test_predict_proba[:, 1], i + 1)
        
        # Calculate the test AUC    
        fpr, tpr, thresholds = roc_curve(Y_test, test_predict_proba[:,1]) # calculate roc curves
        gmeans = np.sqrt(tpr * (1-fpr)) # calculate the g-mean for each threshold
        ix = np.argmax(gmeans) # locate the index of the largest g-mean
        best_threshold=thresholds[ix]
        
        AUC = roc_auc_score(Y_test, test_predict_proba[:,1])
        
        acc = accuracy_score(Y_test, test_predictions)
        
        f1 = f1_score(Y_test, test_predictions)
        
        precision = precision_score(Y_test, test_predictions)
        
        recall = recall_score(Y_test, test_predictions)
        
        return AUC, acc, f1, precision, recall, best_threshold, fpr[ix], tpr[ix]
    
def save_predicted_probabilities(true_labels, predicted_probabilities, run_number):
    output_df = pd.DataFrame({
        'Run': run_number,
        'True_Label': true_labels,
        'Predicted_Probability': predicted_probabilities
    })
    
    # Define the path to save the CSV file
    output_file_path = r"./predicted_probabilities_all_runs.csv"
    
    # Check if file exists
    if os.path.exists(output_file_path):
        # Append the new data without writing the header
        output_df.to_csv(output_file_path, mode='a', header=False, index=False)
    else:
        # Create the file with headers if it doesn't exist
        output_df.to_csv(output_file_path, index=False)
    
    print(f"Predicted probabilities for run {run_number} saved to {output_file_path}")

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual) 

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=6)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 100#200
MAX_INIT_TREE_DEPTH = 5
MIN_INIT_TREE_DEPTH = 4

MAX_GENERATIONS = 50
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = 1
HALLOFFAME_SIZE = 1

CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'
MAX_GENOME_LENGTH = None#'auto'

MAX_TREE_DEPTH = 15 #equivalent to 17 in GP with this grammar
MAX_WRAPS = 0
CODON_SIZE = 255

REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max', 
                'fitness_test',
                'test_AUC',
                'test_acc',
                'test_f1',
                'test_precision',
                'test_recall',
                'best_threshold', 
                'fpr', 
                'tpr',
                'best_ind_length', 'avg_length', 
                'best_ind_nodes', 'avg_nodes', 
                'best_ind_depth', 'avg_depth', 
                'avg_used_codons', 'best_ind_used_codons', 
    #            'behavioural_diversity',
    #            'fitness_diversity',
                'structural_diversity', 
    #            'evaluated_inds',
                'selection_time', 'generation_time',
                'frequency',
                'best_phenotype']

def count_substrings(input_string, n):
    counts = [0] * n

    for i in range(n):
        substring = f'x[{i}]'
        start_index = 0
        while True:
            index = input_string.find(substring, start_index)
            if index == -1:
                break
            counts[i] += 1
            start_index = index + 1

    return counts

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i + run)
    print()
    
    RANDOM_SEED = i + run
    
    toolbox.register("evaluate", fitness_eval1, random_seed=RANDOM_SEED)
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test, BNF_GRAMMAR = setDataSet(problem, RANDOM_SEED) #We set up this inside the loop for the case in which the data is defined randomly

    random.seed(RANDOM_SEED) 

    # create initial population (generation 0):
    population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                       bnf_grammar=BNF_GRAMMAR, 
                                       min_init_depth=MIN_INIT_TREE_DEPTH,
                                       max_init_depth=MAX_INIT_TREE_DEPTH,
                                       codon_size=CODON_SIZE,
                                       codon_consumption=CODON_CONSUMPTION,
                                       genome_representation=GENOME_REPRESENTATION
                                        )
    
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALLOFFAME_SIZE)
    
    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    
    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitismAndValidation(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR, 
                                              codon_size=CODON_SIZE, 
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              max_genome_length=MAX_GENOME_LENGTH,
                                              points_train=[X_train, Y_train], 
                                              points_val=[X_val, Y_val],
                                              points_test=[X_test, Y_test], 
                                              codon_consumption=CODON_CONSUMPTION,
                                              report_items=REPORT_ITEMS,
                                              genome_representation=GENOME_REPRESENTATION,      
                                              invalidate_max_depth=False,
                                              problem=problem,
                                              stats=stats, halloffame=hof, verbose=False)
    
    import textwrap
    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')
    
    n_features = len(X_train)
    frequency = []
    for j in range(MAX_GENERATIONS):
        frequency.append(np.NaN)
    frequency_final = count_substrings(hof.items[0].phenotype, n_features)
    print(frequency_final)
    frequency.append(frequency_final)
    
    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    fitness_test = logbook.select("fitness_test")
    
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")

    selection_time = logbook.select("selection_time")
    generation_time = logbook.select("generation_time")
    gen, invalid = logbook.select("gen", "invalid")
    avg_used_codons = logbook.select("avg_used_codons")
    best_ind_used_codons = logbook.select("best_ind_used_codons")
    
    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    behavioural_diversity = logbook.select("behavioural_diversity") 
    structural_diversity = logbook.select("structural_diversity") 
    fitness_diversity = logbook.select("fitness_diversity")     
    evaluated_inds = logbook.select("evaluated_inds") 
    
    best_phenotype = [float('nan')] * MAX_GENERATIONS
    best_phenotype.append(best)
    
    AUC, acc, f1, precision, recall, best_threshold, fpr, tpr = eval_test(hof.items[0], [X_train, Y_train], [X_test, Y_test], random_seed=RANDOM_SEED)
    
    test_AUC = [float('nan')] * MAX_GENERATIONS
    test_AUC.append(AUC)
    
    test_acc = [float('nan')] * MAX_GENERATIONS
    test_acc.append(acc)
    
    test_f1 = [float('nan')] * MAX_GENERATIONS
    test_f1.append(f1)
    
    test_precision = [float('nan')] * MAX_GENERATIONS
    test_precision.append(precision)
    
    test_recall = [float('nan')] * MAX_GENERATIONS
    test_recall.append(recall)
    
    threshold = [float('nan')] * MAX_GENERATIONS
    threshold.append(best_threshold)
    
    test_fpr = [float('nan')] * MAX_GENERATIONS
    test_fpr.append(fpr)
    
    test_tpr = [float('nan')] * MAX_GENERATIONS
    test_tpr.append(tpr)
        
    r = RANDOM_SEED
    
    header = REPORT_ITEMS
    
    address = r"./result_F/" + problem + "0" + str(scenario) + "/"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(address)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(address)
        
    with open(address + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value], 
                             fitness_test[value],
                             test_AUC[value],
                             test_acc[value],
                             test_f1[value],
                             test_precision[value],
                             test_recall[value],
                             threshold[value],
                             test_fpr[value],
                             test_tpr[value],
                             best_ind_length[value], 
                             avg_length[value], 
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value], 
#                             behavioural_diversity[value],
#                             fitness_diversity[value],
                             structural_diversity[value],
#                             evaluated_inds[value],
                             selection_time[value], 
                             generation_time[value],
                             frequency[value],
                             best_phenotype[value]])