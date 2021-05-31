import warnings

import csv
import numpy as np
import pandas as pd
import torch
from genetic_algorithm import GeneticAlgorithm as ga
from genetic_algorithm import cross_entropy_one_hot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from utils import binning

warnings.filterwarnings('ignore')
device = torch.device("cpu")


# parameter definitions of experiments
parameters = {1: {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-4, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 10,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              2: {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 15,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              3: {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 3e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              4: {'number_runs': 10, 'population_size': 50, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 3e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              5: {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 3e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 6, 'lambda': 0.1,  'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              6: {'number_runs': 10, 'population_size': 50, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 3e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              7: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 3e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 6, 'lambda': 0.2, 'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4}
              }

# Loading iris data
iris_data = load_iris()
x = iris_data.data
y_ = iris_data.target.reshape(-1, 1)  # convert data to a single column

# one hot encode class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# binning of features
X, inputs = binning(pd.DataFrame(x), n_bins=3, encode='onehot-dense', strategy='uniform',
                    feature_names=['Petal length', 'Petal width', 'Sepal length', 'Sepal width'])
label = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

fname = 'iris_results.csv'

# create csv to store results and write header
with open(fname, 'w') as file:
    writer = csv.writer(file)
    header = ["Parameters", "Number of connections", "Training accuracy", "Test accuracy", "Recall", "Precision",
              "F1 score"]
    writer.writerow(header)

for params in parameters.values():
    # set fixed seeds for reproducibility
    torch.manual_seed(2021)
    np.random.seed(2021)  # scikit-learn also uses numpy random seed
    for run in range(params['number_runs']):
        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, stratify=y_train)
        X_tr = Variable(torch.tensor(X_train.to_numpy(), dtype=torch.float))
        X_te = Variable(torch.tensor(X_test.to_numpy(), dtype=torch.float))
        y_tr = Variable(torch.tensor(y_train, dtype=torch.float))
        y_te = Variable(torch.tensor(y_test, dtype=torch.float))
        X_val = Variable(torch.tensor(X_val.to_numpy(), dtype=torch.float))
        y_val = Variable(torch.tensor(y_val, dtype=torch.float))

        model = ga(input_size=X.shape[1], output_size=3, selection_method='tournament_selection',
                   crossover_method='two_point_crossover', mutation_method='flip_mutation', params=params,
                   loss_function=cross_entropy_one_hot)
        model.run(X_tr, y_tr, X_val, y_val, X_te, y_te, input_labels=inputs, class_labels=label, file_name=fname)
