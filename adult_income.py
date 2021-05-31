import csv
import warnings

import numpy as np
import pandas as pd
import torch
from genetic_algorithm import GeneticAlgorithm as ga
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from utils import binning

warnings.filterwarnings('ignore')
device = torch.device("cpu")


# parameter definitions of experiments
parameters = {1: {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              2: {'number_runs': 10, 'population_size': 50, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.2, 'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              3: {'number_runs': 10, 'population_size': 50, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 6, 'lambda': 0.4, 'patience_ES': 5,
                  'tolerance_ES': 1e-6, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              4: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 6, 'lambda': 0.4, 'patience_ES': 5,
                  'tolerance_ES': 1e-6, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              5: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 10, 'number_connections2': 6, 'lambda': 0.4, 'patience_ES': 25,
                  'tolerance_ES': 1e-6, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              6: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 10, 'number_connections2': 6, 'lambda': 0.6, 'patience_ES': 25,
                  'tolerance_ES': 1e-6, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4}
              }

# load dataset
filename = './datasets/adult-all.csv'
dataframe = read_csv(filename, header=None, na_values='?',
                     names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                            "native-country", "Income"])
# drop rows with missing
dataframe = dataframe.dropna()
target = dataframe.values[:, -1]
# split into inputs and outputs
last_ix = len(dataframe.columns) - 1
X_, y = dataframe.drop("Income", axis=1), dataframe["Income"]
# select categorical and numerical features
cat_ix = X_.select_dtypes(include=['object', 'bool']).columns
num_ix = X_.select_dtypes(include=['int64', 'float64']).columns
# label encode the target variable to have the classes 0 and 1
y = LabelEncoder().fit_transform(y)
# one-hot encoding of categorical features
df_cat = pd.get_dummies(X_[cat_ix])
# binning of numerical features
x = X_.drop(columns=cat_ix, axis=1)
df_num, num_list = binning(x, n_bins=3, strategy='uniform', encode='onehot-dense',
                           feature_names=['Age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                          'hours-per-week'])
X = pd.concat([df_cat.reset_index(drop=True), pd.DataFrame(df_num).reset_index(drop=True)], axis=1)

cat_label = df_cat.columns.values
num_label = np.asarray(num_list)
inputs = np.concatenate((cat_label, num_label), axis=0)
label = ['Income $\leq$ 50K', 'Income $>$ 50K']

fname = 'adult_income_results.csv'

# create csv to store results
with open(fname, 'w') as file:
    writer = csv.writer(file)
    header = ["Parameters", "Number of connections", "Training accuracy", "Test accuracy", "Recall", "Precision",
              "F1 score"]
    writer.writerow(header)  # write the header

for params in parameters.values():
    # set fixed seeds for reproducibility
    torch.manual_seed(2021)
    np.random.seed(2021)  # scikit-learn also uses numpy random seed
    for run in range(params['number_runs']):
        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, stratify=y_train)
        X_tr = Variable(torch.tensor(X_train.values, dtype=torch.float))
        X_te = Variable(torch.tensor(X_test.values, dtype=torch.float))
        y_tr = Variable(torch.tensor(y_train, dtype=torch.long))
        y_te = Variable(torch.tensor(y_test, dtype=torch.long))
        X_val = Variable(torch.tensor(X_val.values, dtype=torch.float))
        y_val = Variable(torch.tensor(y_val, dtype=torch.long))

        criterion = torch.nn.CrossEntropyLoss()
        model = ga(input_size=X.shape[1], output_size=2, selection_method='tournament_selection',
                   crossover_method='two_point_crossover', mutation_method='flip_mutation', params=params,
                   loss_function=criterion)
        model.run(X_tr, y_tr, X_val, y_val, X_te, y_te, input_labels=inputs, class_labels=label, file_name=fname)
