import warnings

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from genetic_algorithm import GeneticAlgorithm
from utils import create_csv_with_header

warnings.filterwarnings("ignore")
device = torch.device("cpu")


# parameter definitions of experiments
parameters = {1: {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 3e-2, 'number_epochs': 2000, 'hidden_size': 12,
                  'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              2: {'number_runs': 10, 'population_size': 50, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-2, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.2, 'patience_ES': 5,
                  'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              3: {'number_runs': 10, 'population_size': 50, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 6, 'lambda': 0.4, 'patience_ES': 15,
                  'tolerance_ES': 1e-6, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              4: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 6, 'lambda': 0.4, 'patience_ES': 25,
                  'tolerance_ES': 1e-6, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              5: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 4, 'lambda': 0.6, 'patience_ES': 20,
                  'tolerance_ES': 1e-6, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4},
              6: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'crossover_rate': 0.9,
                  'mutation_rate': 0.001, 'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12,
                  'number_connections1': 8, 'number_connections2': 4, 'lambda': 0.7, 'patience_ES': 20,
                  'tolerance_ES': 1e-6, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4}
              }

data = pd.read_csv("./datasets/mushrooms.csv", header=0, na_values="?")
# drop column with missing values
data.drop("stalk_root", axis=1, inplace=True)
data.head()
X_, y = data.iloc[:, 1:23], data.iloc[:, 0]
le = LabelEncoder()
y = le.fit_transform(y)
X = pd.get_dummies(X_)

inputs = X.columns.values
label = ["edible", "poisonous"]

fname = "mushroom_results.csv"
create_csv_with_header(fname)

for params in parameters.values():
    # set fixed seeds for reproducibility
    torch.manual_seed(2021)
    np.random.seed(2021)  # scikit-learn also uses numpy random seed
    for run in range(params["number_runs"]):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.125, stratify=y_train
        )
        X_tr = Variable(torch.tensor(X_train.values, dtype=torch.float))
        X_te = Variable(torch.tensor(X_test.values, dtype=torch.float))
        y_tr = Variable(torch.tensor(y_train, dtype=torch.long))
        y_te = Variable(torch.tensor(y_test, dtype=torch.long))
        X_val = Variable(torch.tensor(X_val.values, dtype=torch.float))
        y_val = Variable(torch.tensor(y_val, dtype=torch.long))

        criterion = torch.nn.CrossEntropyLoss()
        model = GeneticAlgorithm(
            input_size=X.shape[1],
            output_size=2,
            selection_method="tournament_selection",
            crossover_method="two_point_crossover",
            mutation_method="flip_mutation",
            params=params,
            loss_function=criterion,
        )
        model.run(
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_te,
            y_te,
            input_labels=inputs,
            class_labels=label,
            file_name=fname,
        )
