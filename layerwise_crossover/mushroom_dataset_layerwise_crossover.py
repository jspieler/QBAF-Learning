import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch.autograd import Variable
import numpy as np
import operator
import matplotlib.pyplot as plt
import csv

from genetic_algorithm import GeneticAlgorithm as ga
from GBAG import GBAG as GBAG
from utils import remove_connections, NeuralNetwork, visualization

import warnings

warnings.filterwarnings('ignore')
device = torch.device("cpu")


def _accuracy(y_pred, y_true):
    classes = torch.argmax(y_pred, dim=1)
    labels = y_true
    accuracy = torch.mean((classes == labels).float())
    return accuracy


# parameter definitions of experiments
# parameters = {1: {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'mutation_rate': 0.001,
#                   'learning_rate': 3e-2, 'number_epochs': 2000, 'hidden_size': 12, 'number_connections1': 12,
#                   'number_connections2': 6, 'lambda': 0.1, 'patience': 5, 'threshold': 1e-4, 'elitist_pct': 0.1},
#               2: {'number_runs': 10, 'population_size': 50, 'number_generations': 20, 'mutation_rate': 0.001,
#                   'learning_rate': 1e-2, 'number_epochs': 3000, 'hidden_size': 12, 'number_connections1': 12,
#                   'number_connections2': 6, 'lambda': 0.2, 'patience': 5, 'threshold': 1e-4, 'elitist_pct': 0.1},
#               3: {'number_runs': 10, 'population_size': 50, 'number_generations': 20, 'mutation_rate': 0.001,
#                   'learning_rate': 1e-1, 'number_epochs': 3000, 'hidden_size': 12, 'number_connections1': 8,
#                   'number_connections2': 6, 'lambda': 0.4, 'patience': 15, 'threshold': 1e-6, 'elitist_pct': 0.1},
#               4: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'mutation_rate': 0.001,
#                   'learning_rate': 1e-2, 'number_epochs': 3000, 'hidden_size': 12, 'number_connections1': 8,
#                   'number_connections2': 6, 'lambda': 0.4, 'patience': 25, 'threshold': 1e-6, 'elitist_pct': 0.1},
#               5: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'mutation_rate': 0.001,
#                   'learning_rate': 1e-2, 'number_epochs': 3000, 'hidden_size': 12, 'number_connections1': 8,
#                   'number_connections2': 4, 'lambda': 0.6, 'patience': 20, 'threshold': 1e-6, 'elitist_pct': 0.1},
#               6: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'mutation_rate': 0.001,
#                   'learning_rate': 1e-2, 'number_epochs': 3000, 'hidden_size': 12, 'number_connections1': 8,
#                   'number_connections2': 4, 'lambda': 0.7, 'patience': 20, 'threshold': 1e-6, 'elitist_pct': 0.1}
#               }
parameters = {1: {'number_runs': 10, 'population_size': 100, 'number_generations': 20, 'mutation_rate': 0.001,
                  'learning_rate': 1e-1,
                  'number_epochs': 3000, 'hidden_size': 12, 'number_connections1': 8, 'number_connections2': 4,
                  'lambda': 0.6,
                  'patience': 20, 'threshold': 1e-6, 'elitist_pct': 0.1}
              }

data = pd.read_csv("../datasets/mushrooms.csv", header=0, na_values='?')
# drop column with missing values
data.drop('stalk_root', axis=1, inplace=True)
data.head()
X_, y = data.iloc[:, 1:23], data.iloc[:, 0]
le = LabelEncoder()
y = le.fit_transform(y)
X = pd.get_dummies(X_)

# create csv to store results
with open('mushroom_results_layerwise_crossover.csv', 'w') as file:
    writer = csv.writer(file)
    header = ["Parameters", "Number of connections", "Training accuracy", "Test accuracy", "Recall", "Precision",
              "F1 score"]
    writer.writerow(header)  # write the header

for params in parameters.values():
    # set fixed seeds for reproducibility
    torch.manual_seed(2021)
    np.random.seed(2021)  # scikit-learn also uses numpy random seed
    for run in range(params['number_runs']):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, stratify=y_train)
        X_tr = Variable(torch.tensor(X_train.values, dtype=torch.float))
        X_te = Variable(torch.tensor(X_test.values, dtype=torch.float))
        y_tr = Variable(torch.tensor(y_train, dtype=torch.long))
        y_te = Variable(torch.tensor(y_test, dtype=torch.long))
        X_val = Variable(torch.tensor(X_val.values, dtype=torch.float))
        y_val = Variable(torch.tensor(y_val, dtype=torch.long))

        gbag = GBAG
        criterion = torch.nn.CrossEntropyLoss()
        str_ln = ga(population_size=params['population_size'], mutation_rate=params['mutation_rate'],
                    generations=params['number_generations'])
        ga.create_population(str_ln, gbag, input_size=X.shape[1], hidden_size=params['hidden_size'], output_size=2,
                             num_connections1=params['number_connections1'],
                             num_connections2=params['number_connections2'])
        ga.fitness(str_ln, str_ln.population, X_tr, y_tr, X_val, y_val, epochs=params['number_epochs'],
                   learning_rate=params['learning_rate'], loss_fn=criterion,
                   minibatch=False, metrics=_accuracy, _lambda=params['lambda'], patience=20, tol=1e-6)
        # store accuracy for plots
        best_fitness = []
        mean_fitness = []
        patience = 5  # number of generations
        count = 0
        threshold = 1e-4
        best_score = None
        for g in range(str_ln.generations):
            acc = [indiv.accuracy for indiv in str_ln.population]
            fitness = [indiv.fitness for indiv in str_ln.population]
            idx = np.argmax(fitness)
            best_fitness.append(str_ln.population[idx].fitness)
            mean_fitness.append(np.mean(fitness))
            # stop GA if fitness increases less than threshold for a defined number of generations
            if best_score is None:
                best_score = best_fitness[g]
            elif best_fitness[g] - best_score < threshold:
                count += 1
            else:
                best_score = best_fitness[g]
                count = 0
            if count >= patience:
                break
            # elitist: pass best 10 % to next generation
            pop_sorted = sorted(str_ln.population, key=operator.attrgetter('accuracy', 'sparsity'), reverse=True)
            fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
            fitness_sorted.reverse()  # sort in descending order
            elitist = []  # fitness proportionate selection
            for index in sorted(fitness_sorted[:int(params['elitist_pct'] * str_ln.population_size)]):
                elitist.append(str_ln.population[index])
            parents = ga.tournament_selection(str_ln, int(0.5 * str_ln.population_size))
            crossover_offspring = ga.crossover_layer(str_ln, str_ln.population,
                                                     int((1 - params['elitist_pct']) * str_ln.population_size))
            pc1 = torch.empty(size=(len(crossover_offspring), X.shape[1] * params['hidden_size']))
            pc2 = torch.empty(size=(len(crossover_offspring), params['hidden_size'] * 2))
            for i, p in enumerate(crossover_offspring):
                pc1[i] = ga.encode(str_ln, p.sparse_linear1.connectivity, in_dim=X.shape[1],
                                   out_dim=params['hidden_size'])
                pc2[i] = ga.encode(str_ln, p.sparse_linear2.connectivity, in_dim=params['hidden_size'], out_dim=2)
            mutation_offspring = ga.flip_mutation(str_ln, pc1.numpy())
            mutation_offspring2 = ga.flip_mutation(str_ln, pc2.numpy())
            # new generation
            str_ln.population = elitist
            # reshape connectivity matrices and create new G-BAGs (offspring) given the new connectivity matrix
            for i, o in enumerate(range(int((1 - params['elitist_pct']) * str_ln.population_size))):
                sl1_conn = ga.decode(str_ln, mutation_offspring[i], shape=(X.shape[1], params['hidden_size']))
                sl2_conn = ga.decode(str_ln, mutation_offspring2[i], shape=(params['hidden_size'], 2))
                child = GBAG(input_size=X.shape[1], hidden_size=params['hidden_size'], output_size=2,
                             connections1=sl1_conn, connections2=sl2_conn)
                str_ln.population.append(child)
            ga.fitness(str_ln, str_ln.population[int(params['elitist_pct'] * str_ln.population_size):], X_tr, y_tr,
                       X_val, y_val, epochs=params['number_epochs'],
                       learning_rate=params['learning_rate'], loss_fn=criterion, minibatch=False, metrics=_accuracy,
                       _lambda=params['lambda'], patience=20, tol=1e-6)
            print("Generation {} finished".format(g + 1))
        acc = [indiv.accuracy for indiv in str_ln.population]
        fitness = [indiv.fitness for indiv in str_ln.population]
        idx = np.argmax(fitness)
        best_fitness.append(str_ln.population[idx].fitness)
        mean_fitness.append(np.mean(fitness))
        training_accuracy = str_ln.population[idx].accuracy
        print("Best individual - accuracy on training data: {:.4}".format(training_accuracy))
        print("Mean accuracy on training data: {:.4}".format(np.mean(acc)))
        print("Total sparsity of best G-BAG: {:.4}".format(str_ln.population[idx].sparsity))

        # evaluate best model on test data
        with torch.no_grad():
            str_ln.population[idx].eval()
            y_pred = str_ln.population[idx](X_te)
            test_loss = torch.nn.CrossEntropyLoss()(y_pred, y_te)
            test_acc = _accuracy(y_pred, y_te)
        print("Best individual - Loss on test data: {:.4}".format(test_loss))
        print("Best individual - Accuracy on test data: {:.4}".format(test_acc))

        classes = torch.argmax(y_pred, dim=1)
        labels = y_te
        precision = precision_score(labels, classes, average="macro")
        recall = recall_score(labels, classes, average="macro")
        f1 = f1_score(labels, classes, average="macro")
        print("Best individual - Precision test data: {:.4}".format(precision))
        print("Best individual - Recall test data: {:.4}".format(recall))
        print("Best individual - F1 score test data: {:.4}".format(f1))

        # visualizations (default: not shown, set 'show_plots' to 'True' otherwise)
        y_pred_non_category = [torch.argmax(t) for t in y_pred]
        class_label = ['edible', 'poisonous']
        visualization(best_fitness, mean_fitness, str_ln.population[idx].training_loss, str_ln.population[idx].val_loss,
                      y_te, y_pred_non_category, class_label)

        clf = str_ln.population[idx]
        rem, i_ind = remove_connections(clf)

        input_label = X.columns.values
        inp_used = []
        for inn in i_ind:
            inp_used.append(input_label[inn.item()])

        ind1 = rem.sparse_linear1.weight._indices()
        val1 = rem.sparse_linear1.weight._values()
        s1 = rem.sparse_linear1.out_features
        s2 = rem.sparse_linear1.in_features
        weights1 = torch.sparse.FloatTensor(ind1, val1, torch.Size([s1, s2])).to_dense()

        ind2 = rem.sparse_linear2.weight._indices()
        val2 = rem.sparse_linear2.weight._values()
        s3 = rem.sparse_linear2.out_features
        s4 = rem.sparse_linear2.in_features
        weights2 = torch.sparse.FloatTensor(ind2, val2, torch.Size([s3, s4])).to_dense()

        number_of_neurons_in_widest_layer = max(rem.sparse_linear1.in_features, rem.sparse_linear2.out_features)
        network = NeuralNetwork(number_of_neurons_in_widest_layer, horizontal_distance_between_layers=12)
        network.add_layer(s2, weights1)
        network.add_layer(s4, weights2)
        network.add_layer(s3)
        network.draw(inp_used, class_label)  # G-BAG is not shown by default (set 'show_plot' to 'True')

        num_connections = rem.sparse_linear1.connectivity.shape[1] + rem.sparse_linear2.connectivity.shape[1]
        print("Best individual - number of connections: {}".format(num_connections))
        with open('mushroom_results_layerwise_crossover.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(
                [params, num_connections, round(training_accuracy, 4), round(test_acc.item(), 4), round(recall, 4),
                 round(precision, 4), round(f1, 4)])