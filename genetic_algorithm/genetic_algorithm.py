import numpy as np
import torch
import csv
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score

from GBAG import GBAG as GBAG
from genetic_algorithm.operators.crossover import SinglePointCrossover, TwoPointCrossover
from genetic_algorithm.operators.mutation import FlipMutation, SwapMutationBetweenChromosomes
from genetic_algorithm.operators.selection import RankSelection, RouletteWheelSelection, TournamentSelection
from genetic_algorithm.utils.gbag import create_random_connectivity_matrix
from genetic_algorithm.utils.graph_visualizations import remove_connections, NeuralNetwork
from genetic_algorithm.utils.metrics import accuracy, sparsity
from genetic_algorithm.utils.plots import plot_fitness, plot_loss, plot_conf_matrix


selection_operators = {"roulette_wheel_selection": RouletteWheelSelection, "rank_selection": RankSelection, "tournament_selection": TournamentSelection}
crossover_operators = {"one_point_crossover": SinglePointCrossover, "two_point_crossover": TwoPointCrossover}
mutation_operators = {"flip_mutation": FlipMutation, "swap_mutation": SwapMutationBetweenChromosomes}


class GeneticAlgorithm:
    """Implementation of a genetic algorithm to evolve the structure of (sparse) multilayer perceptrons / GBAGs

    Parameters
    ----------
    input_size : number of input features

    output_size : number of classes

    selection method: method for selection operator
                    {'roulette_wheel_selection', 'tournament_selection', 'rank_selection'}

    crossover_method : method for crossover operator
                    {'one_point_crossover', 'two_point_crossover'}

    mutation_method : method for mutation operator
                    {'flip_mutation', 'swap_mutation'}

    params : dict containing necessary parameters, e.g.

            {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'crossover_rate': 0.9,
             'mutation_rate': 0.001, 'learning_rate': 3e-2, 'number_epochs': 3000, 'hidden_size': 12,
             'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 5,
             'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4}

             ES: Early Stopping
             GA: Genetic Algorithm

    loss_function : loss function for parameter and structure learning

    show_graph : show argumentation classifier graph
                {'True', 'False'}, default='False'

    show_plots : show plots for mean and best fitness, training and validation loss and confusion matrix
                {'True', 'False'}, default='False'

    """
    def __init__(self, input_size, output_size, selection_method, crossover_method, mutation_method, params,
                 loss_function, show_graph=False, show_plots=False):
        self.input_size = input_size
        self.output_size = output_size
        try:
            self.selection_operator = selection_operators[selection_method](num_parents=int(0.5 * params['population_size']))
            self.crossover_operator = crossover_operators[crossover_method](params['crossover_rate'], num_offspring=int(
                    (1 - params['elitist_pct']) * params['population_size']))
            self.mutation_operator = mutation_operators[mutation_method](params['mutation_rate'])
        except KeyError as e:
            raise NotImplementedError(f"Got unknown method {e} for one of the operators.")
        self.params = params
        self.loss_function = loss_function
        self.show_graph = show_graph
        self.show_plots = show_plots
        self.population = None

    def encode(self, connectivity_matrix, in_dim, out_dim):
        """
        Encode structure of graph as a bit string for genetic algorithm (concatenation of rows of connectivity matrix)
        """
        nnz = connectivity_matrix.shape[1]
        adj = torch.sparse.FloatTensor(connectivity_matrix, torch.ones(nnz), torch.Size([out_dim, in_dim])).to_dense()
        adj = adj.type(torch.DoubleTensor)
        adj = torch.where(adj <= 1, adj, 1.)  # remove redundant connections
        return torch.flatten(adj)

    def decode(self, chromosome, shape):
        """
        Convert genotype back to phenotype
        Transform bit string/chromosome representation back to tensor representation
        First, chromosome has to reshaped (unflattened)
        before the dense adjacency matrix has to be converted to sparse adjacency matrix
        input: shape (m,n)
        """
        chrom = torch.reshape(torch.from_numpy(chromosome), shape)
        assert chrom.dim() == 2
        ind = chrom.nonzero(as_tuple=False).t().contiguous()
        # if no connections, create random one
        if ind.size()[1] == 0:
            ind = np.zeros(shape=(2, 1))
            ind[0] = torch.randint(low=0, high=shape[0] - 1, size=(1,))
            ind[1] = torch.randint(low=0, high=shape[1] - 1, size=(1,))
            ind = torch.tensor(ind, dtype=torch.long)
        return torch.stack((ind[1], ind[0]), dim=0)

    def create_population(self, individual):
        self.population = [
            individual(input_size=self.input_size, hidden_size=self.params['hidden_size'], output_size=self.output_size,
                       connections1=create_random_connectivity_matrix(self.input_size, self.params['hidden_size'],
                                                                      self.params['number_connections1']),
                       connections2=create_random_connectivity_matrix(self.params['hidden_size'], self.output_size,
                                                                      self.params['number_connections2']))
            for _ in range(self.params['population_size'])]

    def fitness(self, population, x_train, y_train, x_val, y_val, metrics, opt_func=torch.optim.Adam, minibatch=False):
        # get fitness/loss of each individual
        for individual in population:
            individual.fitness = torch.autograd.Variable(torch.tensor(np.inf, dtype=torch.float))
            optimizer = opt_func(individual.parameters(), lr=self.params['learning_rate'])
            if minibatch:
                train_ds = TensorDataset(x_train, y_train)
                train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
            train_loss = []
            validation_loss = []
            best_score = None
            count = 0
            for epoch in range(self.params['number_epochs']):
                if minibatch:
                    batch_loss = 0.0
                    batch_accuracy = 0.0
                    for nb, (x_batch, y_batch) in enumerate(train_dl):
                        optimizer.zero_grad()
                        y_pred_train = individual(x_batch)
                        loss = self.loss_function(y_pred_train, y_batch)
                        loss.backward()
                        optimizer.step()
                        batch_loss += loss.item()
                        batch_accuracy += metrics(y_pred_train, y_batch).item()
                    train_loss.append(batch_loss / (nb + 1))
                    accuracy = batch_accuracy / (nb + 1)
                else:
                    optimizer.zero_grad()
                    y_pred_train = individual(x_train)
                    # computing the loss function
                    loss = self.loss_function(y_pred_train, y_train)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    accuracy = metrics(y_pred_train, y_train).item()
                individual.fitness = (1 - self.params['lambda']) * accuracy + self.params['lambda'] * sparsity(
                    individual)
                individual.accuracy = accuracy
                individual.sparsity = sparsity(individual)
                individual.training_loss = train_loss
                # Early Stopping
                with torch.no_grad():
                    individual.eval()
                    y_pred_val = individual(x_val)
                    val_loss = self.loss_function(y_pred_val, y_val)
                    validation_loss.append(val_loss.item())
                score = -val_loss.item()
                if best_score is None:
                    best_score = score
                elif score < best_score + self.params['tolerance_ES']:
                    count += 1
                else:
                    best_score = score
                    count = 0
                if count >= self.params['patience_ES']:
                    break
            individual.val_loss = validation_loss

    def roulette_wheel_selection(self, num_parents):

        """
        Selects the parents using the roulette wheel selection technique.
        Later, these parents will mate to produce the offspring.
        It accepts one parameter:
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """
        fitness_sum = sum([individual.fitness for individual in self.population])
        probs = [individual.fitness / fitness_sum for individual in self.population]
        probs_start = np.zeros(np.shape(probs),
                               dtype=np.float)  # array with start values of the ranges of probabilities
        probs_end = np.zeros(np.shape(probs),
                             dtype=np.float)  # array with end values of the ranges of probabilities
        curr = 0.0

        # form roulette wheel
        for _ in range(np.shape(probs)[0]):
            min_probs_idx = np.argmin(probs)
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Select best individuals in current generation as parents for producing offspring of next generation
        parents = []
        for parent_num in range(num_parents):
            rand_prob = np.random.rand()
            for idx in range(np.shape(probs)[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents.append(self.population[idx])
                    break
        return parents

    def rank_selection(self, num_parents):
        """
        Selects 'num_parents' many parent individuals using rank selection technique.
        """
        fitness = [individual.fitness for individual in self.population]
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()  # sort in descending order
        parents = []
        for parent_num in range(num_parents):
            parents.append(self.population[fitness_sorted[parent_num]])
        return parents

    def tournament_selection(self, num_parents, q=3):
        """
        Selects parents using tournament selection technique.
        q: number of tournaments (individuals that are compared in every iteration), default: 3
        """
        fitness = [individual.fitness for individual in self.population]
        fitness = np.array(fitness)
        parents = []
        used_ind = []
        for parent_num in range(num_parents):
            rand_ind = np.random.choice(range(len(fitness)), q, False)
            # check if individual was already used and create new random indices if
            for element in rand_ind:
                if element in used_ind:
                    rand_ind = np.random.choice(range(len(fitness)), q, False)
            parent_idx = np.where(fitness[rand_ind] == np.max(fitness[rand_ind]))[0][0]
            # store used indices to avoid that an individual is used more than once
            used_ind.append(rand_ind[parent_idx])
            parents.append(self.population[rand_ind[parent_idx]])
        return parents

    def run(self, X_tr, y_tr, X_val, y_val, X_te, y_te, input_labels, class_labels, file_name):
        """
        Run genetic algorithm for given configuration

        inputs:
                - training (tr), validation (val) and test (te) data
                - input labels (array containing labels of input features)
                - class labels (array containing class labels)
                - file name for csv to save results
        """
        gbag = GBAG
        self.create_population(gbag)
        self.fitness(self.population, X_tr, y_tr, X_val, y_val, metrics=accuracy)

        best_fitness = []
        mean_fitness = []
        count = 0
        best_score = None
        for g in range(self.params['number_generations']):
            fitness = [indiv.fitness for indiv in self.population]
            idx = np.argmax(fitness)
            best_fitness.append(self.population[idx].fitness)
            mean_fitness.append(np.mean(fitness))

            # stop genetic algorithm if accuracy does not increase for a certain number of generations
            if best_score is None:
                best_score = best_fitness[g]
            elif best_fitness[g] - best_score < self.params['tolerance_GA']:
                count += 1
            else:
                best_score = best_fitness[g]
                count = 0
            if count >= self.params['patience_GA']:
                break

            # elitism: pass certain percentage of best individuals directly to next generation
            fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
            fitness_sorted.reverse()  # sort in descending order
            elitist = []  # fitness proportionate selection
            for index in sorted(fitness_sorted[:int(self.params['elitist_pct'] * self.params['population_size'])]):
                elitist.append(self.population[index])

            # selection
            parents = self.selection_operator.select(self.population)

            # encoding of chromosomes
            pc1 = torch.empty(size=(len(parents), self.input_size * self.params['hidden_size']))
            pc2 = torch.empty(size=(len(parents), self.params['hidden_size'] * self.output_size))
            for i, p in enumerate(parents):
                pc1[i] = self.encode(p.sparse_linear1.connectivity, in_dim=self.input_size,
                                     out_dim=self.params['hidden_size'])
                pc2[i] = self.encode(p.sparse_linear2.connectivity, in_dim=self.params['hidden_size'],
                                     out_dim=self.output_size)

            # crossover
            crossover_offspring = self.crossover_operator.crossover(pc1)
            crossover_offspring2 = self.crossover_operator.crossover(pc2)

            # mutation
            mutation_offspring = self.mutation_operator.mutate(crossover_offspring)
            mutation_offspring2 = self.mutation_operator.mutate(crossover_offspring2)

            # form new generation
            self.population = elitist
            # reshape connectivity matrices and create new G-BAGs (offspring) given the new connectivity matrix
            for i, o in enumerate(range(int((1 - self.params['elitist_pct']) * self.params['population_size']))):
                sl1_conn = self.decode(mutation_offspring[i],
                                       shape=(self.input_size, self.params['hidden_size']))
                sl2_conn = self.decode(mutation_offspring2[i],
                                       shape=(self.params['hidden_size'], self.output_size))
                child = GBAG(input_size=self.input_size, hidden_size=self.params['hidden_size'],
                             output_size=self.output_size, connections1=sl1_conn, connections2=sl2_conn)
                self.population.append(child)

            # evaluate fitness of new population
            self.fitness(self.population[int(self.params['elitist_pct'] * self.params['population_size']):], X_tr, y_tr,
                         X_val, y_val, metrics=accuracy)

            print("Generation {} finished".format(g + 1))

        # select best individual and return results
        acc = [indiv.accuracy for indiv in self.population]
        fitness = [indiv.fitness for indiv in self.population]
        idx = np.argmax(fitness)
        clf = self.population[idx]
        best_fitness.append(clf.fitness)
        mean_fitness.append(np.mean(fitness))
        training_accuracy = clf.accuracy
        print("Best individual - accuracy on training data: {:.4}".format(training_accuracy))
        print("Mean accuracy on training data: {:.4}".format(np.mean(acc)))

        # evaluate best model on test data
        with torch.no_grad():
            clf.eval()
            y_pred = clf(X_te)
            test_loss = self.loss_function(y_pred, y_te)
            test_accuracy = accuracy(y_pred, y_te)
            print("Best individual - loss on test data: {:.4}".format(test_loss))
            print("Best individual - accuracy on test data: {:.4}".format(test_accuracy))

            classes = torch.argmax(y_pred, dim=1)
            if len(y_te.shape) > 1:
                labels = torch.argmax(y_te, dim=1)
            else:
                labels = y_te

            precision = precision_score(labels, classes, average="macro")
            recall = recall_score(labels, classes, average="macro")
            f1 = f1_score(labels, classes, average="macro")

        # remove not meaningful connections
        rem, i_ind = remove_connections(clf)
        num_connections = rem.sparse_linear1.connectivity.shape[1] + rem.sparse_linear2.connectivity.shape[1]
        print("Best individual - number of connections: {}".format(num_connections))

        # visualizations (default: not shown, set 'show_plots' to 'True' otherwise)
        if self.show_plots:
            plot_fitness(best_fitness, mean_fitness)
            plot_loss(clf.training_loss, clf.val_loss)
            plot_conf_matrix(labels, classes, class_labels)

        # draw argumentation classifier (default: not shown, set 'show_graph' to 'True' otherwise)
        if self.show_graph:
            inp_used = []
            for inn in i_ind:
                inp_used.append(input_labels[inn.item()])

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
            network = NeuralNetwork(number_of_neurons_in_widest_layer)
            network.add_layer(s2, weights1)
            network.add_layer(s4, weights2)
            network.add_layer(s3)
            network.draw(inp_used, class_labels)

        # write results to csv file
        with open(file_name, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([self.params, num_connections, round(training_accuracy, 4), round(test_accuracy.item(), 4),
                             round(recall, 4), round(precision, 4), round(f1, 4)])
