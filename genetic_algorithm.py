from copy import deepcopy

import numpy as np
import sparselinear
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return torch.nn.CrossEntropyLoss()(input, labels)


def binary_cross_entropy_one_hot(input, target):
    _, labels = input.max(dim=1)
    # float/integer values required -> cast
    labels = labels.type(torch.LongTensor)
    target = target.type(torch.LongTensor)
    return torch.nn.BCELoss()(labels, target)


def sparsity(model):
    """ Get sparsity of model """
    num_conn = 0
    max_num_conn = 0
    for layer in model.children():
        if isinstance(layer, sparselinear.SparseLinear):
            num_conn += layer.connectivity.shape[1]
            max_num_conn += layer.in_features * layer.out_features
    reg_term = (max_num_conn - num_conn) / max_num_conn
    return reg_term


def create_random_connectivity_matrix(in_size, out_size, num_connections):
    col = torch.randint(low=0, high=in_size, size=(num_connections,)).view(1, -1).long()
    row = torch.randint(low=0, high=out_size, size=(num_connections,)).view(1, -1).long()
    connections = torch.cat((row, col), dim=0)
    return connections


class GeneticAlgorithm:
    """
    Implementation of a genetic algorithm to evolve the structure of (sparse) neural networks/GBAGs
    """

    def __init__(self, population_size, mutation_rate=0.001, crossover_rate=0.9, generations=20):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = None

    def encode(self, connectivity_matrix, in_dim, out_dim):
        """
        Encode structure of graph/network as a bit string for genetic algorithm (concatenation of rows of connectivity matrix)
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

    def create_population(self, individual, input_size, hidden_size, output_size, num_connections1, num_connections2):
        self.population = [individual(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                                      connections1=create_random_connectivity_matrix(input_size, hidden_size,
                                                                                      num_connections1),
                                      connections2=create_random_connectivity_matrix(hidden_size, output_size,
                                                                                      num_connections2)) for
                           _ in range(self.population_size)]

    def fitness(self, population, x_train, y_train, x_val, y_val, epochs, learning_rate, metrics, _lambda=0.1,
                opt_func=torch.optim.Adam, loss_fn=cross_entropy_one_hot, minibatch=False, patience=5, tol=1e-4):
        # get fitness/loss of each individual
        for individual in population:
            individual.fitness = torch.autograd.Variable(torch.tensor(np.inf, dtype=torch.float))
            optimizer = opt_func(individual.parameters(), lr=learning_rate)
            if minibatch:
                train_ds = TensorDataset(x_train, y_train)
                train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
            train_loss = []
            validation_loss = []
            best_score = None
            count = 0
            for epoch in range(epochs):
                if minibatch:
                    batch_loss = 0.0
                    batch_accuracy = 0.0
                    for nb, (x_batch, y_batch) in enumerate(train_dl):
                        optimizer.zero_grad()
                        y_pred_train = individual(x_batch)
                        loss = loss_fn(y_pred_train, y_batch)
                        loss.backward()
                        optimizer.step()
                        batch_loss += loss.item()
                        batch_accuracy += metrics(y_pred_train, y_batch).item()
                    train_loss.append(batch_loss / (nb+1))
                    accuracy = batch_accuracy / (nb+1)
                else:
                    optimizer.zero_grad()
                    y_pred_train = individual(x_train)
                    # computing the loss function
                    loss = loss_fn(y_pred_train, y_train)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    accuracy = metrics(y_pred_train, y_train).item()
                individual.fitness = (1 - _lambda) * accuracy + _lambda * sparsity(individual)
                individual.accuracy = accuracy
                individual.sparsity = sparsity(individual)
                individual.training_loss = train_loss
                # Early Stopping
                with torch.no_grad():
                    individual.eval()
                    y_pred_val = individual(x_val)
                    val_loss = loss_fn(y_pred_val, y_val)
                    validation_loss.append(val_loss.item())
                score = -val_loss.item()
                if best_score is None:
                    best_score = score
                elif score < best_score + tol:
                    count += 1
                else:
                    best_score = score
                    count = 0
                if count >= patience:
                    break
            individual.val_loss = validation_loss

    def roulette_wheel_selection(self, num_parents):

        """
        Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
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
        q: number of tournaments (individuals that are compared in every iteration)
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

    def single_point_crossover(self, parents, num_offspring):
        """
        Single-point crossover randomly selects a point for crossover between pairs of parents
        """
        offspring = np.empty((num_offspring, parents.shape[1]))
        c = 0
        while c < num_offspring:
            par1_idx, par2_idx = np.random.choice(range(parents.shape[0]), 2, False)
            crossover_pt = np.random.randint(low=0, high=parents.shape[1], size=1)[0]
            if self.crossover_rate != None:
                probs = np.random.random(size=2)
                ind = np.where(probs <= self.crossover_rate)[0]
                if ind.shape[0] == 0:  # no crossover, parents are selected
                    offspring[c, :] = parents[c % parents.shape[0], :]
                    c += 2
                    continue
                elif ind.shape[0] == 1:
                    parent1_idx = par1_idx
                    parent2_idx = parent1_idx
                    offspring[c, 0:crossover_pt] = parents[parent1_idx, 0:crossover_pt]
                    offspring[c, crossover_pt:] = parents[parent2_idx, crossover_pt:]
                    c += 1
                    continue
                else:
                    offspring[c, 0:crossover_pt] = parents[par1_idx, 0:crossover_pt]
                    offspring[c, crossover_pt:] = parents[par2_idx, crossover_pt:]
                    if c <= num_offspring - 2:  # only if number of offspring is not reached
                        offspring[c + 1, 0:crossover_pt] = parents[par2_idx, 0:crossover_pt]
                        offspring[c + 1, crossover_pt:] = parents[par1_idx, crossover_pt:]
                        c += 2
                    else:
                        c += 1
        return offspring

    def two_point_crossover(self, parents, num_offspring):
        """
        Two points are randomly selected for crossover between pairs of parents
        Ordered crossover method as proposed by Goldberg

        Genes at beginning and end of chromosome are from first parent,
        genes between the 2 points are copied from second parent
        """
        offspring = np.empty((num_offspring, parents.shape[1]))
        c = 0
        while c < num_offspring:
            par1_idx, par2_idx = np.random.choice(range(parents.shape[0]), 2, False)
            crossover_pt1 = np.random.randint(low=0, high=np.ceil(parents.shape[1] / 2 + 1), size=1)[0]
            crossover_pt2 = crossover_pt1 + int(parents.shape[1] / 2)
            if self.crossover_rate != None:
                probs = np.random.random(size=2)
                ind = np.where(probs <= self.crossover_rate)[0]
                if ind.shape[0] == 0:
                    offspring[c, :] = parents[c % parents.shape[0], :]
                    c += 2
                    continue
                elif ind.shape[0] == 1:
                    parent1_idx = par1_idx
                    parent2_idx = parent1_idx
                    offspring[c, 0:crossover_pt1] = parents[parent1_idx, 0:crossover_pt1]
                    offspring[c, crossover_pt2:] = parents[parent1_idx, crossover_pt2:]
                    offspring[c, crossover_pt1:crossover_pt2] = parents[parent2_idx, crossover_pt1:crossover_pt2]
                    c += 1
                    continue
                else:
                    offspring[c, 0:crossover_pt1] = parents[par1_idx, 0:crossover_pt1]
                    offspring[c, crossover_pt2:] = parents[par1_idx, crossover_pt2:]
                    offspring[c, crossover_pt1:crossover_pt2] = parents[par2_idx, crossover_pt1:crossover_pt2]
                    if c <= num_offspring - 2:  # only if number of offspring is not reached
                        offspring[c + 1, 0:crossover_pt1] = parents[par2_idx, 0:crossover_pt1]
                        offspring[c + 1, crossover_pt2:] = parents[par2_idx, crossover_pt2:]
                        offspring[c + 1, crossover_pt1:crossover_pt2] = parents[par1_idx, crossover_pt1:crossover_pt2]
                        c += 2
                    else:
                        c += 1
        return offspring

    def crossover_layer(self, parents, num_offspring):
        """
        Crossover by exchanging layers between parents: a layer is chosen randomly and their connection indices
        are exchanged between two parents.
        Note: in this case inputs are the parents (GBAGs) not their encoding
        Currently, only implemented for one hidden layer
        """
        offspring = []
        c = 0
        while c < num_offspring:
            prob = np.random.random(size=1)
            par1_idx, par2_idx = np.random.choice(range(len(parents)), 2, False)
            if prob <= self.crossover_rate:
                layer_idx = np.random.randint(low=0, high=1, size=1)
                offspring.append(deepcopy(parents[par1_idx]))
                if layer_idx == 0:
                    value = 'sparse_linear1.connectivity'
                    prefix, suffix = value.rsplit(".", 1)
                    ref = getattr(offspring[c], prefix)
                    setattr(ref, suffix, parents[par2_idx].sparse_linear1.connectivity)
                    if c <= num_offspring - 2:
                        offspring.append(deepcopy(parents[par2_idx]))
                        ref = getattr(offspring[c + 1], prefix)
                        setattr(ref, suffix, parents[par1_idx].sparse_linear1.connectivity)
                        c += 2
                    else:
                        c += 1
                else:
                    value = 'sparse_linear2.connectivity'
                    prefix, suffix = value.rsplit(".", 1)
                    ref = getattr(offspring[c], prefix)
                    setattr(ref, suffix, parents[par2_idx].sparse_linear2.connectivity)
                    if c <= num_offspring - 2:
                        offspring.append(deepcopy(parents[par2_idx]))
                        ref = getattr(offspring[c + 1], prefix)
                        setattr(ref, suffix, parents[par1_idx].sparse_linear2.connectivity)
                        c += 2
                    else:
                        c += 1
            else:  # no crossover applied
                offspring.append(parents[par1_idx])
                if c <= num_offspring - 2:
                    offspring.append(parents[par2_idx])
                    c += 2
                else:
                    c += 1
        return offspring

    def flip_mutation(self, offspring):
        """
        Applies flip mutation on a binary encoded chromosome,
        each gene whose probability is <= mutation rate is mutated randomly
        """
        mutation_offspring = np.array(offspring)
        for offspring_idx in range(offspring.shape[0]):
            probs = np.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= self.mutation_rate:
                    mutation_offspring[offspring_idx, gene_idx] = type(offspring[offspring_idx, gene_idx])(
                        not offspring[offspring_idx, gene_idx])
        return mutation_offspring

    def swap_mutation(self, offspring):
        """
        Applies swap mutation which interchanges 2 randomly selected genes within a chromosome (offspring)
        """
        mutation_offspring = np.array(offspring)
        for offspring_idx in range(offspring.shape[0]):
            prob = np.random.random(size=1)
            if prob <= self.mutation_rate:  # else offspring is returned without mutation
                # get indices of zero and non-zero elements
                ind_nz = np.where(offspring[offspring_idx] != 0)[0]
                ind_z = np.where(offspring[offspring_idx] == 0)[0]
                # randomly choose one index of each
                idx_gene1 = np.random.choice(ind_nz, 1)
                idx_gene2 = np.random.choice(ind_z, 1)
                mutation_offspring[offspring_idx, idx_gene1] = offspring[offspring_idx, idx_gene2]
                mutation_offspring[offspring_idx, idx_gene2] = offspring[offspring_idx, idx_gene1]
        return mutation_offspring

    def swap_mutation2(self, offspring):
        """
        Applies swap mutation that interchanges 2 randomly selected genes between random pairs of chromosomes
        """
        for offspring_idx in range(int(offspring.shape[0] / 2)):
            idx_offspring1, idx_offspring2 = np.random.choice(range(offspring.shape[0]), 2, False)
            idx_gene1, idx_gene2 = np.random.choice(range(offspring.shape[1]), 2, False)
            temp = offspring[idx_offspring1, idx_gene1]
            offspring[idx_offspring1, idx_gene1] = offspring[idx_offspring2, idx_gene2]
            offspring[idx_offspring2, idx_gene2] = temp
        return offspring
