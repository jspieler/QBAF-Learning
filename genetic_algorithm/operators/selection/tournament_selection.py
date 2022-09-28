import numpy as np

from .selection import Selection


class TournamentSelection(Selection):
    def select(self, population, q=3):
        """Selects parents using tournament selection technique.

        q: number of tournaments (individuals that are compared in every iteration), default: 3
        """
        fitness = [individual.fitness for individual in population]
        fitness = np.array(fitness)
        parents = []
        used_ind = []
        for parent_num in range(self.num_parents):
            rand_ind = np.random.choice(range(len(fitness)), q, False)
            # check if individual was already used and create new random indices if
            for element in rand_ind:
                if element in used_ind:
                    rand_ind = np.random.choice(range(len(fitness)), q, False)
            parent_idx = np.where(fitness[rand_ind] == np.max(fitness[rand_ind]))[0][0]
            # store used indices to avoid that an individual is used more than once
            used_ind.append(rand_ind[parent_idx])
            parents.append(population[rand_ind[parent_idx]])
        return parents
