import numpy as np

from .crossover import Crossover


class TwoPointCrossover(Crossover):
    def crossover(self, parents):
        """Ordered crossover method as proposed by Goldberg.

        Two points are randomly selected for crossover between pairs of parents.
        Genes at the beginning and the end of the chromosome are from the first parent,
        genes between the 2 points are copied from second parent.
        """
        offspring = np.empty((self.num_offspring, parents.shape[1]))
        c = 0
        while c < self.num_offspring:
            par1_idx, par2_idx = np.random.choice(range(parents.shape[0]), 2, False)
            crossover_pt1 = np.random.randint(low=0, high=np.ceil(parents.shape[1] / 2 + 1), size=1)[0]
            crossover_pt2 = crossover_pt1 + int(parents.shape[1] / 2)
            if self.crossover_rate is not None:
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
                    if c <= self.num_offspring - 2:  # only if number of offspring is not reached
                        offspring[c + 1, 0:crossover_pt1] = parents[par2_idx, 0:crossover_pt1]
                        offspring[c + 1, crossover_pt2:] = parents[par2_idx, crossover_pt2:]
                        offspring[c + 1, crossover_pt1:crossover_pt2] = parents[par1_idx, crossover_pt1:crossover_pt2]
                        c += 2
                    else:
                        c += 1
        return offspring
