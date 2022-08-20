import numpy as np

from .crossover import Crossover


class SinglePointCrossover(Crossover):
    def crossover(self, parents):
        """Single-point crossover randomly selects a point for crossover between pairs of parents."""
        offspring = np.empty((self.num_offspring, parents.shape[1]))
        c = 0
        while c < self.num_offspring:
            par1_idx, par2_idx = np.random.choice(range(parents.shape[0]), 2, False)
            crossover_pt = np.random.randint(low=0, high=parents.shape[1], size=1)[0]
            if self.crossover_rate is not None:
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
                    if c <= self.num_offspring - 2:  # only if number of offspring is not reached
                        offspring[c + 1, 0:crossover_pt] = parents[par2_idx, 0:crossover_pt]
                        offspring[c + 1, crossover_pt:] = parents[par1_idx, crossover_pt:]
                        c += 2
                    else:
                        c += 1
        return offspring
