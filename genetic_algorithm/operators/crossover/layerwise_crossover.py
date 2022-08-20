from copy import deepcopy

import numpy as np

from .crossover import Crossover


class LayerwiseCrossover(Crossover):
    def crossover(self, parents):
        """Layerwise crossover operator.

        Crossover by exchanging layers between parents: a layer is chosen randomly and their connection indices
        are exchanged between two parents.

        Note:
            In this case inputs are the parents (GBAGs) not their encoding.
            Currently, only implemented for one hidden layer.

        TODO: re-implement it
        """
        offspring = []
        c = 0
        while c < self.num_offspring:
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
                    if c <= self.num_offspring - 2:
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
                    if c <= self.num_offspring - 2:
                        offspring.append(deepcopy(parents[par2_idx]))
                        ref = getattr(offspring[c + 1], prefix)
                        setattr(ref, suffix, parents[par1_idx].sparse_linear2.connectivity)
                        c += 2
                    else:
                        c += 1
            else:  # no crossover applied
                offspring.append(parents[par1_idx])
                if c <= self.num_offspring - 2:
                    offspring.append(parents[par2_idx])
                    c += 2
                else:
                    c += 1
        return offspring
