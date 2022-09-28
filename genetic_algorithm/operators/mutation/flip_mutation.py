import numpy as np

from .mutation import Mutation


class FlipMutation(Mutation):
    def mutate(self, offspring):
        """Applies flip mutation on a binary encoded chromosome.

            Each gene whose probability is <= the mutation probability is mutated randomly.
        """
        mutated_offspring = np.array(offspring)
        for offspring_idx in range(offspring.shape[0]):
            probs = np.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= self.mutation_probability:
                    mutated_offspring[offspring_idx, gene_idx] = type(offspring[offspring_idx, gene_idx])(
                        not offspring[offspring_idx, gene_idx])
        return mutated_offspring
