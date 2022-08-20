import numpy as np

from .mutation import Mutation


class SwapMutationWithinChromosome(Mutation):
    def mutate(self, offspring):
        """Applies swap mutation which interchanges 2 randomly selected genes within a chromosome (offspring)."""
        mutated_offspring = np.array(offspring)
        for offspring_idx in range(offspring.shape[0]):
            prob = np.random.random(size=1)
            if prob <= self.mutation_probability:  # else offspring is returned without mutation
                # get indices of zero and non-zero elements
                ind_nz = np.where(offspring[offspring_idx] != 0)[0]
                ind_z = np.where(offspring[offspring_idx] == 0)[0]
                # randomly choose one index of each
                idx_gene1 = np.random.choice(ind_nz, 1)
                idx_gene2 = np.random.choice(ind_z, 1)
                mutated_offspring[offspring_idx, idx_gene1] = offspring[offspring_idx, idx_gene2]
                mutated_offspring[offspring_idx, idx_gene2] = offspring[offspring_idx, idx_gene1]
        return mutated_offspring


class SwapMutationBetweenChromosomes(Mutation):
    def mutate(self, offspring):
        """Applies swap mutation that interchanges 2 randomly selected genes between random pairs of chromosomes."""
        for offspring_idx in range(int(offspring.shape[0] / 2)):
            idx_offspring1, idx_offspring2 = np.random.choice(range(offspring.shape[0]), 2, False)
            idx_gene1, idx_gene2 = np.random.choice(range(offspring.shape[1]), 2, False)
            temp = offspring[idx_offspring1, idx_gene1]
            offspring[idx_offspring1, idx_gene1] = offspring[idx_offspring2, idx_gene2]
            offspring[idx_offspring2, idx_gene2] = temp
        return offspring
