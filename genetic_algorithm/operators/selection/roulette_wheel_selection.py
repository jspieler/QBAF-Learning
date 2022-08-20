import numpy as np

from .selection import Selection


class RouletteWheelSelection(Selection):
    def select(self, population):
        """Selects the parents using the roulette wheel selection technique.

        Later, these parents will mate to produce the offspring.

        Args:
            population: The population from which to select.

        Returns:
            An array of the selected parents.
        """
        fitness_sum = sum([individual.fitness for individual in population])
        probs = [individual.fitness / fitness_sum for individual in population]
        # arrays with the start and end values of the ranges of probabilities
        probs_start = np.zeros(np.shape(probs), dtype=np.float)
        probs_end = np.zeros(np.shape(probs), dtype=np.float)
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
        for parent_num in range(self.num_parents):
            rand_prob = np.random.rand()
            for idx in range(np.shape(probs)[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents.append(population[idx])
                    break
        return parents
