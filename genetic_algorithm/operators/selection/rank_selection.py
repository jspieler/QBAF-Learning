from .selection import Selection


class RankSelection(Selection):
    def select(self, population):
        """Selects 'num_parents' many parent individuals using rank selection technique."""
        fitness = [individual.fitness for individual in population]
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()  # sort in descending order
        parents = []
        for parent_num in range(self.num_parents):
            parents.append(population[fitness_sorted[parent_num]])
        return parents
