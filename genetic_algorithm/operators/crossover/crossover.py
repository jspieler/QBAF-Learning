from abc import ABC, abstractmethod


class Crossover(ABC):
    """Abstract base class for the crossover operator."""
    def __init__(self, crossover_rate, num_offspring):
        self.crossover_rate = crossover_rate
        self.num_offspring = num_offspring

    @abstractmethod
    def crossover(self, parents):
        """Applies the crossover operator to the parents.

        Creates 'num_offspring' many offspring from the parents.
        """
        pass
