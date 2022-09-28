from abc import ABC, abstractmethod


class Selection(ABC):
    """Abstract base class for the selection operator."""
    def __init__(self, num_parents):
        self.num_parents = num_parents

    @abstractmethod
    def select(self, population):
        """Applies the selection operator.

        Selects 'num_parents' individuals from the population.
        """
        pass
