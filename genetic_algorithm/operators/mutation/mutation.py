from abc import ABC, abstractmethod


class Mutation(ABC):
    """Abstract base class for the mutation operator."""
    def __init__(self, mutation_probability):
        self.mutation_probability = mutation_probability

    @abstractmethod
    def mutate(self, offspring):
        """Applies the mutation operator to the offspring."""
        pass
