"""Provides utils used for G-BAGs."""
import torch

device = torch.device("cpu")


def create_random_connectivity_matrix(in_size, out_size, num_connections):
    """Creates a random connectivity matrix."""
    col = torch.randint(low=0, high=in_size, size=(num_connections,)).view(1, -1).long()
    row = (
        torch.randint(low=0, high=out_size, size=(num_connections,)).view(1, -1).long()
    )
    connections = torch.cat((row, col), dim=0)
    return connections
