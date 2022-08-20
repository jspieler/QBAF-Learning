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


def flatten_connectivity_matrix(connectivity_matrix, in_dim, out_dim):
    """Flattens the connectivity matrix (concatenation of the rows)."""
    nnz = connectivity_matrix.shape[1]
    adj = torch.sparse.FloatTensor(
        connectivity_matrix, torch.ones(nnz), torch.Size([out_dim, in_dim])
    ).to_dense()
    adj = adj.type(torch.DoubleTensor)
    adj = torch.where(adj <= 1, adj, 1.0)  # remove redundant connections
    return torch.flatten(adj)