import torch
import torch.nn as nn

import sparselinear as sl


class GBAG(nn.Module):
    """
    Implementation of a Gradual Bipolar Argumentation Graph / edge-weighted QBAF as a sparse multi-layer perceptron
    using SparseLinear extension library for PyTorch (https://pypi.org/project/sparselinear/)
    """
    def __init__(self, input_size, hidden_size, output_size, connections1, connections2):
        super().__init__()

        self.sparse_linear1 = sl.SparseLinear(input_size, hidden_size, connectivity=connections1)
        self.activation1 = nn.Sigmoid()
        self.sparse_linear2 = sl.SparseLinear(hidden_size, output_size, connectivity=connections2)
        self.output_layer = nn.Softmax()

    def forward(self, x):
        x = self.sparse_linear1.forward(x)
        x = self.activation1(x)
        x = self.sparse_linear2.forward(x)
        output = x
        return output
