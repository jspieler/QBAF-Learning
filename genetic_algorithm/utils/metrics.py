"""Provides metrics."""
import torch
import sparselinear


def accuracy(y_pred, y_true):
    """Gets the accuracy."""
    classes = torch.argmax(y_pred, dim=1)
    if len(y_true.shape) > 1:
        labels = torch.argmax(y_true, dim=1)
    else:
        labels = y_true
    _accuracy = torch.mean((classes == labels).float())
    return _accuracy


def sparsity(model):
    """Gets the sparsity of the model."""
    num_conn = 0
    max_num_conn = 0
    for layer in model.children():
        if isinstance(layer, sparselinear.SparseLinear):
            num_conn += layer.connectivity.shape[1]
            max_num_conn += layer.in_features * layer.out_features
    reg_term = (max_num_conn - num_conn) / max_num_conn
    return reg_term
