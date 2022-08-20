"""Provides loss functions."""
import torch


def cross_entropy_one_hot(input, target):
    """Gets the cross entropy loss for one hot encoding."""
    _, labels = target.max(dim=1)
    return torch.nn.CrossEntropyLoss()(input, labels)


def binary_cross_entropy_one_hot(input, target):
    """Gets the binary cross entropy loss for one hot encoding."""
    _, labels = input.max(dim=1)
    # casting necessary
    labels = labels.type(torch.LongTensor)
    target = target.type(torch.LongTensor)
    return torch.nn.BCELoss()(labels, target)
