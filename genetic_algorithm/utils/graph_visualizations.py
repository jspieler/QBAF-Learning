"""Provides utils for the graph representation of the classifiers."""
import warnings
from math import cos, sin, atan

import numpy as np
import torch
import torch.nn as nn
import torch_sparse
from matplotlib import pyplot
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
device = torch.device("cpu")


def remove_connections(classifier):
    """Removes connections that are redundant and/or not meaningful.

    Connections between the input and the hidden layer are removed if they have no further connection
    towards the output layer.
    Connections between the hidden layer and the output layer that do not have a connection to a previous layer
    are translated, added to the bias values of the output layer and then removed

    Note:
        currently only works for one hidden layer
    """
    copy_ind = []  # indices of elements that remain
    ind = []
    for conn1 in classifier.sparse_linear1.connectivity[0]:
        if conn1 in classifier.sparse_linear2.connectivity[1]:
            idx = (classifier.sparse_linear1.connectivity[0] == conn1).nonzero()
            for tensor in idx:
                copy_ind.append(tensor.item())
        elif conn1 not in classifier.sparse_linear2.connectivity[1]:
            idx = (classifier.sparse_linear1.connectivity[0] == conn1).nonzero()
            for tensor in idx:
                ind.append(tensor.item())
    # remove duplicates
    copy_ind = list(set(copy_ind))
    ind = list(set(ind))
    copy_ind = torch.tensor(copy_ind, dtype=torch.long)
    connectivity1 = torch.index_select(
        classifier.sparse_linear1.connectivity, 1, copy_ind
    )
    iw1 = classifier.sparse_linear1.weight._indices()
    w1_ind = [
        torch.prod((connectivity1.T[i] == iw1.T), dim=1).nonzero()[0]
        for i in range(connectivity1.size(1))
    ]  # indices for weights
    w1_ind = torch.tensor(w1_ind, dtype=torch.long)
    vw1 = classifier.sparse_linear1.weight._values()
    val_weight1 = torch.index_select(vw1, 0, w1_ind)
    b1 = classifier.sparse_linear1.bias.detach()
    b_ind = torch.unique(connectivity1[0])
    bias1 = torch.index_select(b1, 0, b_ind)

    # connections between hidden layer and output
    copy_ind = []
    bias_ind = []
    for conn2 in classifier.sparse_linear2.connectivity[1]:
        if conn2 in classifier.sparse_linear1.connectivity[0]:
            idx = (classifier.sparse_linear2.connectivity[1] == conn2).nonzero()
            for tensor in idx:
                copy_ind.append(tensor.item())
        elif conn2 not in classifier.sparse_linear1.connectivity[0]:
            idx = (classifier.sparse_linear2.connectivity[1] == conn2).nonzero()
            for tensor in idx:
                bias_ind.append(tensor.item())
    copy_ind = list(set(copy_ind))
    copy_ind = torch.tensor(copy_ind, dtype=torch.long)
    connectivity2 = torch.index_select(
        classifier.sparse_linear2.connectivity, 1, copy_ind
    )
    iw2 = classifier.sparse_linear2.weight._indices()
    w2_ind = [
        torch.prod((connectivity2.T[i] == iw2.T), dim=1).nonzero()[0]
        for i in range(connectivity2.size(1))
    ]  # indices for weights
    w2_ind = torch.tensor(w2_ind, dtype=torch.long)
    ind_weight2 = torch.index_select(iw2, 1, w2_ind)
    vw2 = classifier.sparse_linear2.weight._values()
    val_weight2 = torch.index_select(vw2, 0, w2_ind)

    if bias_ind != None:
        bias_ind = list(set(bias_ind))
        bias_ind = torch.tensor(bias_ind, dtype=torch.long)
        for b in bias_ind:
            out_idx = classifier.sparse_linear2.connectivity[0, b]
            bias_idx = classifier.sparse_linear2.connectivity[1, b]
            tt = torch.tensor([out_idx, bias_idx])
            w_ind = torch.prod(
                (tt.T == classifier.sparse_linear2.weight._indices().T), dim=1
            ).nonzero()[0]
            w = classifier.sparse_linear2.weight._values()[w_ind]
            bias2 = classifier.sparse_linear2.bias.detach()[out_idx].unsqueeze(dim=0)
            bias2 += w.data * torch.sigmoid(
                classifier.sparse_linear1.bias.data[bias_idx]
            )
            with torch.no_grad():
                classifier.sparse_linear2.bias.data[out_idx] = bias2

    # store indices of used inputs (required for labelling of graph)
    input_ind = torch.unique(connectivity1[1])
    # "reset" weight indices and number of features
    ind = connectivity1
    out_ind = rankdata(ind[0], method="dense") - 1
    inp_ind = rankdata(ind[1], method="dense") - 1
    ind1 = torch.LongTensor(np.stack((out_ind, inp_ind), axis=0))
    classifier.sparse_linear1.connectivity = ind1
    classifier.sparse_linear1.in_features = len(
        torch.unique(classifier.sparse_linear1.connectivity[1])
    )
    classifier.sparse_linear1.out_features = len(
        torch.unique(classifier.sparse_linear1.connectivity[0])
    )

    coalesce_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    indices, values = torch_sparse.coalesce(
        ind1,
        val_weight1,
        classifier.sparse_linear1.out_features,
        classifier.sparse_linear1.in_features,
    )
    weights = nn.Parameter(values.cpu())

    ind = connectivity2
    # out_ind = rankdata(ind[0], method='dense') - 1
    out_ind = ind[0]  # number of outputs stays the same
    inp_ind = rankdata(ind[1], method="dense") - 1
    ind2 = torch.LongTensor(np.stack((out_ind, inp_ind), axis=0))
    classifier.sparse_linear2.connectivity = ind2
    classifier.sparse_linear2.in_features = len(
        torch.unique(classifier.sparse_linear2.connectivity[1])
    )
    # classifier.sparse_linear2.out_features = len(torch.unique(classifier.sparse_linear2.connectivity[0]))  # number of output neurons should not change

    indices2, values2 = torch_sparse.coalesce(
        ind2,
        val_weight2,
        classifier.sparse_linear2.out_features,
        classifier.sparse_linear2.in_features,
    )
    weights2 = nn.Parameter(values2.cpu())

    # assign "new" weights to classifier
    with torch.no_grad():
        classifier.sparse_linear1.bias.data = bias1
        classifier.sparse_linear1.weights = weights  #
        classifier.sparse_linear1.indices = indices
        classifier.sparse_linear2.weights = weights2  #
        classifier.sparse_linear2.indices = indices2
    # update sparsity
    classifier.sparse_linear1.sparsity = 1 - (
        classifier.sparse_linear1.connectivity.shape[1]
        / (
            classifier.sparse_linear1.in_features
            * classifier.sparse_linear1.out_features
        )
    )
    classifier.sparse_linear2.sparsity = 1 - (
        classifier.sparse_linear2.connectivity.shape[1]
        / (
            classifier.sparse_linear2.in_features
            * classifier.sparse_linear2.out_features
        )
    )
    return classifier, input_ind


# visualization based on https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network
class Neuron:
    def __init__(self, x, y, neuron_radius=0.5):
        self.x = x
        self.y = y
        self.neuron_radius = neuron_radius

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=self.neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer:
    def __init__(self, network, number_of_neurons, weights):
        self.network = network
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        y = self.__calculate_horizontal_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(self.x, y)
            neurons.append(neuron)
            y += self.network.vertical_distance_between_neurons
        return neurons

    def __calculate_horizontal_margin_so_layer_is_centered(self, number_of_neurons):
        return (
            self.network.vertical_distance_between_neurons
            * (self.network.number_of_neurons_in_widest_layer - number_of_neurons)
            / 2
        )

    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return (
                self.previous_layer.x + self.network.horizontal_distance_between_layers
            )
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, weight):
        angle = atan((neuron2.y - neuron1.y) / float(neuron2.x - neuron1.x))
        x_adjustment = self.network.neuron_radius * cos(angle)
        y_adjustment = self.network.neuron_radius * sin(angle)
        line_x_data = (neuron2.x + x_adjustment, neuron1.x - x_adjustment)
        line_y_data = (neuron2.y + y_adjustment, neuron1.y - y_adjustment)
        if weight > 0:
            ls = "--"

        elif weight < 0:
            ls = "-"

        else:
            ls = "None"
        line = pyplot.Line2D(
            line_x_data, line_y_data, linestyle=ls, color="k", linewidth=1.0
        )
        pyplot.gca().add_line(line)
        # add arrow head to line
        if weight != 0:
            # weight positioning has to be improved
            dy = neuron2.y - neuron1.y
            dx = neuron2.x - neuron1.x
            rotn = -np.degrees(np.arctan(dy / dx))
            pyplot.annotate(
                " ",
                (neuron1.x - x_adjustment, neuron1.y - y_adjustment),
                xytext=(neuron2.x + 2 * x_adjustment, neuron2.y + y_adjustment),
                arrowprops=dict(
                    linewidth=0,
                    arrowstyle="-|>",
                    shrinkA=0,
                    shrinkB=0,
                    edgecolor="none",
                    facecolor="k",
                    linestyle="solid",
                ),
            )
            pyplot.annotate(
                round(weight.item(), 2),
                (neuron2.x + 2 * x_adjustment, neuron2.y + 2 * y_adjustment),
                ha="center",
                va="bottom",
                rotation=rotn,
            )

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(
                    len(self.previous_layer.neurons)
                ):
                    previous_layer_neuron = self.previous_layer.neurons[
                        previous_layer_neuron_index
                    ]
                    weight = self.previous_layer.weights[
                        this_layer_neuron_index, previous_layer_neuron_index
                    ]
                    self.__line_between_two_neurons(
                        neuron, previous_layer_neuron, weight
                    )

    def annotate(self, input_labels, output_labels, layerType=0):
        y_text = (
            self.network.vertical_distance_between_neurons
            * self.network.number_of_neurons_in_widest_layer
        )
        if layerType == 0:
            for li, neuron in enumerate(self.neurons):
                pyplot.annotate(
                    input_labels[li],
                    xy=(neuron.x - neuron.neuron_radius, neuron.y),
                    xytext=(neuron.x - 2.5, neuron.y),
                    va="center",
                    ha="right",
                )
            pyplot.text(self.x, y_text, "Input Layer", ha="center", fontsize=12)
        elif layerType == -1:
            for lo, neuron in enumerate(self.neurons):
                pyplot.annotate(
                    output_labels[lo],
                    xy=(neuron.x + neuron.neuron_radius, neuron.y),
                    xytext=(neuron.x + 1, neuron.y),
                    va="center",
                    ha="left",
                )
            pyplot.text(self.x, y_text, "Output Layer", ha="center", fontsize=12)
        else:
            pyplot.text(
                self.x,
                y_text,
                "Hidden Layer" + str(layerType),
                ha="center",
                fontsize=12,
            )


class NeuralNetwork:
    def __init__(
        self,
        number_of_neurons_in_widest_layer,
        horizontal_distance_between_layers=6,
        vertical_distance_between_neurons=2,
        neuron_radius=0.5,
    ):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.horizontal_distance_between_layers = horizontal_distance_between_layers
        self.vertical_distance_between_neurons = vertical_distance_between_neurons
        self.neuron_radius = neuron_radius
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self, input_labels, output_labels):
        pyplot.figure(figsize=(10, 6))
        pyplot.gca().invert_yaxis()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw()
            layer.annotate(input_labels, output_labels, i)
        pyplot.axis("scaled")
        pyplot.axis("off")
        pyplot.tight_layout()
        pyplot.ion()
        pyplot.show()
        pyplot.draw()
        pyplot.pause(0.01)
