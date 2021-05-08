# Learning Gradual Argumentation Frameworks using Genetic Algorithms

Research project on structure learning of Gradual Bipolar Argumentation Graphs (G-BAGs) using Genetic Algorithms.
G-BAGs are implemented in PyTorch as sparse multilayer perceptrons.

Code to reproduce experiments.

## Motivation and Background
A *Gradual Bipolar Argumentation Graph* (G-BAG) is defined as a quadruple (<img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}">, Att, Sup, w), where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}"> is a finite set of arguments, <img src="https://render.githubusercontent.com/render/math?math=Att \subseteq \mathcal{A} \times \mathcal{A}"> is the attack relation, <img src="https://render.githubusercontent.com/render/math?math=Att \subseteq \mathcal{A} \times \mathcal{A}"><img src="https://render.githubusercontent.com/render/math?math=Sup \subseteq \mathcal{A} \times \mathcal{A}"> is the support relation and <img src="https://render.githubusercontent.com/render/math?math=w : \mathcal{A} \rightarrow [0,1]"> is a weight function.
A G-BAG can be graphically represented as a directed graph consisting of nodes that show the arguments and edges that describe the relation between arguments. Attack relations are represented by solid and support relations by dashed edges.

![image](img/GBAG.png?raw=true "Graphical representation of a Gradual Bipolar Argumentation Graph")

As shown by [[1]](#1), a multi-layer perceptron (MLP) can be interpreted as a G-BAG when an aggregation function is used that uses summation and an influence function that is similar to the activation function of a neural network. Since only acyclic G-BAGs are considered, they can be seen as sparse MLPs. This allows to perform parameter learning by means of a forward-pass and backpropagation.

The aim is then to use genetic algorithms in order to learn the structure of G-BAGs.

## Prerequisites

Following packages and libraries are required:
* PyTorch
* [SparseLinear](https://github.com/hyeon95y/SparseLinear)
* Matplotlib
* Numpy
* Pandas
* Scikit-learn

Follow installation instructions and install [Pytorch Sparse](https://github.com/rusty1s/pytorch_sparse) package before installing ```SparseLinear```.

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/jspieler/QBAF-Learning.git
   ```
2. Install required packages

3. If you use ```pip``` to install ```SparseLinear```, make sure that the definition of the sparsity for user-defined connectivity matrices is correct.



## Usage

Examples provided for the iris data set, the adult income and the mushroom data set.


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

Please contact me for questions.

## References
<a id="1">[1]</a>
Potyka, N. (2020) -
Foundations for Solving Classification Problems with Quantitative Abstract Argumentation,
Conference: International Workshop on Explainable and Interpretable Machine Learning (XI-ML)