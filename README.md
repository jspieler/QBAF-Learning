# Learning Gradual Argumentation Frameworks using Genetic Algorithms

Research project on structure learning of Gradual Bipolar Argumentation Graphs (G-BAGs) using Genetic Algorithms.\
G-BAGs are implemented in PyTorch as sparse multilayer perceptrons.

Code to reproduce experiments.

## Motivation and Background
An edge-weighted *Gradual Bipolar Argumentation Graph* (G-BAG) or *Gradual Argumentation Framework* (GAF) is defined as a quadruple
<img src="https://render.githubusercontent.com/render/math?math=(\mathcal{A}, E, \beta, w)">, where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}"> is a finite set of arguments, <img src="https://render.githubusercontent.com/render/math?math=E \subseteq \mathcal{A} \times \mathcal{A}"> is a set of edges between the arguments, <img src="https://render.githubusercontent.com/render/math?math=\beta : \mathcal{A} \rightarrow [0,1]"> is a function that assigns a base score <img src="https://render.githubusercontent.com/render/math?math=\beta"> to every argument and <img src="https://render.githubusercontent.com/render/math?math=w : E \rightarrow [0,1]"> is a function that assign a weight to every edge.
Edges with negative weights are called *attack* and edges with positive weights are called *support* relations.
A G-BAG can be graphically represented as a directed graph consisting of nodes that show the arguments and edges that describe the relations between arguments. Attack relations are represented by solid and support relations by dashed edges.

![image](img/GBAG.png?raw=true "Graphical representation of a Gradual Bipolar Argumentation Graph")

Arguments are evaluated by numerical values, so-called acceptance or *strength values*. The acceptability of arguments is determined by a semantics which provides a partial function that assigns acceptability degrees to arguments based on their initial base scores and the relations between the arguments. The initial base scores may be seen as the acceptability of an argument without taking support or attack relations into account. The acceptability degree of an argument is typically calculated by a modular update function consisting of an aggregation function <img src="https://render.githubusercontent.com/render/math?math=\alpha"> and an influence function <img src="https://render.githubusercontent.com/render/math?math=\iota">.

As shown by [[1]](#1), a multilayer perceptron (MLP) can be interpreted as a G-BAG when an aggregation function is used that uses summation and an influence function that is similar to the activation function of a neural network. Since only acyclic G-BAGs are considered, they can be seen as sparse MLPs. This allows to perform parameter learning by means of a forward-pass and backpropagation.

The aim is then to use genetic algorithms in order to learn the structure of G-BAGs. Since we are interested in well interpretable G-BAGs to solve classification problems, the fitness function balances sparseness and accuracy of the classifier. 

## Prerequisites

Following packages and libraries are required:
* PyTorch
* [SparseLinear](https://github.com/hyeon95y/SparseLinear)
* Matplotlib
* Numpy
* Pandas
* Scikit-learn
* SciPy

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

The following picture shows an example of a GAF for the iris dataset.

![image](img/GAF_iris.png?raw=true "Example of a GAF for the iris dataset")


## Results
The following table shows the results of GAFs compared to logistic regression and decision trees on the iris, the adult income and the mushroom dataset (test data). The best values are printed in bold. Two decision trees of different depth were used. One, whose best parameter set was determined using `sklearn.model_selection.GridSearchCV` and another one whose depth was limited to obtain a comparable performance to GAFs. Each of the performance measures is reported by their mean and standard deviation (%) for 10 runs.

![image](img/results.png?raw=true "Performance measures of different classifiers for test data sets reported by their mean and standard deviation for 10 runs")


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

Please contact me for questions.

## References
<a id="1">[1]</a>
Potyka, N. (2020) -
Foundations for Solving Classification Problems with Quantitative Abstract Argumentation,
Conference: International Workshop on Explainable and Interpretable Machine Learning (XI-ML)
