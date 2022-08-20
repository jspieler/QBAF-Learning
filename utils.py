import warnings

import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")


def binning(features, n_bins, strategy, encode, feature_names):
    """
    Returns binned features and the corresponding labels for each bin
    'n_bins' can either be an integer or a list/numpy array of n integers (different number of bins for n features)
    'strategy' and 'encode' are inputs for Scikit-learns KBinsDiscretizer
    """
    X = []
    binning_feature_names = []
    for i in range(features.shape[1]):
        x = features.iloc[:, i]
        if isinstance(n_bins, (list, np.ndarray)):
            num_bins = n_bins[i]
        elif isinstance(n_bins, int):
            num_bins = n_bins
        else:
            raise ValueError("`n_bins` should be a an integer, list or numpy array.")
        est = KBinsDiscretizer(n_bins=num_bins, encode=encode, strategy=strategy)
        x = est.fit_transform(x.values.reshape(-1, 1))
        bin_edges = est.bin_edges_[0]
        X.append(x)
        for j in range(num_bins):
            if j == 0:
                fname = feature_names[i] + " x < " + "{:.1f}".format(bin_edges[j + 1])
            elif j == num_bins - 1:
                fname = (
                    feature_names[i] + " " + "{:.1f}".format(bin_edges[j]) + "$\leq x$"
                )
            else:
                fname = (
                    feature_names[i]
                    + " "
                    + "{:.1f}".format(bin_edges[j])
                    + "$\leq x <$"
                    + "{:.1f}".format(bin_edges[j + 1])
                )
            binning_feature_names.append(fname)
    df = pd.DataFrame(np.concatenate(X, axis=1))
    return df, binning_feature_names


def create_csv_with_header(fname):
    """Creates a csv to store the results and writes a header."""
    with open(fname, "w") as file:
        writer = csv.writer(file)
        header = [
            "Parameters",
            "Number of connections",
            "Training accuracy",
            "Test accuracy",
            "Recall",
            "Precision",
            "F1 score",
        ]
        writer.writerow(header)
