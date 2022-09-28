"""Provides plots."""
import warnings

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

warnings.filterwarnings("ignore")


def plot_fitness(best_fitness, mean_fitness):
    """Plots the best and mean fitness over the number of generations."""
    plt.figure()
    plt.plot(range(len(best_fitness)), best_fitness, "-k", label="Best individual")
    plt.plot(range(len(mean_fitness)), mean_fitness, "--k", label="Mean of population")
    plt.xlabel("Number of generations")
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.01)


def plot_loss(training_loss, validation_loss):
    """Plots the training and validation loss over the number of generations."""
    plt.figure()
    plt.plot(range(len(training_loss)), training_loss, "-k", label="Training loss")
    plt.plot(
        range(len(validation_loss)), validation_loss, "--k", label="Validation loss"
    )
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.01)


def plot_conf_matrix(y_test_non_category, y_pred_non_category, class_names):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_test_non_category, y_pred_non_category)
    fig, ax = plot_confusion_matrix(
        conf_mat=cm,
        show_absolute=True,
        show_normed=False,
        colorbar=True,
        cmap=plt.cm.gray_r,
        class_names=class_names,
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("Confusion Matrix")
    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.01)
