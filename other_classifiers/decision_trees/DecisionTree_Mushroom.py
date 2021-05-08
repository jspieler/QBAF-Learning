import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


def accuracy(y_true, y_predict):
    count = 0
    for j in range(0, len(y_true)):
        if y_true[j] == y_predict[j]:
            count = count + 1
    return count * 1.0 / len(y_true)


# set fixed seeds for reproducibility
np.random.seed(2021)  # scikit-learn also uses numpy random seed

num_runs = 10

# create csv to store results
with open('decision_tree_mushroom.csv', 'w') as file:
    writer = csv.writer(file)
    header = ["Parameter", "Training accuracy", "Test accuracy", "Recall", "Precision", "F1 score"]
    writer.writerow(header)  # write the header

data = pd.read_csv("../../datasets/mushrooms.csv", header=0, na_values='?')
# drop column with missing values
data.drop('stalk_root', axis=1, inplace=True)
data.head()
X_, y = data.iloc[:, 1:23], data.iloc[:, 0]
le = LabelEncoder()
y = le.fit_transform(y)
X = pd.get_dummies(X_)

# store results
train_acc_ld = np.empty(num_runs)  # ld: limited depth
test_acc_ld = np.empty(num_runs)
prec_ld = np.empty(num_runs)
rec_ld = np.empty(num_runs)
f1_sc_ld = np.empty(num_runs)
train_acc_best = np.empty(num_runs)
test_acc_best = np.empty(num_runs)
prec_best = np.empty(num_runs)
rec_best = np.empty(num_runs)
f1_sc_best = np.empty(num_runs)

for i in range(0, num_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fitting the decision tree with default hyperparameters
    dt_default = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10)
    dt_default.fit(X_train, y_train)

    # making predictions
    y_pred_default = dt_default.predict(X_test)
    train_accuracy = accuracy(y_train, dt_default.predict(X_train))
    test_accuracy = accuracy(y_test, dt_default.predict(X_test))

    precision = precision_score(y_test, y_pred_default, average="macro")
    recall = recall_score(y_test, y_pred_default, average="macro")
    f1 = f1_score(y_test, y_pred_default, average="macro")
    with open('decision_tree_mushroom.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(
            [dt_default.get_params(), round(train_accuracy, 4), round(test_accuracy, 4), round(recall, 4),
             round(precision, 4), round(f1, 4)])
    train_acc_ld[i] = train_accuracy
    test_acc_ld[i] = test_accuracy
    prec_ld[i] = precision
    rec_ld[i] = recall
    f1_sc_ld[i] = f1

    # dot_data = tree.export_graphviz(dt_default,
    #                                 feature_names=X.columns.values,
    #                                 class_names=le.classes_,
    #                                 out_file=None,
    #                                 filled=True,
    #                                 rounded=True,
    #                                 special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    #
    # nodes = graph.get_node_list()
    #
    # colors = ('gray35', 'gray60', 'gray90', 'white')
    #
    # for node in nodes:
    #     if node.get_name() not in ('node', 'edge'):
    #         values = dt_default.tree_.value[int(node.get_name())][0]
    #         #color only nodes where only one class is present
    #         if max(values) == sum(values):
    #             node.set_fillcolor(colors[np.argmax(values)])
    #         #mixed nodes get the default color
    #         else:
    #             node.set_fillcolor(colors[-1])

    # graph.write_pdf('Mushroom_DecisionTree_depth.pdf')

    # Create the parameter grid
    param_grid = {
        'max_depth': range(3, 12, 1),
        'min_samples_leaf': range(1, 50, 5),
        'min_samples_split': range(2, 50, 5),
        'criterion': ["entropy", "gini"]
    }

    n_folds = 5

    # Instantiate the grid search model
    dtree = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid,
                               cv=n_folds, verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    # Training accuracy and Test accuracy
    y_pred_train = grid_search.predict(X_train)
    y_pred_test = grid_search.predict(X_test)
    train_accuracy = accuracy(y_train, y_pred_train)
    test_accuracy = accuracy(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average="macro")
    recall = recall_score(y_test, y_pred_test, average="macro")
    f1 = f1_score(y_test, y_pred_test, average="macro")
    with open('decision_tree_mushroom.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(
            [grid_search.best_params_, round(train_accuracy, 4), round(test_accuracy, 4), round(recall, 4),
             round(precision, 4), round(f1, 4)])
    train_acc_best[i] = train_accuracy
    test_acc_best[i] = test_accuracy
    prec_best[i] = precision
    rec_best[i] = recall
    f1_sc_best[i] = f1

    # dot_data2 = tree.export_graphviz(grid_search.best_estimator_,
    #                                 feature_names=X.columns.values,
    #                                 class_names=le.classes_,
    #                                 out_file=None,
    #                                 filled=True,
    #                                 rounded=True,
    #                                 special_characters=True)
    # graph2 = pydotplus.graph_from_dot_data(dot_data2)
    # nodes = graph2.get_node_list()
    #
    # colors = ('gray35', 'gray60', 'gray90', 'white')
    #
    # for node in nodes:
    #     if node.get_name() not in ('node', 'edge'):
    #         values = grid_search.best_estimator_.tree_.value[int(node.get_name())][0]
    #         #color only nodes where only one class is present
    #         if max(values) == sum(values):
    #             node.set_fillcolor(colors[np.argmax(values)])
    #         #mixed nodes get the default color
    #         else:
    #             node.set_fillcolor(colors[-1])
    # graph2.write_pdf('Mushroom_DecisionTree_best.pdf')

print("Training accuracy limited depth - mean: {:.4}, std: {:.4}".format(np.mean(train_acc_ld, axis=0),
                                                                         np.std(train_acc_ld, axis=0)))
print("Test accuracy limited depth - mean: {:.4}, std: {:.4}".format(np.mean(test_acc_ld, axis=0),
                                                                     np.std(test_acc_ld, axis=0)))
print("Precision limited depth - mean: {:.4}, std: {:.4}".format(np.mean(prec_ld, axis=0), np.std(prec_ld, axis=0)))
print("Recall limited depth - mean: {:.4}, std: {:.4}".format(np.mean(rec_ld, axis=0), np.std(rec_ld, axis=0)))
print("F1 score limited depth - mean: {:.4}, std: {:.4}".format(np.mean(f1_sc_ld, axis=0), np.std(f1_sc_ld, axis=0)))

print("Training accuracy best - mean: {:.4}, std: {:.4}".format(np.mean(train_acc_best, axis=0),
                                                                np.std(train_acc_best, axis=0)))
print("Test accuracy best - mean: {:.4}, std: {:.4}".format(np.mean(test_acc_best, axis=0),
                                                            np.std(test_acc_best, axis=0)))
print("Precision best - mean: {:.4}, std: {:.4}".format(np.mean(prec_best, axis=0), np.std(prec_best, axis=0)))
print("Recall best - mean: {:.4}, std: {:.4}".format(np.mean(rec_best, axis=0), np.std(rec_best, axis=0)))
print("F1 score best - mean: {:.4}, std: {:.4}".format(np.mean(f1_sc_best, axis=0), np.std(f1_sc_best, axis=0)))
