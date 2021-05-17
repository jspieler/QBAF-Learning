import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pydotplus
from pandas import read_csv
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


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
with open('decision_tree_adult_income.csv', 'w') as file:
    writer = csv.writer(file)
    header = ["Parameter", "Training accuracy", "Test accuracy", "Recall", "Precision", "F1 score"]
    writer.writerow(header)  # write the header

# load dataset
filename = '../../datasets/adult-all.csv'
dataframe = read_csv(filename, header=None, na_values='?',
                     names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                            "native-country", "Income"])
# drop rows with missing
dataframe = dataframe.dropna()
target = dataframe.values[:, -1]
# split into inputs and outputs
last_ix = len(dataframe.columns) - 1
X_, y = dataframe.drop("Income", axis=1), dataframe["Income"]
# select categorical and numerical features
cat_ix = X_.select_dtypes(include=['object', 'bool']).columns
num_ix = X_.select_dtypes(include=['int64', 'float64']).columns
# label encode the target variable to have the classes 0 and 1
le = LabelEncoder()
y = le.fit_transform(y)
# one-hot encoding of categorical features
df_cat = pd.get_dummies(X_[cat_ix])
# binning of numerical features
df_num = X_[num_ix]
X = pd.concat([df_cat.reset_index(drop=True), df_num.reset_index(drop=True)], axis=1)

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
    # split training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Fitting the decision tree with default hyperparameters
    dt_default = DecisionTreeClassifier(max_depth=2, min_impurity_decrease=0.01)
    dt_default.fit(X_train, y_train)

    # making predictions
    y_pred_default = dt_default.predict(X_test)
    train_accuracy = accuracy(y_train, dt_default.predict(X_train))
    test_accuracy = accuracy(y_test, dt_default.predict(X_test))

    precision = precision_score(y_test, y_pred_default, average="macro")
    recall = recall_score(y_test, y_pred_default, average="macro")
    f1 = f1_score(y_test, y_pred_default, average="macro")
    with open('decision_tree_adult_income.csv', 'a') as file:
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
    #                                 class_names=['below 50K', 'above 50K'],
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

    # graph.write_pdf('adult_income_decision_tree_depth.pdf')

    # Create the parameter grid
    param_grid = {
        'max_depth': range(5, 10, 1),
        'min_samples_leaf': range(1, 50, 10),
        'min_samples_split': range(2, 50, 10),
        'criterion': ["entropy", "gini"]
    }

    n_folds = 5

    # Instantiate the grid search model
    dtree = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid,
                               cv=n_folds, verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    # printing the optimal accuracy score and hyperparameters
    print("Best score", grid_search.best_score_)
    # Training accuracy and Test accuracy
    y_pred_train = grid_search.predict(X_train)
    y_pred_test = grid_search.predict(X_test)
    train_accuracy = accuracy(y_train, y_pred_train)
    test_accuracy = accuracy(y_test, y_pred_test)
    print('Training accuracy best classifier', train_accuracy)
    print('Test accuracy best classifier', test_accuracy, '\n')
    precision = precision_score(y_test, y_pred_test, average="macro")
    recall = recall_score(y_test, y_pred_test, average="macro")
    f1 = f1_score(y_test, y_pred_test, average="macro")
    with open('decision_tree_adult_income.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(
            [grid_search.best_params_, round(train_accuracy, 4), round(test_accuracy, 4), round(recall, 4),
             round(precision, 4), round(f1, 4)])
    train_acc_best[i] = train_accuracy
    test_acc_best[i] = test_accuracy
    prec_best[i] = precision
    rec_best[i] = recall
    f1_sc_best[i] = f1

    # model with optimal hyperparameters
    # dot_data2 = tree.export_graphviz(grid_search.best_estimator_,
    #                                 feature_names=X.columns.values,
    #                                 class_names=['below 50K', 'above 50K'],
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
    # graph2.write_pdf('adult_income_DecisionTree_best.pdf')

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
