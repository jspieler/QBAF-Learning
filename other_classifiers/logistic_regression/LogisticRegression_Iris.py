import csv
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# set fixed seeds for reproducibility
np.random.seed(2021)  # scikit-learn also uses numpy random seed

num_runs = 10

# create csv to store results
with open('logistic_regression_iris.csv', 'w') as file:
    writer = csv.writer(file)
    header = ["Training accuracy", "Test accuracy", "Recall", "Precision", "F1 score"]
    writer.writerow(header)  # write the header

iris_data = load_iris()
x = iris_data.data
y = iris_data.target

# store results
train_acc = np.empty(num_runs)
test_acc = np.empty(num_runs)
prec = np.empty(num_runs)
rec = np.empty(num_runs)
f1_sc = np.empty(num_runs)

for i in range(0, num_runs):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    logistic_regression = LogisticRegression(max_iter=1000)
    y_pred_train = logistic_regression.fit(X_train, y_train)
    train_accuracy = y_pred_train.score(X_train, y_train)

    y_pred_test = logistic_regression.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    precision = precision_score(y_test, y_pred_test, average="macro")
    recall = recall_score(y_test, y_pred_test, average="macro")
    f1 = f1_score(y_test, y_pred_test, average="macro")
    with open('logistic_regression_iris.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(
            [round(train_accuracy, 4), round(test_accuracy, 4), round(recall, 4),
             round(precision, 4), round(f1, 4)])
    train_acc[i] = train_accuracy
    test_acc[i] = test_accuracy
    prec[i] = precision
    rec[i] = recall
    f1_sc[i] = f1

print("Training accuracy - mean: {:.4}, std: {:.4}".format(np.mean(train_acc, axis=0), np.std(train_acc, axis=0)))
print("Test accuracy - mean: {:.4}, std: {:.4}".format(np.mean(test_acc, axis=0), np.std(test_acc, axis=0)))
print("Precision - mean: {:.4}, std: {:.4}".format(np.mean(prec, axis=0), np.std(prec, axis=0)))
print("Recall - mean: {:.4}, std: {:.4}".format(np.mean(rec, axis=0), np.std(rec, axis=0)))
print("F1 score - mean: {:.4}, std: {:.4}".format(np.mean(f1_sc, axis=0), np.std(f1_sc, axis=0)))
