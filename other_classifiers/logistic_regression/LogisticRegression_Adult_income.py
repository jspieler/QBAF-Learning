import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# set fixed seeds for reproducibility
np.random.seed(2021)  # scikit-learn also uses numpy random seed

num_runs = 10

# create csv to store results
with open('logistic_regression_adult_income.csv', 'w') as file:
    writer = csv.writer(file)
    header = ["Training accuracy", "Test accuracy", "Recall", "Precision", "F1 score"]
    writer.writerow(header)  # write the header

filename = '../../datasets/adult-all.csv'
dataframe = pd.read_csv(filename, header=None, na_values='?',
                        names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                               "hours-per-week", "native-country", "Income"])
# drop rows with missing
dataframe = dataframe.dropna()
target = dataframe.values[:, -1]
# split into inputs and outputs
X_, y = dataframe.drop("Income", axis=1), dataframe["Income"]
# select categorical and numerical features
cat_ix = X_.select_dtypes(include=['object', 'bool']).columns
num_ix = X_.select_dtypes(include=['int64', 'float64']).columns
# label encode the target variable to have the classes 0 and 1
y = LabelEncoder().fit_transform(y)
df_cat = X_.drop(columns=num_ix, axis=1)
for idx in cat_ix:
    df_cat[idx] = LabelEncoder().fit_transform(X_[idx])
# binning of numerical features
df_num = X_.drop(columns=cat_ix, axis=1)
X = pd.concat([df_cat.reset_index(drop=True), pd.DataFrame(df_num).reset_index(drop=True)], axis=1)

# store results
train_acc = np.empty(num_runs)
test_acc = np.empty(num_runs)
prec = np.empty(num_runs)
rec = np.empty(num_runs)
f1_sc = np.empty(num_runs)

for i in range(0, num_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logistic_regression = LogisticRegression(max_iter=1000)
    y_pred_train = logistic_regression.fit(X_train, y_train)
    train_accuracy = y_pred_train.score(X_train, y_train)

    y_pred_test = logistic_regression.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    precision = precision_score(y_test, y_pred_test, average="macro")
    recall = recall_score(y_test, y_pred_test, average="macro")
    f1 = f1_score(y_test, y_pred_test, average="macro")
    with open('logistic_regression_adult_income.csv', 'a') as file:
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
