import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

path_train = "../data/Q1/train.csv"
path_valid = "../data/Q1/valid.csv"
path_test = "../data/Q1/test.csv"

df_train = pd.read_csv(path_train)
df_valid = pd.read_csv(path_valid)
df_test = pd.read_csv(path_test)

label = "income"
label_0 = " <=50K"
label_1 = " >50K"

# Returns a list of continuous and categorical attributes separately.
def separate_attributes(df):
    continuous = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return continuous, categorical

continuous, categorical = separate_attributes(df_train)
# Remove the label column from the list of categorical attributes.
if label in categorical:
    categorical.remove(label)

# One-hot encoding: create new numeric features for each categorical attribute.
def one_hot_encoding(df, categorical):
    one_hot_categorical = []
    for attribute in categorical:
        # For each unique value in the categorical column.
        for val in df[attribute].unique():
            new_col = attribute + "_" + str(val)
            df[new_col] = df[attribute].apply(lambda x: 1 if x == val else 0)
            one_hot_categorical.append(new_col)
    return one_hot_categorical

one_hot_categorical = one_hot_encoding(df_train, categorical)
one_hot_encoding(df_valid, categorical)
one_hot_encoding(df_test, categorical)

# Remove categorical string type values
df_train = df_train.drop(columns=categorical)
df_valid = df_valid.drop(columns=categorical)
df_test  = df_test.drop(columns=categorical)

# Set absent one_hot_train columns to 0 in df_valid and df_test
def update_remaining_one_hot_cols(df, one_hot_categorical):
    for attribute in one_hot_categorical:
        if attribute not in df.columns:
            df[attribute] = 0

update_remaining_one_hot_cols(df_valid, one_hot_categorical)
update_remaining_one_hot_cols(df_test, one_hot_categorical)

# Generating sklearn format for training
one_hot_attribute = [attribute for attribute in df_train.columns if attribute != label]
X_train = df_train[one_hot_attribute]
X_valid = df_valid[one_hot_attribute]
X_test = df_test[one_hot_attribute]
y_train = df_train[label]
y_valid = df_valid[label]
y_test = df_test[label]


parameter_grid = {
    "n_estimators": [50, 150, 250, 350],
    "max_features": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    "min_samples_split": [2, 4, 6, 8, 10]
}

grid = list(ParameterGrid(parameter_grid))

oob_accuracy_max = 0
optimal_model = None
optimal_parameters = None
output = []

for parameters in grid:
    clf = RandomForestClassifier(
        criterion="entropy",
        n_estimators=parameters["n_estimators"],
        max_features=parameters["max_features"],
        min_samples_split=parameters["min_samples_split"],
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    y_train_predicted = clf.predict(X_train)
    y_valid_predicted = clf.predict(X_valid)
    y_test_predicted = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_predicted)
    oob_accuracy = clf.oob_score_
    valid_accuracy = accuracy_score(y_valid, y_valid_predicted)
    test_accuracy = accuracy_score(y_test, y_test_predicted)

    output.append({
        "params": parameters,
        "train_acc": train_accuracy,
        "oob_acc": oob_accuracy,
        "valid_acc": valid_accuracy,
        "test_acc": test_accuracy
    })

    if oob_accuracy > oob_accuracy_max:
        oob_accuracy_max = oob_accuracy
        optimal_model = clf
        optimal_parameters = parameters

y_train_predicted = optimal_model.predict(X_train)
y_valid_predicted = optimal_model.predict(X_valid)
y_test_predicted = optimal_model.predict(X_test)
print("Optimal Parameters:", optimal_parameters)
print("Training Accuracy:", accuracy_score(y_train, y_train_predicted))
print("OOB Accuracy:", optimal_model.oob_score_)
print("Validation Accuracy:", accuracy_score(y_valid, y_valid_predicted))
print("Test Accuracy:", accuracy_score(y_test, y_test_predicted))