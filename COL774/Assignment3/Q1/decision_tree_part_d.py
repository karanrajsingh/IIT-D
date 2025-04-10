import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Read the datasets
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

# d-i: Varying max_depth
max_depth = [25, 35, 45, 55]
train_accuracy = []
valid_accuracy = []
test_accuracy = []

for depth in max_depth:
    clf = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    clf.fit(X_train, y_train)
    y_train_predicted = clf.predict(X_train)
    y_valid_predicted = clf.predict(X_valid)
    y_test_predicted = clf.predict(X_test)
    train_accuracy.append(accuracy_score(y_train, y_train_predicted))
    valid_accuracy.append(accuracy_score(y_valid, y_valid_predicted))
    test_accuracy.append(accuracy_score(y_test, y_test_predicted))

# Accuracy v/s Max Depth
plt.figure()
plt.plot(max_depth, train_accuracy, label="Train Accuracy", marker='o')
plt.plot(max_depth, valid_accuracy, label="Validation Accuracy", marker='o')
plt.plot(max_depth, test_accuracy, label="Test Accuracy", marker='o')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs Max Depth")
plt.legend()
plt.grid(True)
plt.show()

max_acc = max(valid_accuracy)
max_index = valid_accuracy.index(max_acc)
depth_max_acc = max_depth[max_index]
print("Train accuracy: ", train_accuracy)
print("Valid accuracy: ", valid_accuracy)
print("Test accuracy: ", test_accuracy)
print("Depth for max validation accuracy", depth_max_acc)

# Part (ii): Varying ccp_alpha
ccp_alpha = [0.001, 0.01, 0.1, 0.2]
alpha_train_accuracy = []
alpha_valid_accuracy = []
alpha_test_accuracy = []

for alpha in ccp_alpha:
    clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    y_train_predicted = clf.predict(X_train)
    y_valid_predicted = clf.predict(X_valid)
    y_test_predicted = clf.predict(X_test)
    alpha_train_accuracy.append(accuracy_score(y_train, y_train_predicted))
    alpha_valid_accuracy.append(accuracy_score(y_valid, y_valid_predicted))
    alpha_test_accuracy.append(accuracy_score(y_test, y_test_predicted))

# Accuracy v/s ccp_alpha
plt.figure()
plt.plot(ccp_alpha, alpha_train_accuracy, label="Train Accuracy", marker='o')
plt.plot(ccp_alpha, alpha_valid_accuracy, label="Validation Accuracy", marker='o')
plt.plot(ccp_alpha, alpha_test_accuracy, label="Test Accuracy", marker='o')
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs CCP Alpha")
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

max_acc =  max(alpha_valid_accuracy)
max_index = alpha_valid_accuracy.index(max_acc)
ccp_max = ccp_alpha[max_index]
print("Train accuracy: ", alpha_train_accuracy)
print("Valid accuracy: ", alpha_valid_accuracy)
print("Test accuracy: ", alpha_test_accuracy)

clf = DecisionTreeClassifier(max_depth=depth_max_acc, criterion='entropy', ccp_alpha=ccp_max)
clf.fit(X_train, y_train)
y_train_predicted = clf.predict(X_train)
y_valid_predicted = clf.predict(X_valid)
y_test_predicted = clf.predict(X_test)
print("Train accuracy for best params: ", accuracy_score(y_train, y_train_predicted))
print("Validation accuracy for best params: ", accuracy_score(y_valid, y_valid_predicted))
print("Test accuracy for best params: ",accuracy_score(y_test, y_test_predicted))