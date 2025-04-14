import pandas as pd
import tree
import math
import sys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

label = "income"
label_0 = " <=50K"
label_1 = " >50K"
categorical = None
continuous = None
one_hot_categorical = None

# returns a list of continuous and categorical attributes separately
def separate_attributes(df):
    global continuous, categorical
    continuous = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical.remove(label)
    return continuous, categorical

def entropy(df):

    count_0 = len(df[df[label] == label_0])
    count_1 = len(df[df[label] == label_1])
    prob_0 = float(count_0)/(count_0 + count_1)
    prob_1 = float(count_1)/(count_0 + count_1)
    log_0 = 0
    log_1 = 0
    if(prob_0 != 0):
        log_0 = prob_0*math.log(prob_0)
    if(prob_1 != 0):
        log_1 = prob_1*math.log(prob_1)   
    entropy = -(log_0 + log_1)
    return entropy

def choose_attribute(df):
    # H(Y)
    H_y = entropy(df)
    I_gain_max = 0
    split_attr = ""
    median_max = -sys.maxsize
    count = len(df)

    for attribute in categorical:
        attr_val = df[attribute].unique()
        H_y_val = 0
        for val in attr_val:
            df_attr = df[df[attribute] == val]
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr)        
        I_gain = H_y - H_y_val
        if I_gain > I_gain_max:
            split_attr = attribute
            I_gain_max = I_gain

    for attribute in continuous:
        H_y_val = 0
        median = df[attribute].median()
        
        df_attr = df[df[attribute] <= median]
        if len(df_attr) != 0:
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr)     
        
        df_attr = df[df[attribute] > median]
        if len(df_attr) != 0:
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr) 
        
        I_gain = H_y - H_y_val
        if I_gain > I_gain_max:
            split_attr = attribute
            I_gain_max = I_gain
            median_max = median

    return split_attr, median_max

def grow_tree(df, depth, parent_attribute = "", parent_val = -1, current_level = 1):

    df_label_0 = df[df[label] == label_0]
    df_label_1 = df[df[label] == label_1]
    if len(df_label_0) == 0 or len(df_label_1) == 0 or current_level >= depth:
        if current_level >= depth:
            if len(df_label_0) > len(df_label_1):
                label_val = label_0
            elif len(df_label_0) < len(df_label_1):
                label_val = label_1
            else:
                label_val = label_0
        elif len(df_label_0) == 0:
            label_val = label_1
        elif len(df_label_1) == 0:
            label_val = label_0
        return tree.Tree(attribute = "", parent_attribute = parent_attribute, parent_val = parent_val, median = -sys.maxsize, cat_val_list = [], child_list = [], label = label_val, level = current_level)

    attribute, median = choose_attribute(df)

    # Creating Decision Tree node
    node = tree.Tree(attribute = attribute, parent_attribute = parent_attribute, parent_val = parent_val, median = None, cat_val_list = [], child_list = [], label = -1, level = current_level)
    # Store majority label at each node
    if len(df_label_0) > len(df_label_1):
        label_val = label_0
    elif len(df_label_0) < len(df_label_1):
        label_val = label_1
    else:
        label_val = label_0
    node.label = label_val

    if attribute in categorical:
        cat_val_list = df[attribute].unique()
        node.cat_val_list = cat_val_list
        for val in cat_val_list:
            child_df = df[df[attribute] == val]
            child_node = grow_tree(child_df, depth, attribute, val, current_level + 1)
            if child_node is not None:
                node.add_child_node(child_node)
    elif attribute in continuous:
        node.median = median
        # Left child has values less or equal to median
        left_child_df = df[df[attribute] <= median]
        left_child = grow_tree(left_child_df, depth, attribute, "left", current_level + 1)
        if left_child is not None:
            node.add_child_node(left_child)
        # Right child has values greater than median
        right_child_df = df[df[attribute] > median]
        right_child = grow_tree(right_child_df, depth, attribute, "right", current_level + 1)
        if right_child is not None:
            node.add_child_node(right_child)
    return node

def predict(df, dec_tree):
    df["predicted"] = df.apply(lambda row: predict_label(row, dec_tree), axis=1)
    return df

def predict_label(row, current_node):
    if current_node.attribute == "":
        return current_node.label

    attribute = current_node.attribute

    if attribute in categorical:
        attr_val = row[attribute]
        for child in current_node.child_list:
            if child.parent_attribute == attribute and child.parent_val == attr_val:
                return predict_label(row, child)

    if attribute in continuous:
        attr_val = row[attribute]
        median = current_node.median
        if attr_val <= median:
            for child in current_node.child_list:
                if child.parent_val == "left":
                    return predict_label(row, child)
        else:
            for child in current_node.child_list:
                if child.parent_val == "right":
                    return predict_label(row, child)

    return current_node.label

def accuracy(df, label_col= label):    
    correct = (df[label_col].astype(str).str.strip().str.lower() == df["predicted"].astype(str).str.strip().str.lower()).sum()
    return float(correct)/len(df)

def one_hot_encoding(df, categorical):
    global one_hot_categorical
    one_hot_categorical = []
    for attribute in categorical:
        attr_val = df[attribute].unique()
        for val in attr_val:
            df[attribute + "_" + str(val)] = df[attribute].apply(lambda x: 1 if x == val else 0)
            one_hot_categorical.append(attribute + "_" + str(val))
    return one_hot_categorical

def choose_attribute_one_hot(df):
    # H(Y)
    H_y = entropy(df)
    I_gain_max = 0
    split_attr = ""
    median_max = -sys.maxsize
    count = len(df)

    for attribute in one_hot_categorical:
        attr_val = df[attribute].unique()
        H_y_val = 0
        for val in attr_val:
            df_attr = df[df[attribute] == val]
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr)        
        I_gain = H_y - H_y_val
        if I_gain > I_gain_max:
            split_attr = attribute
            I_gain_max = I_gain

    for attribute in continuous:
        H_y_val = 0
        median = df[attribute].median()
        
        df_attr = df[df[attribute] <= median]
        if len(df_attr) != 0:
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr)     
        
        df_attr = df[df[attribute] > median]
        if len(df_attr) != 0:
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr) 
        
        I_gain = H_y - H_y_val
        if I_gain > I_gain_max:
            split_attr = attribute
            I_gain_max = I_gain
            median_max = median

    return split_attr, median_max

def grow_tree_one_hot(df, depth, parent_attribute = "", parent_val = -1, current_level = 1):

    df_label_0 = df[df[label] == label_0]
    df_label_1 = df[df[label] == label_1]
    if len(df_label_0) == 0 or len(df_label_1) == 0 or current_level >= depth:
        if current_level >= depth:
            if len(df_label_0) > len(df_label_1):
                label_val = label_0
            elif len(df_label_0) < len(df_label_1):
                label_val = label_1
            else:
                label_val = label_0
        elif len(df_label_0) == 0:
            label_val = label_1
        elif len(df_label_1) == 0:
            label_val = label_0
        return tree.Tree(attribute = "", parent_attribute = parent_attribute, parent_val = parent_val, median = -sys.maxsize, cat_val_list = [], child_list = [], label = label_val, level = current_level)

    attribute, median = choose_attribute_one_hot(df)

    # Creating Decision Tree node
    node = tree.Tree(attribute = attribute, parent_attribute = parent_attribute, parent_val = parent_val, median = None, cat_val_list = [], child_list = [], label = -1, level = current_level)
    # Store majority label at each node
    if len(df_label_0) > len(df_label_1):
        label_val = label_0
    elif len(df_label_0) < len(df_label_1):
        label_val = label_1
    else:
        label_val = label_0
    node.label = label_val

    if attribute in one_hot_categorical:
        cat_val_list = df[attribute].unique()
        node.cat_val_list = cat_val_list
        for val in cat_val_list:
            child_df = df[df[attribute] == val]
            child_node = grow_tree_one_hot(child_df, depth, attribute, val, current_level + 1)
            if child_node is not None:
                node.add_child_node(child_node)
    elif attribute in continuous:
        node.median = median
        # Left child has values less or equal to median
        left_child_df = df[df[attribute] <= median]
        left_child = grow_tree_one_hot(left_child_df, depth, attribute, "left", current_level + 1)
        if left_child is not None:
            node.add_child_node(left_child)
        # Right child has values greater than median
        right_child_df = df[df[attribute] > median]
        right_child = grow_tree_one_hot(right_child_df, depth, attribute, "right", current_level + 1)
        if right_child is not None:
            node.add_child_node(right_child)
    return node       

def predict_one_hot(df, dec_tree):
    df["predicted"] = df.apply(lambda row: predict_label_one_hot(row, dec_tree), axis=1)
    return df

def predict_label_one_hot(row, current_node):
    if current_node.attribute == "" or current_node.attribute not in row.index:
        return current_node.label

    attribute = current_node.attribute

    if attribute in one_hot_categorical:
        attr_val = row[attribute]
        for child in current_node.child_list:
            if child.parent_attribute == attribute and child.parent_val == attr_val:
                return predict_label_one_hot(row, child)

    if attribute in continuous:
        attr_val = row[attribute]
        median = current_node.median
        if attr_val <= median:
            for child in current_node.child_list:
                if child.parent_val == "left":
                    return predict_label_one_hot(row, child)
        else:
            for child in current_node.child_list:
                if child.parent_val == "right":
                    return predict_label_one_hot(row, child)

        return current_node.label
    
def grow_tree_one_hot_fully(df, parent_attribute="", parent_val=-1, current_level=1):
    df_label_0 = df[df[label] == label_0]
    df_label_1 = df[df[label] == label_1]
    if len(df_label_0) == 0 or len(df_label_1) == 0:
        if len(df_label_0) == 0:
            label_val = label_1
        else:
            label_val = label_0
        return tree.Tree(attribute="", parent_attribute=parent_attribute, parent_val=parent_val,
                         median=-sys.maxsize, cat_val_list=[], child_list=[], label=label_val, level=current_level)

    attribute, median = choose_attribute_one_hot(df)

    node = tree.Tree(attribute=attribute, parent_attribute=parent_attribute, parent_val=parent_val,
                     median=None, cat_val_list=[], child_list=[], label=-1, level=current_level)
    node.label = label_0 if len(df_label_0) >= len(df_label_1) else label_1

    if attribute in one_hot_categorical:
        cat_val_list = df[attribute].unique()
        node.cat_val_list = cat_val_list
        for val in cat_val_list:
            child_df = df[df[attribute] == val]
            child_node = grow_tree_one_hot_fully(child_df, attribute, val, current_level + 1)
            if child_node is not None:
                node.add_child_node(child_node)
    elif attribute in continuous:
        node.median = median
        left_child_df = df[df[attribute] <= median]
        right_child_df = df[df[attribute] > median]
        left_child = grow_tree_one_hot_fully(left_child_df, attribute, "left", current_level + 1)
        right_child = grow_tree_one_hot_fully(right_child_df, attribute, "right", current_level + 1)
        if left_child is not None:
            node.add_child_node(left_child)
        if right_child is not None:
            node.add_child_node(right_child)
    return node

def prune_tree_nodes(df_valid, dec_tree, node, accuracy_max, depth_nodes, node_set, df_train, df_test, train_accuracy_list, valid_accuracy_list, test_accuracy_list, nodes_num):
    
    for depth in range(len(depth_nodes), 1, -1):
        nodes = depth_nodes[depth]
        # print(depth, len(nodes))
        for node in nodes:
            attr_org = node.attribute
            node.attribute = "" # make the child node a leaf, i.e, being pruned
            df_valid = predict_one_hot(df_valid, dec_tree)
            new_accuracy = accuracy(df_valid)
            # print(depth, new_accuracy, accuracy_max)
            if new_accuracy >= accuracy_max:
                accuracy_max = new_accuracy
                nodes.remove(node)
                node_set.remove(node)
                df_train = predict_one_hot(df_train, dec_tree)
                df_valid = predict_one_hot(df_valid, dec_tree)
                df_test = predict_one_hot(df_test, dec_tree)
                train_accuracy = accuracy(df_train)
                train_accuracy_list.append(train_accuracy)
                valid_accuracy = accuracy(df_valid)
                valid_accuracy_list.append(valid_accuracy)
                test_accuracy = accuracy(df_test)
                test_accuracy_list.append(test_accuracy)
                nodes_num.append(len(node_set))
            else:
                node.attribute = attr_org
    return accuracy_max, dec_tree, df_valid, node_set, train_accuracy_list, valid_accuracy, test_accuracy, nodes_num



# Greedy because as soon as we see a decrease in accuracy we break out completely
def prune_tree(df_valid, dec_tree, depth_nodes, node_set, df_train, df_test):
    
    passes = 0
    accuracy_max_by_pass = []
    train_accuracy_list = []
    test_accuracy_list = []
    valid_accuracy_list = []
    nodes_num = []
    prune_accuracy = 0
    df_train = predict_one_hot(df_train, dec_tree)
    df_valid = predict_one_hot(df_valid, dec_tree)
    df_test = predict_one_hot(df_test, dec_tree)
    train_accuracy = accuracy(df_train)
    train_accuracy_list.append(train_accuracy)
    valid_accuracy = accuracy(df_valid)
    valid_accuracy_list.append(valid_accuracy)
    test_accuracy = accuracy(df_test)
    test_accuracy_list.append(test_accuracy)
    nodes_num.append(len(node_set))
    print("Len: ", len(node_set))
    while True:
        passes = passes + 1
        print("Pass: ", passes)
        accuracy_max =  accuracy(df_valid) # before pruning decision tree
        prune_accuracy, dec_tree, df_valid, node_set, train_accuracy_list, valid_accuracy, test_accuracy, nodes_num = prune_tree_nodes(df_valid, dec_tree, dec_tree, accuracy_max, depth_nodes, node_set, df_train, df_test, train_accuracy_list, valid_accuracy_list, test_accuracy_list, nodes_num) # after pruning tree
        print(prune_accuracy, accuracy_max)
        if prune_accuracy > accuracy_max:
            accuracy_max = prune_accuracy
            # df_train = predict_one_hot(df_train, dec_tree)
            # df_valid = predict_one_hot(df_valid, dec_tree)
            # df_test = predict_one_hot(df_test, dec_tree)
            # train_accuracy = accuracy(df_train)
            # train_accuracy_list.append(train_accuracy)
            # valid_accuracy = accuracy(df_valid)
            # valid_accuracy_list.append(valid_accuracy)
            # test_accuracy = accuracy(df_test)
            # test_accuracy_list.append(test_accuracy)
            # nodes_num.append(len(node_set))
        else:
            break
        if passes > 1 and abs(accuracy_max_by_pass[-1] - accuracy_max) < 5e-3:
            break
        else:
            accuracy_max_by_pass.append(accuracy_max)
    return dec_tree, train_accuracy_list, valid_accuracy_list, test_accuracy_list, nodes_num

def dfs(node, depth, depth_nodes, node_set):
    if depth not in depth_nodes:
        depth_nodes[depth] = []
    depth_nodes[depth].append(node)
    node_set.add(node)
    for child in node.child_list:
        dfs(child, depth + 1, depth_nodes, node_set)
    return depth_nodes, node_set

# part d

def update_remaining_one_hot_cols(df, one_hot_categorical):
    for attribute in one_hot_categorical:
        if attribute not in df.columns:
            df[attribute] = 0

def part_d(df_trn, df_val, df_tst, output_folder):

    df_train = df_trn.copy(deep=True)
    df_valid = df_val.copy(deep=True)
    df_test = df_tst.copy(deep=True)
    separate_attributes(df_train)
    global continuous, categorical
    one_hot_categorical = one_hot_encoding(df_train, categorical)
    one_hot_encoding(df_valid, categorical)
    one_hot_encoding(df_test, categorical)

    # Remove categorical string type values
    df_train = df_train.drop(columns=categorical)
    df_valid = df_valid.drop(columns=categorical)
    df_test  = df_test.drop(columns=categorical)
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
    plt.title("Decision Tree Accuracy vs Max Depth (part d)")
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
    print(ccp_max)
    print("Train accuracy for best params: ", accuracy_score(y_train, y_train_predicted))
    print("Validation accuracy for best params: ", accuracy_score(y_valid, y_valid_predicted))
    print("Test accuracy for best params: ",accuracy_score(y_test, y_test_predicted))
    
    df_predicted = pd.DataFrame(y_test_predicted, columns=['Predicted'])
    df_predicted.to_csv(f"{output_folder}/predicted_d.csv", index=False)

def part_e(df_trn, df_val, df_tst, output_folder):

    df_train = df_trn.copy(deep=True)
    df_valid = df_val.copy(deep=True)
    df_test = df_tst.copy(deep=True)
    separate_attributes(df_train)
    global continuous, categorical
    one_hot_categorical = one_hot_encoding(df_train, categorical)
    one_hot_encoding(df_valid, categorical)
    one_hot_encoding(df_test, categorical)

    # Remove categorical string type values
    df_train = df_train.drop(columns=categorical)
    df_valid = df_valid.drop(columns=categorical)
    df_test  = df_test.drop(columns=categorical)

    # Set absent one_hot_train columns to 0 in df_valid and df_test
    update_remaining_one_hot_cols(df_valid, one_hot_categorical)
    update_remaining_one_hot_cols(df_test, one_hot_categorical)

    # Generating sklearn format for training
    sklearn_attribute = [attribute for attribute in df_train.columns if attribute != label]
    X_train = df_train[sklearn_attribute]
    X_valid = df_valid[sklearn_attribute]
    X_test = df_test[sklearn_attribute]
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

    df_predicted = pd.DataFrame(y_test_predicted, columns=['Predicted'])
    df_predicted.to_csv(f"{output_folder}/predicted_d.csv", index=False)


def main():
    
    global categorical
    path_train = sys.argv[1]
    path_valid = sys.argv[2]
    path_test  = sys.argv[3]
    output_folder = sys.argv[4]
    q_part = sys.argv[5].lower()

    df_train = pd.read_csv(path_train)
    df_valid = pd.read_csv(path_valid)
    df_test  = pd.read_csv(path_test)
    separate_attributes(df_train)

    if q_part == "a":

        depths = [1, 2, 3, 4, 5, 10, 15, 20]
        train_accuracy_list = []
        test_accuracy_list = []

        for depth in depths:
            flag = False
            if depth == 20:
                flag = True
            decision_tree = grow_tree(df=df_train, depth=depth)
            df_train = predict(df_train, decision_tree)
            df_valid = predict(df_valid, decision_tree)
            df_test = predict(df_test, decision_tree)
            train_accuracy = accuracy(df_train)
            test_accuracy = accuracy(df_test)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            print("Depth: {0}, train accuracy: {1}, test accuracy: {2}".format(depth, train_accuracy, test_accuracy))


        # Accuracy v/s Max Depth
        plt.figure()
        plt.plot(depths, train_accuracy_list, label="Train Accuracy", marker='o')
        plt.plot(depths, test_accuracy_list, label="Test Accuracy", marker='o')
        plt.xlabel("Max Depth")
        plt.ylabel("Accuracy")
        plt.title("Decision Tree (part a): Accuracy vs Max Depth")
        plt.legend()
        plt.grid(True)
        plt.show()    

    elif q_part == "b":
        one_hot_encoding(df_train, categorical)
        one_hot_encoding(df_valid, categorical)
        one_hot_encoding(df_test, categorical)

        depths = [5, 10, 15, 20, 25, 35, 45, 55]
        train_accuracy_list = []
        test_accuracy_list = []

        for depth in depths:
            flag = False
            if depth == 55:
                flag = True
            decision_tree = grow_tree_one_hot(df=df_train, depth=depth)
            df_train = predict_one_hot(df_train, decision_tree)
            df_valid = predict_one_hot(df_valid, decision_tree)
            df_test = predict_one_hot(df_test, decision_tree)
            train_accuracy = accuracy(df_train)
            test_accuracy = accuracy(df_test)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            print("Depth: {0}, train accuracy: {1}, test accuracy: {2}".format(depth, train_accuracy, test_accuracy))
            if flag:
                df_test["Predicted"].to_csv(f"{output_folder}/predicted_b.csv", index=False)

        # Accuracy v/s Max Depth
        plt.figure()
        plt.plot(depths, train_accuracy_list, label="Train Accuracy", marker='o')
        plt.plot(depths, test_accuracy_list, label="Test Accuracy", marker='o')
        plt.xlabel("Max Depth")
        plt.ylabel("Accuracy")
        plt.title("Decision Tree (part b): Accuracy vs Max Depth")
        plt.legend()
        plt.grid(True)
        plt.show() 

    elif q_part == "c":
        one_hot_encoding(df_train, categorical)
        one_hot_encoding(df_valid, categorical)
        one_hot_encoding(df_test, categorical)

        depths = [5, 10, 15, 20, 25, 35, 45, 55]

        for depth in depths:
            flag = False
            if depth == 55:
                flag = True
            print("Depth: {0}".format(depth))
            decision_tree = grow_tree_one_hot(df=df_train, depth=depth)
            depth_nodes, node_set = dfs(decision_tree, 1, depth_nodes={}, node_set=set())
            decision_tree, train_accuracy_list, valid_accuracy_list, test_accuracy_list,  node_num = prune_tree(df_valid, decision_tree, depth_nodes, node_set, df_train, df_test)       
            df_train = predict_one_hot(df_train, decision_tree)
            df_valid = predict_one_hot(df_valid, decision_tree)
            df_test = predict_one_hot(df_train, decision_tree)
            train_accuracy = accuracy(df_train)
            valid_accuracy = accuracy(df_valid)
            test_accuracy = accuracy(df_test)
            print("Train accuracy: ", train_accuracy)
            print("Valid accuracy: ", valid_accuracy)
            print("Test accuracy: ", test_accuracy)
            # Accuracy v/s Number of nodes
            plt.figure()
            plt.plot(node_num, train_accuracy_list, label="Train Accuracy", marker='o')
            plt.plot(node_num, valid_accuracy_list, label="Validation Accuracy", marker='o')
            plt.plot(node_num, test_accuracy_list, label="Test Accuracy", marker='o')
            plt.xlabel("Number of nodes in tree")
            plt.ylabel("Accuracy")
            plt.title("Decision Tree (part c): Accuracy vs Nodes for depth: {0}".format(depth))
            plt.legend()
            plt.grid(True)
            plt.show()
            if flag:
                df_test["Predicted"].to_csv(f"{output_folder}/predicted_c.csv", index=False)
    elif q_part == "d":
        part_d(df_trn=df_train, df_val=df_valid, df_tst=df_test, output_folder = output_folder)
    elif q_part == "e":
        part_e(df_trn=df_train, df_val=df_valid, df_tst=df_test, output_folder = output_folder)        

if __name__ == "__main__":
    main()

# depth_nodes, node_set = dfs(decision_tree, 1, depth_nodes={}, node_set=set())

# print(len(depth_nodes))

# print("Pruning tree...")
# accuracy_max = accuracy(df_valid)
# t1 = time.time()
# decision_tree, train_accuracy_list, valid_accuracy_list, test_accuracy_list, nodes_num = prune_tree(df_valid, decision_tree, depth_nodes, node_set, df_train, df_test)
# t2 = time.time()
# print("Tree pruned in {}".format(t2 - t1))
# df_train = predict_one_hot(df_train, decision_tree)
# df_valid = predict_one_hot(df_valid, decision_tree)
# df_test = predict_one_hot(df_test, decision_tree)
# train_accuracy = accuracy(df_train)
# valid_accuracy = accuracy(df_valid)
# test_accuracy = accuracy(df_test)
# print("Train accuracy: ", train_accuracy)
# print("Valid acuracy: ", valid_accuracy)
# print("Test accuracy: ", test_accuracy)

# # Accuracy v/s Nodes in decision tree
# plt.figure()
# plt.plot(nodes_num, train_accuracy_list, label="Train Accuracy", marker='o')
# plt.plot(nodes_num, valid_accuracy_list, label="Validation Accuracy", marker='o')
# plt.plot(nodes_num, test_accuracy_list, label="Test Accuracy", marker='o')
# plt.xlabel("Nodes")
# plt.ylabel("Accuracy")
# plt.title("Decision Tree Accuracy vs Nodes")
# plt.legend()
# plt.grid(True)
# plt.show()

# one_hot_categorical = one_hot_encoding(df_train, categorical)
# one_hot_encoding(df_valid, categorical)
# one_hot_encoding(df_test, categorical)

# df_train.to_csv("one_hot_train.csv", index=False)

# print(one_hot_categorical)

# for attribute in one_hot_categorical:
#     df_attr = df_train[attribute]
#     print(df_attr[df_attr == 1])


# decision_tree = grow_tree_one_hot(df_train, depth = 25)  

# print("Training full decision tree...")
# t1 = time.time()
# # decision_tree = grow_tree_one_hot(df_train, depth=25) 
# decision_tree = grow_tree_one_hot_fully(df_train)         
# t2 = time.time()
# print("Training time: ", t2 - t1)

# print("Predicting...")
# df_train = predict_one_hot(df_train, decision_tree)
# df_valid = predict_one_hot(df_valid, decision_tree)
# df_test = predict_one_hot(df_test, decision_tree)

# accuracy(df_train)
# accuracy(df_valid)
# accuracy(df_test)