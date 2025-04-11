import helper
import pandas as pd
import tree
import math
import sys
import random
from collections import deque
import time

path_train = "../data/Q1/train.csv"
path_valid = "../data/Q1/valid.csv"
path_test = "../data/Q1/test.csv"

df_train = pd.read_csv(path_train)
df_valid = pd.read_csv(path_valid)
df_test = pd.read_csv(path_test)

label = "income"
label_0 = " <=50K"
label_1 = " >50K"

# returns a list of continuous and categorical attributes separately
def separate_attributes(df):
    continuous = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return continuous, categorical

continuous, categorical = separate_attributes(df_train)
categorical.remove(label)

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

# decision_tree = grow_tree(df_train, depth = 5)         

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
# df_train = predict(df_train, decision_tree)
# df_valid = predict(df_valid, decision_tree)
# df_test = predict(df_test, decision_tree)

def accuracy(df, label_col= label):
    correct = (df[label_col].astype(str).str.strip().str.lower() == 
               df["predicted"].astype(str).str.strip().str.lower()).sum()
    total = len(df)
    acc = correct / total
    # print(f"Misses: {total - correct}, Accuracy: {acc:.4f}")
    return acc
# accuracy(df_train)
# accuracy(df_valid)
# accuracy(df_test)

def one_hot_encoding(df, categorical):
    one_hot_categorical = []
    for attribute in categorical:
        attr_val = df[attribute].unique()
        for val in attr_val:
            df[attribute + "_" + str(val)] = df[attribute].apply(lambda x: 1 if x == val else 0)
            # str = attribute + "_" + val
            one_hot_categorical.append(attribute + "_" + str(val))
    return one_hot_categorical

one_hot_categorical = one_hot_encoding(df_train, categorical)
one_hot_encoding(df_valid, categorical)
one_hot_encoding(df_test, categorical)

df_train.to_csv("one_hot_train.csv", index=False)

# print(one_hot_categorical)

# for attribute in one_hot_categorical:
#     df_attr = df_train[attribute]
#     print(df_attr[df_attr == 1])

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

# decision_tree = grow_tree_one_hot(df_train, depth = 25)         

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
# df_train = predict_one_hot(df_train, decision_tree)
# df_valid = predict_one_hot(df_valid, decision_tree)
# df_test = predict_one_hot(df_test, decision_tree)
# accuracy(df_train)
# accuracy(df_valid)
# accuracy(df_test)

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

print("Training full decision tree...")
t1 = time.time()
# decision_tree = grow_tree_one_hot(df_train, depth=25) 
decision_tree = grow_tree_one_hot_fully(df_train)         
t2 = time.time()
print("Training time: ", t2 - t1)

print("Predicting...")
df_train = predict_one_hot(df_train, decision_tree)
df_valid = predict_one_hot(df_valid, decision_tree)
df_test = predict_one_hot(df_test, decision_tree)

accuracy(df_train)
accuracy(df_valid)
accuracy(df_test)

# def prune_tree_nodes(df_valid, dec_tree, node, best_acc):
#     # Base case: if node is already a leaf, nothing to prune.
#     if node.attribute == "":
#         return best_acc, dec_tree

#     # First, try pruning the children (post-order traversal).
#     for child in node.child_list:
#         # Recursively prune each child.
#         best_acc, dec_tree = prune_tree_nodes(df_valid, dec_tree, child, best_acc)

#     # Now, attempt to prune the current node by turning it into a leaf.
#     original_attr = node.attribute  # backup current attribute
#     node.attribute = ""               # prune this node (make it a leaf)
    
#     # Evaluate accuracy using a copy of df_valid.
#     df_valid_pred = predict_one_hot(df_valid.copy(), dec_tree)
#     new_acc = accuracy(df_valid_pred)

#     # If pruning improves or maintains accuracy, update best_acc.
#     if new_acc >= best_acc:
#         best_acc = new_acc
#         # (Keep the pruned node: attribute remains "")
#     else:
#         # Otherwise, revert the pruning.
#         node.attribute = original_attr

#     return best_acc, dec_tree

# def prune_tree(df_valid, dec_tree):
#     # Evaluate the initial accuracy on the validation set.
#     current_acc = accuracy(predict_one_hot(df_valid.copy(), dec_tree))
    
#     # Continue pruning until no further improvements are found.
#     passes = 0
#     while True:
#         passes = passes + 1
#         new_acc, pruned_tree = prune_tree_nodes(df_valid, dec_tree, dec_tree, current_acc)
#         print("Passes: ", passes, new_acc, current_acc)
#         if new_acc > current_acc and abs(new_acc - current_acc) < 5e-3:
#             current_acc = new_acc
#             dec_tree = pruned_tree
#         else:
#             break
#     return dec_tree

# def dfs(node, depth, depth_nodes):
#     if depth not in depth_nodes:
#         depth_nodes[depth] = []
#     depth_nodes[depth].append(node)
#     for child in node.child_list:
#         dfs(child, depth + 1, depth_nodes)
#     return depth_nodes

# depth_nodes = dfs(decision_tree, 1, depth_nodes={})

# print(len(depth_nodes))

def prune_tree_one_hot(tree_root, df_valid):
    """
    Prune a decision tree trained on one-hot features using post-pruning.
    """
    base_acc = accuracy(predict_one_hot(df_valid.copy(), tree_root))
    changed = True
    passes = 0
    while changed:
        passes += 1
        print("Passes: ", passes)
        changed, base_acc = prune_tree_helper_one_hot(tree_root, df_valid, base_acc, tree_root)
    return tree_root

def prune_tree_helper_one_hot(node, df_valid, current_acc, tree_root):
    pruned = False
    # Post-order traversal: prune children first
    for child in node.child_list:
        child_pruned, current_acc = prune_tree_helper_one_hot(child, df_valid, current_acc, tree_root)
        if child_pruned:
            pruned = True

    if node.child_list:
        # Backup node state
        original_attribute = node.attribute
        original_children = node.child_list
        original_label = node.label

        # Prune: convert this node to leaf
        node.attribute = ""
        node.child_list = []

        # Evaluate accuracy with pruned tree
        new_acc = accuracy(predict_one_hot(df_valid.copy(), tree_root))

        if new_acc >= current_acc:
            current_acc = new_acc
            pruned = True
        else:
            # Restore original state
            node.attribute = original_attribute
            node.child_list = original_children
            node.label = original_label

    return pruned, current_acc


print("Pruning tree...")
accuracy_max = accuracy(df_valid)
t1 = time.time()
decision_tree = prune_tree_one_hot(decision_tree, df_valid)
t2 = time.time()
print("Tree pruned in {}".format(t2 - t1))
df_train = predict_one_hot(df_train, decision_tree)
df_valid = predict_one_hot(df_valid, decision_tree)
df_test = predict_one_hot(df_test, decision_tree)
train_accuracy = accuracy(df_train)
valid_accuracy = accuracy(df_valid)
test_accuracy = accuracy(df_test)
print("Train accuracy: ", train_accuracy)
print("Valid acuracy: ", valid_accuracy)
print("Test accuracy: ", test_accuracy)

