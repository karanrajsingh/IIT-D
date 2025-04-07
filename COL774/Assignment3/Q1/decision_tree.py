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

df_train = helper.read(path_train)
df_valid = helper.read(path_valid)
df_test = helper.read(path_test)

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
    print(f"Misses: {total - correct}, Accuracy: {acc:.4f}")
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