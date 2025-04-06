import helper
import pandas as pd
import tree
import math
import sys
import random
from collections import deque

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
    if(prob_0 == 0):
        log_0 = 0
    else:
        log_0 = prob_0*math.log(prob_0)
    if(prob_1 == 0):
        log_1 = 1
    else:
        log_1 = prob_1*math.log(prob_1)
    entropy = -(log_0 + log_1)
    return entropy

def choose_attribute(df):
    # H(Y)
    H_y = entropy(df)
    I_gain_max = 0
    split_attr = ""
    median = -sys.maxsize
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

    return split_attr, median

attribute = choose_attribute(df_train)


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
                value = random.randint(0, 10)%2
                if(value == 0):
                    label_val = label_0
                else:
                    label_val = label_1
        elif len(df_label_0) == 0:
            label_val = label_0
        else:
            label_val = label_1
        return tree.Tree(attribute = "", parent_attribute = parent_attribute, parent_val = parent_val, median = -sys.maxsize, cat_val_list = [], child_list = [], label = label_val, level = current_level)

    attribute, median = choose_attribute(df)

    # Creating Decision Tree node
    node = tree.Tree(attribute = attribute, parent_attribute = parent_attribute, parent_val = parent_val, median = None, cat_val_list = [], child_list = [], label = -1, level = current_level)

    if attribute in categorical:
       
        cat_val_list = df[attribute].unique()
        node.cat_val_list = cat_val_list
        for val in cat_val_list:
            child_df = df[df[attribute] == val]
            child_node = grow_tree(child_df, depth, attribute, val, current_level + 1)
            if child_node is not None:
                node.add_child_node(child_node)
    else:
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

decision_tree = grow_tree(df_train, depth=5)

def print_node(node):
    print(node.level, node.attribute)

def bfs_traversal(root):
    if not root:
        return []

    queue = deque([root])
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)
        print("Attribute: {0}, Parent attribute: {4}, Parent Val: {3}, Level: {1}, Label: {2}".format(current.attribute, current.level, current.label, current.parent_val, current.parent_attribute))
        for child in current.child_list:
            queue.append(child)

    return result

bfs_traversal(decision_tree)

