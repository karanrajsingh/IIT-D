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
    # print(count_0, count_1)
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
    # print("Choose attribute: ", len(df))
    H_y = entropy(df)
    I_gain_max = 0
    split_attr = ""
    median = -sys.maxsize
    count = len(df)

    for attribute in categorical:
        # print("Cat")
        attr_val = df[attribute].unique()
        H_y_val = 0
        for val in attr_val:
            df_attr = df[df[attribute] == val]
            # print(len(df_attr), val)
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr)        
        I_gain = H_y - H_y_val
        # print(attribute, H_y, H_y_val, I_gain)
        if I_gain > I_gain_max:
            split_attr = attribute
            I_gain_max = I_gain

    for attribute in continuous:
        # print("Cnt")
        H_y_val = 0
        median = df[attribute].median()
        
        df_attr = df[df[attribute] <= median]
        # print("Len: ", len(df), len(df_attr))
        if len(df_attr) != 0:
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr)     
        
        df_attr = df[df[attribute] > median]
        if len(df_attr) != 0:
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr) 
        
        I_gain = H_y - H_y_val
        # print(attribute, H_y, H_y_val, I_gain)
        if I_gain > I_gain_max:
            split_attr = attribute
            I_gain_max = I_gain

    return split_attr, median

attribute = choose_attribute(df_train)


# decision_tree = tree.Tree(attribute=None, cat_val=None, cont_dir=None level= 1)

# def grow_tree(df, dec_tree, depth):
    
#     #base condition
#     df_label_0 = df[df[label] == label_0]
#     df_label_1 = df[df[label] == label_1]
#     if len(df_label_0) == 0 or len(df_label_1) == 0 or dec_tree.level >= depth:
#         return
        
    
#     # find attribute with maximum information gain
#     # print(len(df))
#     attribute, median = choose_attribute(df)

#     # split data
#     if attribute in categorical:
#         # print("categorical")
#         dec_tree.attribute = attribute
#         cat_val_list = df[attribute].unique()
#         dec_tree.cat_val_list = cat_val_list
#         for val in cat_val_list:     
#             node = tree.Tree(attribute = "", median = None, cat_val_list = [], child_list = [], level = dec_tree.level + 1)
#             grow_tree(df[df[attribute] == val], node, depth)
#             dec_tree.add_child_node(node)
            
#     else:
#         # print("continuous")
#         dec_tree.attribute = attribute
#         dec_tree.median = median
#         node = tree.Tree(attribute = "", median = None, cat_val_list = [], child_list = [], level = dec_tree.level + 1)
#         grow_tree(df[df[attribute] <= median], node, depth)
#         dec_tree.add_child_node(node)
#         node = tree.Tree(attribute = "", median = None, cat_val_list = [], child_list = [], level = dec_tree.level + 1)
#         grow_tree(df[df[attribute] > median], node, depth)
#         dec_tree.add_child_node(node)


# decision_tree = tree.Tree(attribute="", median=None, cat_val_list=[], child_list=[], level=1)
# grow_tree(df_train, decision_tree, 5)

def grow_tree(df, depth, parent_attribute = "", parent_val = -1, current_level=1):

    df_label_0 = df[df[label] == label_0]
    df_label_1 = df[df[label] == label_1]
    if len(df_label_0) == 0 or len(df_label_1) == 0 or current_level >= depth:
        if current_level >= depth:
            if len(df_label_0) > len(df_label_1):
                label_val = 0
            elif len(df_label_0) < len(df_label_1):
                label_val = 1
            else:
                label_val = random.randint(0, 10)%2
        elif len(df_label_0) == 0:
            label_val = 0
        else:
            label_val = 1
        return tree.Tree(attribute = "", parent_attribute = parent_attribute, parent_val = parent_val, median = -sys.maxsize, cat_val_list = [], child_list = [], label = label_val, level = current_level)

    # Find the best attribute to split on (and median if the attribute is continuous)
    attribute, median = choose_attribute(df)

    # Create the current node
    node = tree.Tree(attribute = attribute, parent_attribute = parent_attribute, parent_val = parent_val, median = None, cat_val_list = [], child_list = [], label = -1, level = current_level)

    if attribute in categorical:
        # For categorical attributes, get unique values and split accordingly.
        cat_val_list = df[attribute].unique()
        node.cat_val_list = cat_val_list  # Store the possible categories in this node.
        for val in cat_val_list:
            child_df = df[df[attribute] == val]
            child_node = grow_tree(child_df, depth, attribute, val, current_level + 1)
            if child_node is not None:
                node.add_child_node(child_node)
    else:
        # For continuous attributes, split using the median.
        node.median = median  # Assign the split value.
        # Left branch: values less than or equal to the median.
        left_child_df = df[df[attribute] <= median]
        left_child = grow_tree(left_child_df, depth, attribute, "left", current_level + 1)
        if left_child is not None:
            node.add_child_node(left_child)
        # Right branch: values greater than the median.
        right_child_df = df[df[attribute] > median]
        right_child = grow_tree(right_child_df, depth, attribute, "right", current_level + 1)
        if right_child is not None:
            node.add_child_node(right_child)
    return node

# To construct your decision tree:
decision_tree = grow_tree(df_train, depth=5)

def print_node(node):
    print(node.level, node.attribute)

# def print_tree(dec_tree):
#     # print_node(dec_tree)
#     # for child in dec_tree.child_list:
#     #     print_tree(child)


# print_tree(decision_tree)

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

# BFS traversal
# bfs_traversal(decision_tree)

