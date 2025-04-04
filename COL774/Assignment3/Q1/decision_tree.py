import helper
import pandas as pd
import tree
import math

path_train = "../data/Q1/train.csv"
path_valid = "../data/Q1/valid.csv"
path_test = "../data/Q1/test.csv"

df_train = helper.read(path_train)
df_valid = helper.read(path_valid)
df_test = helper.read(path_test)

label = "income"
label_0 = " <=50K"
label_1 = " >50K"

def separate_attributes(df):
    continuous = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return continuous, categorical

continuous, categorical = separate_attributes(df_train)
categorical.remove(label)

def entropy(df):
    # Entropy of root attribute
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
    
    H_y = entropy(df)
    I_gain_max = 0
    split_attr = ""
    count = len(df)

    for attribute in categorical:
        attr_val = df[attribute].unique()
        H_y_val = 0
        for val in attr_val:
            df_attr = df[df[attribute] == val]
            prob_attr = float(len(df_attr))/count
            H_y_val += prob_attr*entropy(df_attr)        
        I_gain = H_y - H_y_val
        print(attribute, H_y, H_y_val, I_gain)
        if I_gain > I_gain_max:
            split_attr = attribute
            I_gain_max = I_gain

    for attribute in continuous:
        
        H_y_val = 0
        median = df[attribute].median()
        
        df_attr = df[df[attribute] <= median]
        prob_attr = float(len(df_attr))/count
        H_y_val += prob_attr*entropy(df_attr)     
        
        df_attr = df[df[attribute] > median]
        prob_attr = float(len(df_attr))/count
        H_y_val += prob_attr*entropy(df_attr) 
        
        I_gain = H_y - H_y_val
        print(attribute, H_y, H_y_val, I_gain)
        if I_gain > I_gain_max:
            split_attr = attribute
            I_gain_max = I_gain

    return split_attr

attribute = choose_attribute(df_train)
print(attribute)

def grow_tree(df, depth):
    #base condition
    if depth == 0:
        return
    
    # find attribute with maximum information gain
    attribute = choose_attribute(df)

    # split data
    