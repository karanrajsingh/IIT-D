import helper
import pandas as pd

path_train = "../data/Q1/train.csv"
path_valid = "../data/Q1/valid.csv"
path_test = "../data/Q1/test.csv"

df_train = helper.read(path_train)
df_valid = helper.read(path_valid)
df_test = helper.read(path_test)

print(df_train.iloc[0])
print(df_valid.iloc[0])
print(df_test.iloc[0])


