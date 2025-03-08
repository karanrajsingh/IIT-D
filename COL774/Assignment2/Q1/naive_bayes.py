import numpy as np
import pandas as pd
import string
import re

class NaiveBayes:

    df = ""

    def __init__(self):
        
        #Initial test for test.csv (as dataset is smaller)
        path_test = '../data/Q1/test.csv'
        #df = self.preprocessing(path_test)
        self.read(path_test)
        self.tokenize(text_col="Description")
        self.print()

    def print(self):
        global df
        feature_column = "Description"
        for i in range(len(feature_column)):
            print("\n")
            print(df.iloc[i])
            print("\n")


    def read(self, path):
        global df
        df = pd.read_csv(path)
        print('----Successful Read----')

    # Clean a dataframe row of punctuations
    def tokenize(self, text_col):
        global df
        feature_column = text_col
        for i in range(len(feature_column)):
            s = df[feature_column].iloc[i]
            df.at[i, feature_column] = self.tokenize_text(s)


    # Cleans the text from punctuations and returns a list of words in the text
    def tokenize_text(self, text):
        punctuation = string.punctuation
        map = str.maketrans(punctuation, ' '*len(punctuation))
        remove_punctuation = text.translate(map)
        list = remove_punctuation.split()
        return list
        #return " ".join(list)


    def fit(self, df, smoothening, class_col = "Class Index", text_col = "Tokenized Description"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """

        pass
    
    def predict(self, df, text_col = "Tokenized Description", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        pass