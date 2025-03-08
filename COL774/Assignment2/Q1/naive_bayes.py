import numpy as np
import pandas as pd
import string
import re

class NaiveBayes:

    def __init__(self):
        
        #Initial test for test.csv (as dataset is smaller)
        path_test = '../data/Q1/test.csv'
        self.preprocessing(path_test)

    def preprocessing(self, path):
        df = pd.read_csv(path)
        print('----Successful Read----')

        feature_column = "Description"
        for i in range(len(feature_column)):
            s = df[feature_column].iloc[i]
            df.at[i, feature_column] = self.tokenizer(s)

        # Test for correct update
        '''
        for i in range(len(feature_column)):
            print("\n")
            print(df[feature_column].iloc[i])
            print("\n")

        '''


    # Cleans the text from punctuations and returns a list of words in the text
    def tokenizer(self, text):
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