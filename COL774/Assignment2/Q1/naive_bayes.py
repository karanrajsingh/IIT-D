import numpy as np
import pandas as pd
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class NaiveBayes:

    df = ""

    def __init__(self):
        
        #Initial test for test.csv (as dataset is smaller)
        path_train = '../data/Q1/train.csv'
        #df = self.preprocessing(path_test)
        self.read(path_train)
        self.tokenize(text_col="Description")
        self.wordcloud(text_col="Description")
        #self.wordcloud(text_col="Description", class_index=1)
        #print(df)
    
        
    def read(self, path):
        global df
        df = pd.read_csv(path)
        print('----Successful Read----')

    # Clean a dataframe row of punctuations
    def tokenize(self, text_col):
        global df
        feature_column = text_col
        df[feature_column] = df[feature_column].apply(self.tokenize_text)


    # Cleans the text from punctuations and returns a list of words in the text
    def tokenize_text(self, text):
        punctuation = string.punctuation
        map = str.maketrans(punctuation, ' '*len(punctuation))
        remove_punctuation = text.translate(map)
        list = remove_punctuation.split()
        return list
        #return " ".join(list)

    
    def wordcloud(self, text_col):
        global df
        max = df["Class Index"].max(axis=0)
        for i in range(max):
            text = self.build_text(text_col, i+1)
            class_wc = WordCloud(background_color='white', max_words=20000)
            class_wc.generate(text)
            plt.imshow(class_wc, interpolation='bilinear')
            plt.axis('off')
            plt.show()
    
    def build_text(self, text_col, class_index):
        selected_df = self.rows_by_class(class_index)
        selected_df[text_col] = selected_df[text_col].apply(" ".join)
        text = ""
        print(len(selected_df[text_col]))
        for i in range(len(selected_df[text_col])):
            text += " " + selected_df[text_col].iloc[i]
        return text
        
    def rows_by_class(self, class_index):
        global df
        selected_df = df[df["Class Index"] == class_index]
        return selected_df


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