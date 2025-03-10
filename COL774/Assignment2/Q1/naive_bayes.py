import numpy as np
import pandas as pd
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class NaiveBayes:

    df_train = ""
    df_test = ""

    def __init__(self):
        

        global df_train, df_test
        #Initial test for test.csv (as dataset is smaller)
        path_train = '../data/Q1/train.csv'
        path_test = '../data/Q1/test.csv'
        #df = self.preprocessing(path_test)
        #df_train, df_test = self.read(path_train, path_test)
        #self.tokenize(text_col="Description")
        #self.wordcloud(text_col="Description")
        #self.wordcloud(text_col="Description", class_index=1)
        #print(df)
    
        
    def read(self, path):
        df= pd.read_csv(path)
        return df

    # Cleans a dataframe column from punctuations, and returns the updated dataframe
    def clean_punctuation(self, df, text_col):
        df[text_col] = df[text_col].apply(self.clean_punctuation_from_cell)
        return df


    # Cleans any text string from punctuations and returns the updated string
    def clean_punctuation_from_cell(self, text):
        punctuation = string.punctuation
        map = str.maketrans(punctuation, ' '*len(punctuation))
        remove_punctuation = text.translate(map)
        #list = remove_punctuation.split()
        #return list
        #return " ".join(list)
        return remove_punctuation
    
    # Returns a list of tokens for each cell in the dataframe column
    def tokenize_col(self, df, text_col):
        df["Tokenized" + " " + text_col] = df[text_col].apply(self.tokenize)
        return df

    def tokenize(self, text):
        return text.split()

    def merge_col(self, df, text_col1, text_col2):
        df[text_col1 + " " + text_col2] = df[text_col1] + df[text_col2]
        return df
    
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

        """
        Steps to follow:
            1. For each class in Class Index in df:
                a. Merge all tokens and store in a dictionary.
                b. Calculate probabilities (Theta parameter) for each word (token) given the class.
                c. Store the parameters from (b) in a dictionary ('token/word' : probability).
            2. Calculate phi for each class
            3. Give different variations to text_col. These would be pre-processed and fed to the 'fit' function, with steps as under:
                1. Remove punctuations. This would return each cell of text_col as list of tokens.
                2.  

        """

        word_count = self.build_word_count(df, class_col, text_col)[0]
        total_words = self.build_word_count(df, class_col, text_col)[1]

        print(word_count.keys())
        print(total_words)
        # for i in range(max):
        #     class_index = i + 1
        #     class_df = df[df["Class Index"] == class_index]
        #     temp_df = class_df[text_col].apply(" ".join).str.cat(sep= " ")
            
        #     print(temp_df)
    
    def build_word_count(self, df,  class_col = "Class Index", text_col = "Tokenized Description"):
        max = df["Class Index"].max(axis=0)
        word_count_by_class = {}
        total_words_by_class = {}
        for i in range(max):
            class_index = i+1
            word_dict_by_class = self.build_word_count_for_class(df, class_col, text_col, class_index)[0]
            total_words_by_class[class_index] = self.build_word_count_for_class(df, class_col, text_col, class_index)[1]
            word_count_by_class[class_index] =  word_dict_by_class
        return word_count_by_class, total_words_by_class

        
    def build_word_count_for_class(self, df, class_col = "Class Index", text_col = "Tokenized Description", class_index = '1'):
        merged_doc = self.merge_documents(df, class_col, text_col, class_index)[0]
        total_words = self.merge_documents(df, class_col, text_col, class_index)[1]
        word_count = self.build_word_count_for_class_helper(merged_doc, text_col)
        #print(word_count)
        return word_count, total_words
    
    def merge_documents(self, df,class_col = "Class Index", text_col = "Tokenized Description", class_index = '1'):
        class_df = df[df["Class Index"] == class_index]
        merged_doc = class_df[text_col].apply(" ".join).str.cat(sep= " ")
        return merged_doc, len(merged_doc)

    def build_word_count_for_class_helper(self, merged_doc, text_col = "Tokenized Description"):
        word_dict = {}
        merged_words = merged_doc.split()
        for word in merged_words:
            print
            word = word.lower()
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
            
        return word_dict
        

    
    def predict(self, df, text_col = "Tokenized Description", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        pass