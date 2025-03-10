import numpy as np
import pandas as pd
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import math
import sys

class NaiveBayes:

    df_train = ""
    df_test = ""
    mod_V = 1e+6

    priors = ""
    word_estimates = ""

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

        mod_V = 1e+6

        """

        'word_count' stores counts of each word across all documents in a class.
        After fit function completes execution, 'word_count' stores log of probability of each word given a class

        word_count:
            class_index_1: word_dict
                            {word_1: count_1 -> later updated to P(word_1|class_index_1)}
                            {word_2: count_2 -> later updated to P(word_2|class_index_1)}
                            {-----------------------------------------------------------}
                            {-----------------------------------------------------------}
                            {word_j: count_j -> later updated to P(word_j|class_index_1)}
                            {-----------------------------------------------------------}
                            {-----------------------------------------------------------}                            
                            {word_n: count_n -> later updated to P(word_n|class_index_1)}
            :
            :
            :
            class_index_4: word_dict
                            {word_1: count_1 -> later updated to P(word_1|class_index_4)}
                            {word_2: count_2 -> later updated to P(word_2|class_index_4)}
                            {-----------------------------------------------------------}
                            {-----------------------------------------------------------}
                            {word_j: count_j -> later updated to P(word_j|class_index_4)}
                            {-----------------------------------------------------------}
                            {-----------------------------------------------------------}
                            {word_n: count_n -> later updated to P(word_n|class_index_4)}
        
        total_words:
            class_index_1: total words in all documents of class 1
            class_index_2: total words in all documents of class 2
            class_index_3: total words in all documents of class 3
            class_index_4: total words in all documents of class 4

        """
        print("Estimating class wise conditional porbabilities of each word...")
        for class_index in word_count:
            word_dict = word_count[class_index]
            total_words_smoothed = self.smoothen(total_words[class_index], mod_V, smoothening)
            for word in word_dict:
                val_1 = word_dict[word]
                word_dict[word] = self.smoothen(word_dict[word], 1, smoothening)
                val_2 = word_dict[word]
                word_dict[word] = math.log(word_dict[word]) - math.log(total_words_smoothed)
        #        print(word + " : ", val_1, val_2, word_dict[word])
            word_dict[";;;"] = math.log(smoothening) - math.log(total_words_smoothed)
        print("Estimating class priors...")
        class_priors = self.class_priors(df, class_col)
        #print(class_priors)

        global priors
        global word_estimates
        priors = class_priors
        word_estimates = word_count       

        return (class_priors, word_count)

    def smoothen(self, value, offset, smoothening):
        value += offset*smoothening
        return value

    def class_priors(self, df, class_col = "Class Index"):
        max = df[class_col].max(axis=0)
        class_priors = {}
        for i in range(max):
            class_index = i+1
            class_priors[class_index] = df[class_col].value_counts().get(class_index, 0)/len(df)
        return class_priors
    
    def build_word_count(self, df,  class_col = "Class Index", text_col = "Tokenized Description"):
        max = df[class_col].max(axis=0)
        word_count_by_class = {}
        total_words_by_class = {}
        for i in range(max):
            class_index = i+1
            word_dict_by_class = self.build_word_count_for_class(df, class_col, text_col, class_index)[0]
            total_words_by_class[class_index] = self.build_word_count_for_class(df, class_col, text_col, class_index)[1]
            word_count_by_class[class_index] =  word_dict_by_class
        return word_count_by_class, total_words_by_class

        
    def build_word_count_for_class(self, df, class_col = "Class Index", text_col = "Tokenized Description", class_index = '1'):
        merged_words = self.merge_documents(df, class_col, text_col, class_index)[0]
        total_words = self.merge_documents(df, class_col, text_col, class_index)[1]
        word_count = self.build_word_count_for_class_helper(merged_words, text_col)
        #print(word_count)
        return word_count, total_words
    
    def merge_documents(self, df,class_col = "Class Index", text_col = "Tokenized Description", class_index = '1'):
        class_df = df[df[class_col] == class_index]
        merged_doc = class_df[text_col].apply(" ".join).str.cat(sep= " ")
        merged_words = merged_doc.split()
        return merged_words, len(merged_words)

    def build_word_count_for_class_helper(self, merged_words, text_col = "Tokenized Description"):
        word_dict = {}
        for word in merged_words:
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
        
        df[predicted_col] = df[text_col].apply(self.predict_text)
        # for i in range(len(df)):
        #      #print(df["Class Index"].iloc[i], df[predicted_col].iloc[i])
        #      print(df.iloc[i])
        return df

    def predict_text(self, text):
        #print(text)
        global priors
        global word_estimates
        class_probability = {}
        for class_index in priors:
            class_probability[class_index] = self.get_probability(text, class_index)
        max_prob = max(class_probability.values())
        for class_index in class_probability:
            if class_probability[class_index] == max_prob:
                predicted_class = class_index
        return predicted_class



    def get_probability(self, text, class_index = 1):
        global priors
        global word_estimates
        word_dict = word_estimates[class_index]
        probability = 0
        mod_V = 1e+6
        for word in text:
            word = word.lower()
            if word in word_dict:
                probability += word_dict[word]
            else:
                probability += word_dict[";;;"]
        probability += priors[class_index]
        return probability
    
    def accuracy(self, df):
        accuracy = 0;
        total = len(df["Class Index"])
        for i in range(len(df["Class Index"])):
            if df["Class Index"].iloc[i] == df["Predicted"].iloc[i]:
                accuracy += 1
        return float(accuracy)/total