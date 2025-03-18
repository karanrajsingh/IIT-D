import pandas as pd
import naive_bayes
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import itertools
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import bigrams
import string

model = naive_bayes.NaiveBayes()

def read(path):
    df= pd.read_csv(path)
    return df

# Cleans a dataframe column from punctuations, and returns the updated dataframe
def clean_punctuation(df, text_col):
    df[text_col] = df[text_col].apply(clean_punctuation_from_cell)
    return df


# Cleans any text string from punctuations and returns the updated string
def clean_punctuation_from_cell(text):
    punctuation = string.punctuation
    # punctuation = string.punctuation + "1234567890"
    map = str.maketrans(punctuation, ' '*len(punctuation))
    remove_punctuation = text.translate(map)
    return remove_punctuation

# Returns a list of tokens for each cell in the dataframe column
def tokenize_col(df, text_col):
    df["Tokenized" + " " + text_col] = df[text_col].apply(tokenize)
    return df

def tokenize(text):
    l = text.split()
    word_list = [word.lower() for word in l]
    return word_list

def merge_col(df, text_col1, text_col2):
    df[text_col1 + " " + text_col2] = df[text_col1] + df[text_col2]
    return df


def wordcloud(df, text_col = "Tokenized Description"):
    max = df["Class Index"].max(axis=0)
    for i in range(max):
        class_index = i+1
        merged_words = model.merge_documents(df, "Class Index", text_col, class_index)[0]
        class_wc = WordCloud(background_color='white', max_words=20000)
        class_wc.generate(" ".join(merged_words))
        plt.imshow(class_wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()


def remove_stopwords(df, text_col = "Tokenized Description"):
    df[text_col + " No Stopwords"] = df[text_col].apply(remove_stopwords_from_text)
    return df

def remove_stopwords_from_text(text):
    stop_words = set(stopwords.words('english'))
    clean_stopwords = [word for word in text if not word in stop_words]
    return clean_stopwords

def stem(df,  text_col):
    df[text_col + " Stemmed"] = df[text_col].apply(stemming_the_text)
    return df        

def stemming_the_text(text):
    ps = PorterStemmer()
    stemmed_text = [ps.stem(word) for word in text]
    return stemmed_text

def generate_bigrams(df, text_col = "Tokenized Description"):
    df["Bigrams " + text_col] = df[text_col].apply(generate_bigrams_for_text)
    return df

def generate_bigrams_for_text(text):
    # Assuming that text already tokenized
    return list(bigrams(text))

def generate_trigrams(df, text_col = "Tokenized Description"):
    df["Trigrams " + text_col] = df[text_col].apply(generate_trigrams_for_text)
    return df

def generate_trigrams_for_text(text):
    # Assuming that text already tokenized
    return list(bigrams(text))

def combine_unigram_bigram(df, text_col1, text_col2):
    df[text_col1 + " " + text_col2] = df.apply(lambda row: combine_unigram_bigram_for_text(row[text_col1], row[text_col2]), axis = 1)
    return df

def combine_unigram_bigram_for_text(tokens1, tokens2):
        merged_list = list(itertools.chain(tokens1, tokens2))
        return merged_list

def combine_unigram_trigram(df, text_col1, text_col2):
    df[text_col1 + " " + text_col2] = df.apply(lambda row: combine_unigram_trigram_for_text(row[text_col1], row[text_col2]), axis = 1)
    return df

def combine_unigram_trigram_for_text(tokens1, tokens2):
        merged_list = list(itertools.chain(tokens1, tokens2))
        return merged_list