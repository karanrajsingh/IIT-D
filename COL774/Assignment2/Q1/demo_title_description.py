import naive_bayes
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report


def get_metrics(df):
    y_test = df["Class Index"]
    y_predicted = df["Predicted"]
    # print(y_pred)
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    # Part 8
    cm = confusion_matrix(y_test, y_predicted)
    print(cm)

    print(f'Accuracy: ', accuracy)
    print(f'Precision: ', precision)
    print(f'Recall: ', recall)
    print(f'F1-score: ', f1)    

def train_and_check(model, df_train, df_test, text_col1 = "Title", text_col2 = "Description"):
    
    smoothening = 1
    
    print("Removing punctuations...")
    df_train = model.clean_punctuation(df_train, text_col=text_col1)
    df_test = model.clean_punctuation(df_test, text_col=text_col1)
    df_train = model.clean_punctuation(df_train, text_col=text_col2)
    df_test = model.clean_punctuation(df_test, text_col=text_col2)
    df_train = model.merge_col(df_train, text_col1="Title", text_col2="Description")
    df_test = model.merge_col(df_test, text_col1="Title", text_col2="Description")
    
    print("Tokenizing text columns...")
    df_train = model.tokenize_col(df_train, text_col=text_col1)
    df_test = model.tokenize_col(df_test, text_col=text_col1)
    df_train = model.tokenize_col(df_train, text_col=text_col2)
    df_test = model.tokenize_col(df_test, text_col=text_col2)
    df_train = model.tokenize_col(df_train, text_col="Title Description")
    df_test = model.tokenize_col(df_test, text_col="Title Description")    
    
    # Remove Stopwords
    print("Removing stopwords...")
    df_train = model.remove_stopwords(df_train, text_col="Tokenized Description")
    df_test = model.remove_stopwords(df_test, text_col="Tokenized Description")
    df_train = model.remove_stopwords(df_train, text_col="Tokenized Title")
    df_test = model.remove_stopwords(df_test, text_col="Tokenized Title")
    df_train = model.remove_stopwords(df_train, text_col="Tokenized Title Description")
    df_test = model.remove_stopwords(df_test, text_col="Tokenized Title Description")

    # Remove stopwords and stemming
    print("Remove stopwords and stemming...")
    df_train = model.stem(df_train, text_col="Tokenized Description No Stopwords")
    df_test = model.stem(df_test, text_col="Tokenized Description No Stopwords")
    df_train = model.stem(df_train, text_col="Tokenized Title No Stopwords")
    df_test = model.stem(df_test, text_col="Tokenized Title No Stopwords")
    df_train = model.stem(df_train, text_col="Tokenized Title Description No Stopwords")
    df_test = model.stem(df_test, text_col="Tokenized Title Description No Stopwords")

    # Generate bigrams
    print("Generating bigrams...")
    df_train = model.generate_bigrams(df_train, text_col="Tokenized Description No Stopwords Stemmed")
    df_test = model.generate_bigrams(df_test, text_col="Tokenized Description No Stopwords Stemmed")
    df_train = model.generate_bigrams(df_train, text_col="Tokenized Title No Stopwords Stemmed")
    df_test = model.generate_bigrams(df_test, text_col="Tokenized Title No Stopwords Stemmed")
    df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Description No Stopwords Stemmed")
    df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Description No Stopwords Stemmed")

    # print(df_train.iloc[0])

    # Combining unigrams and bigrams
    print("Combining unigrams and bigrams...")
    df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Description No Stopwords Stemmed", text_col2="Bigrams Tokenized Description No Stopwords Stemmed")
    df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Description No Stopwords Stemmed", text_col2="Bigrams Tokenized Description No Stopwords Stemmed")
    df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Title No Stopwords Stemmed", text_col2="Bigrams Tokenized Title No Stopwords Stemmed")
    df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Title No Stopwords Stemmed", text_col2="Bigrams Tokenized Title No Stopwords Stemmed")
    df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Title Description No Stopwords Stemmed", text_col2="Bigrams Tokenized Title Description No Stopwords Stemmed")
    df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Title Description No Stopwords Stemmed", text_col2="Bigrams Tokenized Title Description No Stopwords Stemmed")


    print("Training...")

    # Part 6.b

    str1 = "Tokenized Title No Stopwords Stemmed Bigrams Tokenized Title No Stopwords Stemmed"
    str2 = "Tokenized Description No Stopwords Stemmed Bigrams Tokenized Description No Stopwords Stemmed"

    model.fit_two_features(df_train, smoothening, text_col1 = str1, text_col2= str2)
    df_train= model.predict_two_features(df_train, text_col1 = str1, text_col2= str2, predicted_col = "Predicted")
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy:  ", train_accuracy)
    df_train= model.predict_two_features(df_test, text_col1 = str1, text_col2= str2, predicted_col = "Predicted")
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for:  ", test_accuracy)
    get_metrics(df_test)

    # Part 7.a and 7.b
    class_val = np.random.randint(1,4)
    baseline_train_accuracy = model.baseline_accuracy(df_train, class_val)
    baseline_test_accuracy = model.baseline_accuracy(df_test, class_val)
    print("Training accuracy, for random class: ", baseline_train_accuracy)
    print("Testing accuracy, for random class", baseline_test_accuracy)
    baseline_train_accuracy = model.baseline_accuracy(df_train, 1)
    baseline_test_accuracy = model.baseline_accuracy(df_test, 1)
    print("Training accuracy, for always predicting 1: ", baseline_train_accuracy)
    print("Testing accuracy, for always predicting 1", baseline_test_accuracy)


    # Part 6.a

    str3 = "Tokenized Title Description No Stopwords Stemmed Bigrams Tokenized Title Description No Stopwords Stemmed"
    params = model.fit(df_train, smoothening, text_col=str3)
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col=str3)
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy:  ", train_accuracy)
    df_test = model.predict(df_test, text_col=str3)
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy:  ", test_accuracy)
    get_metrics(df_test)

    return df_train, df_test

model = naive_bayes.NaiveBayes()

'''
PREPROCESSING
'''
# Read dataframe...
print("Reading data...")
path_train = '../data/Q1/train.csv'
path_test = '../data/Q1/test.csv'
df_train = model.read(path_train)
df_test = model.read(path_test)

df_train, df_test = train_and_check(model, df_train, df_test, text_col1="Title", text_col2="Description")
