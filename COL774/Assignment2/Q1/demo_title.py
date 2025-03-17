import naive_bayes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def get_metrics(df):
    y_test = df["Class Index"]
    y_pred = df["Predicted"]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: ', accuracy)
    print(f'Precision: ', precision)
    print(f'Recall: ', recall)
    print(f'F1-score: ', f1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    

def train_and_check(model, df_train, df_test, text = "Title"):
    
    smoothening = 1
    
    print("Removing punctuations...")
    df_train = model.clean_punctuation(df_train, text_col=text)
    df_test = model.clean_punctuation(df_test, text_col=text)
    
    print("Tokenizing text columns...")
    df_train = model.tokenize_col(df_train, text_col=text)
    df_test = model.tokenize_col(df_test, text_col=text)
    
    print("Training simple data over {}...".format(text))
    params = model.fit(df_train, smoothening, text_col="Tokenized {}".format(text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Tokenized {}".format(text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Tokenized {}':  ".format(text), train_accuracy)
    df_test = model.predict(df_test, text_col="Tokenized {}".format(text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Tokenized {}':  ".format(text), test_accuracy)
    get_metrics(df_test)

    print("Removing stopwords...")
    df_train = model.remove_stopwords(df_train, text_col="Tokenized {}".format(text))
    df_test = model.remove_stopwords(df_test, text_col="Tokenized {}".format(text))

    print("Training the model after stopword removal for '{}'...".format(text))
    params = model.fit(df_train, smoothening, text_col="Tokenized {} No Stopwords".format(text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Tokenized {} No Stopwords".format(text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Tokenized {} No Stopwords'".format(text), train_accuracy)
    df_test = model.predict(df_test, text_col="Tokenized {} No Stopwords".format(text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Tokenized {} No Stopwords':  ".format(text), test_accuracy)
    get_metrics(df_test)

    print("Stemming...")
    df_train = model.stem(df_train, text_col="Tokenized {}".format(text))
    df_test = model.stem(df_test, text_col="Tokenized {}".format(text))

    print("Training the model after stemming for '{}'...".format(text))
    params = model.fit(df_train, smoothening, text_col="Tokenized {} Stemmed".format(text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Tokenized {} Stemmed".format(text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Tokenized {} Stemmed'".format(text), train_accuracy)
    df_test = model.predict(df_test, text_col="Tokenized {} Stemmed".format(text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Tokenized {} Stemmed':  ".format(text), test_accuracy)
    get_metrics(df_test)

    print("Remove stopwords and stemming...")
    df_train = model.stem(df_train, text_col="Tokenized {} No Stopwords".format(text))
    df_test = model.stem(df_test, text_col="Tokenized {} No Stopwords".format(text))

    print("Training after removing stopwords and stemming...")
    params = model.fit(df_train, smoothening, text_col="Tokenized {} No Stopwords Stemmed".format(text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Tokenized {} No Stopwords Stemmed".format(text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy after removing stopwords and stemming for 'Tokenized {}':  ".format(text), train_accuracy)
    df_test = model.predict(df_test, text_col="Tokenized {} No Stopwords Stemmed".format(text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy after removing stopwords and stemming for 'Tokenized {}':  ".format(text), test_accuracy)
    get_metrics(df_test)

    print("Generating bigrams for simple data...")
    df_train = model.generate_bigrams(df_train, text_col="Tokenized {}".format(text))
    df_test = model.generate_bigrams(df_test, text_col="Tokenized {}".format(text))
    df_train = model.generate_bigrams(df_train, text_col="Tokenized {} No Stopwords".format(text))
    df_test = model.generate_bigrams(df_test, text_col="Tokenized {} No Stopwords".format(text))
    df_train = model.generate_bigrams(df_train, text_col="Tokenized {} Stemmed".format(text))
    df_test = model.generate_bigrams(df_test, text_col="Tokenized {} Stemmed".format(text))
    df_train = model.generate_bigrams(df_train, text_col="Tokenized {} No Stopwords Stemmed".format(text))
    df_test = model.generate_bigrams(df_test, text_col="Tokenized {} No Stopwords Stemmed".format(text))

    print("Training bigrams...")
    params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized {}".format(text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Bigrams Tokenized {}".format(text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Bigrams Tokenized {}':  ".format(text), train_accuracy)
    df_test = model.predict(df_test, text_col="Bigrams Tokenized {}".format(text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Bigrams Tokenized {}':  ".format(text), test_accuracy)
    get_metrics(df_test)

    params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized {} No Stopwords".format(text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Bigrams Tokenized {} No Stopwords".format(text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Bigrams Tokenized {} No Stopwords':  ".format(text), train_accuracy)
    df_test = model.predict(df_test, text_col="Bigrams Tokenized {} No Stopwords".format(text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Bigrams Tokenized {} No Stopwords':  ".format(text), test_accuracy)
    get_metrics(df_test)

    params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized {} Stemmed".format(text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Bigrams Tokenized {} Stemmed".format(text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Bigrams Tokenized {} Stemmed':  ".format(text), train_accuracy)
    df_test = model.predict(df_test, text_col="Bigrams Tokenized {} Stemmed".format(text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Bigrams Tokenized {} Stemmed':  ".format(text), test_accuracy)
    get_metrics(df_test)

    params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized {} No Stopwords Stemmed".format(text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Bigrams Tokenized {} No Stopwords Stemmed".format(text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Bigrams Tokenized {} No Stopwords Stemmed':  ".format(text), train_accuracy)
    df_test = model.predict(df_test, text_col="Bigrams Tokenized {} No Stopwords Stemmed".format(text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Bigrams Tokenized {} No Stopwords Stemmed':  ".format(text), test_accuracy)
    get_metrics(df_test)

    print("Combining unigrams and bigrams for simple data...")
    df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized {}".format(text), text_col2="Bigrams Tokenized {}".format(text))
    df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized {}".format(text), text_col2="Bigrams Tokenized {}".format(text))

    print("Training bigrams over combined data...")
    params = model.fit(df_train, smoothening, text_col="Tokenized {0} Bigrams Tokenized {1}".format(text, text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Tokenized {0} Bigrams Tokenized {1}".format(text, text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Tokenized {0} Bigrams Tokenized {1}':  ".format(text, text), train_accuracy)
    df_test = model.predict(df_test, text_col="Tokenized {0} Bigrams Tokenized {1}".format(text, text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Tokenized {0} Bigrams Tokenized {1}':  ".format(text, text), test_accuracy)
    get_metrics(df_test)

    print("Combining unigrams and bigrams for removed stopwords...")
    df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized {} No Stopwords".format(text), text_col2="Bigrams Tokenized {} No Stopwords".format(text))
    df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized {} No Stopwords".format(text), text_col2="Bigrams Tokenized {} No Stopwords".format(text))

    print("Training bigrams over combined data for removed stopwords...")
    params = model.fit(df_train, smoothening, text_col="Tokenized {0} No Stopwords Bigrams Tokenized {1} No Stopwords".format(text, text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Tokenized {0} No Stopwords Bigrams Tokenized {1} No Stopwords".format(text, text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Tokenized {0} No Stopwords Bigrams Tokenized {1} No Stopwords':  ".format(text, text), train_accuracy)
    df_test = model.predict(df_test, text_col="Tokenized {0} No Stopwords Bigrams Tokenized {1} No Stopwords".format(text, text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Tokenized {0} No Stopwords Bigrams Tokenized {1} No Stopwords':  ".format(text, text), test_accuracy)
    get_metrics(df_test)


    print("Combining unigrams and bigrams for stemming...")
    df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized {} Stemmed".format(text), text_col2="Bigrams Tokenized {} Stemmed".format(text))
    df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized {} Stemmed".format(text), text_col2="Bigrams Tokenized {} Stemmed".format(text))

    print("Training bigrams over combined data for stemmed...")
    params = model.fit(df_train, smoothening, text_col="Tokenized {0} Stemmed Bigrams Tokenized {1} Stemmed".format(text, text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Tokenized {0} Stemmed Bigrams Tokenized {1} Stemmed".format(text, text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Tokenized {0} Stemmed Bigrams Tokenized {1} Stemmed':  ".format(text, text), train_accuracy)
    df_test = model.predict(df_test, text_col="Tokenized {0} Stemmed Bigrams Tokenized {1} Stemmed".format(text, text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Tokenized {0} Stemmed Bigrams Tokenized {1} Stemmed':  ".format(text, text), test_accuracy)
    get_metrics(df_test)

    df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized {} No Stopwords Stemmed".format(text), text_col2="Bigrams Tokenized {} No Stopwords Stemmed".format(text))
    df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized {} No Stopwords Stemmed".format(text), text_col2="Bigrams Tokenized {} No Stopwords Stemmed".format(text))

    print("Training bigrams over combined data for removed stopwords and stemmed...")
    params = model.fit(df_train, smoothening, text_col="Tokenized {0} No Stopwords Stemmed Bigrams Tokenized {1} No Stopwords Stemmed".format(text, text))
    priors = params[0]
    estimates = params[1]
    df_train= model.predict(df_train, text_col="Tokenized {0} No Stopwords Stemmed Bigrams Tokenized {1} No Stopwords Stemmed".format(text, text))
    train_accuracy = model.accuracy(df_train)
    print("Train accuracy for 'Tokenized {0} No Stopwords Stemmed Bigrams Tokenized {1} No Stopwords Stemmed':  ".format(text, text), train_accuracy)
    df_test = model.predict(df_test, text_col="Tokenized {0} No Stopwords Stemmed Bigrams Tokenized {1} No Stopwords Stemmed".format(text, text))
    test_accuracy = model.accuracy(df_test)
    print("Test accuracy for 'Tokenized {0} No Stopwords Stemmed Bigrams Tokenized {1} No Stopwords Stemmed':  ".format(text, text), test_accuracy)
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

df_train, df_test = train_and_check(model, df_train, df_test, text="Title")
