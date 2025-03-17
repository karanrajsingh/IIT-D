import naive_bayes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def get_metrics(df):
    y_test = df["Class Index"]
    y_pred = df["Predicted"]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' handles multiclass
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

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

# Remove punctuations
print("Removing punctuations...")
df_train = model.clean_punctuation(df_train, text_col="Title")
df_test = model.clean_punctuation(df_test, text_col="Title")
# df_train = model.clean_punctuation(df_train, text_col="Description")
# df_test = model.clean_punctuation(df_test, text_col="Description")
# df_train = model.merge_col(df_train, text_col1="Title", text_col2="Description")
# df_test = model.merge_col(df_test, text_col1="Title", text_col2="Description")

smoothening = 1

# Tokenize
print("Tokenizing text columns...")
df_train = model.tokenize_col(df_train, text_col="Title")
df_test = model.tokenize_col(df_test, text_col="Title")
# df_train = model.tokenize_col(df_train, text_col="Description")
# df_test = model.tokenize_col(df_test, text_col="Description")
# df_train = model.tokenize_col(df_train, text_col="Title Description")
# df_test = model.tokenize_col(df_test, text_col="Title Description")

# print(df_test.iloc[0])

# print("Training on 'Description' column...")

# Training simple data

print("Training simple data over 'Title'...")
params = model.fit(df_train, smoothening, text_col="Tokenized Title")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Title")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Tokenized Title':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Title")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Tokenized Title':  ", test_accuracy)

# print("Training simple data over 'Description'...")
# params = model.fit(df_train, smoothening = 1)
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train)
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Tokenized Description':  ", train_accuracy)
# df_test = model.predict(df_test)
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Tokenized Description':  ", test_accuracy)
get_metrics(df_test)

# print("Training simple data over 'Title Description'...")
# params = model.fit(df_train, smoothening=1, text_col="Tokenized Title Description")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Title Description")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Tokenized Title Description':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Title Description")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Tokenized Title Description':  ", test_accuracy)

# model.wordcloud(df_train)

# Remove Stopwords
print("Removing stopwords...")
df_train = model.remove_stopwords(df_train, text_col="Tokenized Title")
df_test = model.remove_stopwords(df_test, text_col="Tokenized Title")
# df_train = model.remove_stopwords(df_train, text_col="Tokenized Description")
# df_test = model.remove_stopwords(df_test, text_col="Tokenized Description")
# df_train = model.remove_stopwords(df_train, text_col="Tokenized Title Description")
# df_test = model.remove_stopwords(df_test, text_col="Tokenized Title Description")

# Training after removing stopwords
print("Training the model after stopword removal for 'Title'...")
params = model.fit(df_train, smoothening, text_col="Tokenized Title No Stopwords")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Title No Stopwords")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Tokenized Title No Stopwords'", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Title No Stopwords")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Tokenized Title No Stopwords':  ", test_accuracy)

# print("Training the model after stopword removal for 'Description'...")
# params = model.fit(df_train, smoothening, text_col="Tokenized Description No Stopwords")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Description No Stopwords")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Tokenized Description No Stopwords'", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Description No Stopwords")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Tokenized Description No Stopwords':  ", test_accuracy)
get_metrics(df_test)

# print("Training the model after stopword removal for 'Title Description'...")
# params = model.fit(df_train, smoothening, text_col="Tokenized Title Description No Stopwords")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Title Description No Stopwords")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Tokenized Title Description No Stopwords'", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Title Description No Stopwords")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Tokenized Title Description No Stopwords':  ", test_accuracy)

# Stemming
print("Stemming...")
df_train = model.stem(df_train, text_col="Tokenized Title")
df_test = model.stem(df_test, text_col="Tokenized Title")
# df_train = model.stem(df_train, text_col="Tokenized Description")
# df_test = model.stem(df_test, text_col="Tokenized Description")
# df_train = model.stem(df_train, text_col="Tokenized Title Description")
# df_test = model.stem(df_test, text_col="Tokenized Title Description")

# Training after stemming
print("Training the model after stemming for 'Title'...")
params = model.fit(df_train, smoothening, text_col="Tokenized Title Stemmed")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Title Stemmed")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Tokenized Title Stemmed'", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Title Stemmed")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Tokenized Title Stemmed':  ", test_accuracy)

# print("Training the model after stemming for 'Description'...")
# params = model.fit(df_train, smoothening, text_col="Tokenized Description Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Description Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Tokenized Description Stemmed'", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Description Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Tokenized Description Stemmed':  ", test_accuracy)
get_metrics(df_test)

# print("Training the model after stemming for 'Title Description'...")
# params = model.fit(df_train, smoothening, text_col="Tokenized Title Description Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Title Description Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Tokenized Title Description Stemmed'", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Title Description Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Tokenized Title Description Stemmed':  ", test_accuracy)

# Remove stopwords and stemming
print("Remove stopwords and stemming...")
df_train = model.stem(df_train, text_col="Tokenized Title No Stopwords")
df_test = model.stem(df_test, text_col="Tokenized Title No Stopwords")
# df_train = model.stem(df_train, text_col="Tokenized Description No Stopwords")
# df_test = model.stem(df_test, text_col="Tokenized Description No Stopwords")
# df_train = model.stem(df_train, text_col="Tokenized Title Description No Stopwords")
# df_test = model.stem(df_test, text_col="Tokenized Title Description No Stopwords")

# Training after removing stopwords and stemming
print("Training after removing stopwords and stemming...")
params = model.fit(df_train, smoothening, text_col="Tokenized Title No Stopwords Stemmed")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Title No Stopwords Stemmed")
train_accuracy = model.accuracy(df_train)
print("Train accuracy after removing stopwords and stemming for 'Tokenized Title':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Title No Stopwords Stemmed")
test_accuracy = model.accuracy(df_test)
print("Test accuracy after removing stopwords and stemming for 'Tokenized Title':  ", test_accuracy)

# params = model.fit(df_train, smoothening, text_col="Tokenized Description No Stopwords Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Description No Stopwords Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy after removing stopwords and stemming for 'Tokenized Description':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Description No Stopwords Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy after removing stopwords and stemming for 'Tokenized Description':  ", test_accuracy)
get_metrics(df_test)

# model.wordcloud(df_train)
# model.wordcloud(df_test)

# params = model.fit(df_train, smoothening, text_col="Tokenized Title Description No Stopwords Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Title Description No Stopwords Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy after removing stopwords and stemming for 'Tokenized Title Description':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Title Description No Stopwords Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy after removing stopwords and stemming for 'Tokenized Title Description':  ", test_accuracy)

# # Generate bigrams
print("Generating bigrams for simple data...")
df_train = model.generate_bigrams(df_train, text_col="Tokenized Title")
df_test = model.generate_bigrams(df_test, text_col="Tokenized Title")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Description")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Description")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Description")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Description")
df_train = model.generate_bigrams(df_train, text_col="Tokenized Title No Stopwords")
df_test = model.generate_bigrams(df_test, text_col="Tokenized Title No Stopwords")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Description No Stopwords")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Description No Stopwords")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Description No Stopwords")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Description No Stopwords")
df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Stemmed")
df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Stemmed")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Description Stemmed")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Description Stemmed")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Description Stemmed")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Description Stemmed")
df_train = model.generate_bigrams(df_train, text_col="Tokenized Title No Stopwords Stemmed")
df_test = model.generate_bigrams(df_test, text_col="Tokenized Title No Stopwords Stemmed")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Description No Stopwords Stemmed")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Description No Stopwords Stemmed")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Description No Stopwords Stemmed")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Description No Stopwords Stemmed")

# Training bigrams over simple data
print("Training bigrams...")

params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Title")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Bigrams Tokenized Title")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Bigrams Tokenized Title':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Bigrams Tokenized Title")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Bigrams Tokenized Title':  ", test_accuracy)

# params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Description")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Description")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Description':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Description")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Description':  ", test_accuracy)
get_metrics(df_test)

# params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Title Description")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Title Description")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Title Description':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Title Description")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Title Description':  ", test_accuracy)

params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Title No Stopwords")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Bigrams Tokenized Title No Stopwords")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Bigrams Tokenized Title No Stopwords':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Bigrams Tokenized Title No Stopwords")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Bigrams Tokenized Title No Stopwords':  ", test_accuracy)

# params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Description No Stopwords")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Description No Stopwords")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Description No Stopwords':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Description No Stopwords")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Description No Stopwords':  ", test_accuracy)
get_metrics(df_test)

# params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Title Description No Stopwords")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Title Description No Stopwords")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Title Description No Stopwords':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Title Description No Stopwords")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Title Description No Stopwords':  ", test_accuracy)

params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Title Stemmed")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Bigrams Tokenized Title Stemmed")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Bigrams Tokenized Title Stemmed':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Bigrams Tokenized Title Stemmed")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Bigrams Tokenized Title Stemmed':  ", test_accuracy)

# params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Description Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Description Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Description Stemmed':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Description Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Description Stemmed':  ", test_accuracy)
get_metrics(df_test)

# params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Title Description Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Title Description Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Title Description Stemmed':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Title Description Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Title Description Stemmed':  ", test_accuracy)

params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Title No Stopwords Stemmed")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Bigrams Tokenized Title No Stopwords Stemmed")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Bigrams Tokenized Title No Stopwords Stemmed':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Bigrams Tokenized Title No Stopwords Stemmed")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Bigrams Tokenized Title No Stopwords Stemmed':  ", test_accuracy)

# params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Description No Stopwords Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Description No Stopwords Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Description No Stopwords Stemmed':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Description No Stopwords Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Description No Stopwords Stemmed':  ", test_accuracy)
get_metrics(df_test)

# params = model.fit(df_train, smoothening, text_col="Bigrams Tokenized Title Description No Stopwords Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Title Description No Stopwords Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Title Description No Stopwords Stemmed':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Title Description No Stopwords Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Title Description No Stopwords Stemmed':  ", test_accuracy)

# print(df_train.iloc[0])

# Combining unigrams and bigrams
print("Combining unigrams and bigrams for simple data...")
df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Title", text_col2="Bigrams Tokenized Title")
df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Title", text_col2="Bigrams Tokenized Title")
# df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Description", text_col2="Bigrams Tokenized Description")
# df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Description", text_col2="Bigrams Tokenized Description")
# # # df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Title Description", text_col2="Bigrams Tokenized Title Description")
# # # df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Title Description", text_col2="Bigrams Tokenized Title Description")

# # # print(df_test.iloc[0])

# Training bigrams over combined data
print("Training bigrams over combined data...")
params = model.fit(df_train, smoothening, text_col="Tokenized Title Bigrams Tokenized Title")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Title Bigrams Tokenized Title")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Tokenized Title Bigrams Tokenized Title':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Title Bigrams Tokenized Title")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Tokenized Title Bigrams Tokenized Title':  ", test_accuracy)
get_metrics(df_test)

# Combining unigrams and bigrams
print("Combining unigrams and bigrams for removed stopwords...")
df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Title No Stopwords", text_col2="Bigrams Tokenized Title No Stopwords")
df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Title No Stopwords", text_col2="Bigrams Tokenized Title No Stopwords")
# df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Description No Stopwords", text_col2="Bigrams Tokenized Description No Stopwords")
# df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Description No Stopwords", text_col2="Bigrams Tokenized Description No Stopwords")
# df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Title Description No Stopwords", text_col2="Bigrams Tokenized Title Description No Stopwords")
# # df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Title Description No Stopwords", text_col2="Bigrams Tokenized Title Description No Stopwords")

# # print(df_test.iloc[0])

# Training bigrams over combined data
print("Training bigrams over combined data...")
params = model.fit(df_train, smoothening, text_col="Tokenized Title No Stopwords Bigrams Tokenized Title No Stopwords")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Title No Stopwords Bigrams Tokenized Title No Stopwords")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Tokenized Title Bigrams No Stopwords Tokenized Title No Stopwords':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Title No Stopwords Bigrams Tokenized Title No Stopwords")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Tokenized Title No Stopwords Bigrams Tokenized Title No Stopwords':  ", test_accuracy)
get_metrics(df_test)