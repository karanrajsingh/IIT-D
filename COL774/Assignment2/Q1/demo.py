import naive_bayes

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
df_train = model.clean_punctuation(df_train, text_col="Description")
df_test = model.clean_punctuation(df_test, text_col="Description")
df_train = model.merge_col(df_train, text_col1="Title", text_col2="Description")
df_test = model.merge_col(df_test, text_col1="Title", text_col2="Description")

# Tokenize description for Part 1
print("Tokenizing text columns...")
df_train = model.tokenize_col(df_train, text_col="Title")
df_test = model.tokenize_col(df_test, text_col="Title")
df_train = model.tokenize_col(df_train, text_col="Description")
df_test = model.tokenize_col(df_test, text_col="Description")
df_test = model.tokenize_col(df_test, text_col="Description")
df_train = model.tokenize_col(df_train, text_col="Title Description")
df_test = model.tokenize_col(df_test, text_col="Title Description")

# print(df_test.iloc[0])

print("Training on 'Description' column...")

# # Training simple data
# print("Training simple data...")
# params = model.fit(df_train, smoothening=0.1)
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train)
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Tokenized Description':  ", train_accuracy)
# df_test = model.predict(df_test)
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Tokenized Description':  ", test_accuracy)

# model.wordcloud(df_train)

# # Remove Stopwords
# print("Removing stopwords...")
# df_train = model.remove_stopwords(df_train, text_col="Tokenized Title")
# df_test = model.remove_stopwords(df_test, text_col="Tokenized Title")
# df_train = model.remove_stopwords(df_train, text_col="Tokenized Description")
# df_test = model.remove_stopwords(df_test, text_col="Tokenized Description")
# df_train = model.remove_stopwords(df_train, text_col="Tokenized Title Description")
# df_test = model.remove_stopwords(df_test, text_col="Tokenized Title Description")

# # Training after removing stopwords
# print("Training the model after stopword removal...")
# params = model.fit(df_train, smoothening=0.1, text_col="Tokenized Description No Stopwords")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Description No Stopwords")
# train_accuracy = model.accuracy(df_train, )
# print("Train accuracy after stopword removal for 'Tokenized Description'", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Description No Stopwords")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy after stopword removal for 'Tokenized Description':  ", test_accuracy)

# # Stemming
# print("Stemming...")
# df_train = model.stem(df_train, text_col="Tokenized Title No Stopwords")
# df_test = model.stem(df_test, text_col="Tokenized Title No Stopwords")
# df_train = model.stem(df_train, text_col="Tokenized Description No Stopwords")
# df_test = model.stem(df_test, text_col="Tokenized Description No Stopwords")
# df_train = model.stem(df_train, text_col="Tokenized Title Description No Stopwords")
# df_test = model.stem(df_test, text_col="Tokenized Title Description No Stopwords")

# # Training after stemming
# print("Training after stemming...")
# params = model.fit(df_train, smoothening=0.1, text_col="Tokenized Description No Stopwords Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Tokenized Description No Stopwords Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy after stemming for 'Tokenized Description':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Tokenized Description No Stopwords Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy after stemming for 'Tokenized Description':  ", test_accuracy)

# Generate bigrams
print("Generating bigrams for simple data...")
df_train = model.generate_bigrams(df_train, text_col="Tokenized Title")
df_test = model.generate_bigrams(df_test, text_col="Tokenized Title")
df_train = model.generate_bigrams(df_train, text_col="Tokenized Description")
df_test = model.generate_bigrams(df_test, text_col="Tokenized Description")
df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Description")
df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Description")

# # Training bigrams over simple data
# print("Training bigrams over simple data...")
# params = model.fit(df_train, smoothening=0.1, text_col="Bigrams Tokenized Description")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Description")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Description':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Description")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Description':  ", test_accuracy)

# # Generate bigrams for removed stopwords
# print("Generating bigrams for removed stopwords...")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Title No Stopwords")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Title No Stopwords")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Description No Stopwords")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Description No Stopwords")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Description No Stopwords")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Description No Stopwords")

# # Training bigrams over simple data
# print("Training bigrams over simple data...")
# params = model.fit(df_train, smoothening=0.1, text_col="Bigrams Tokenized Description No Stopwords")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Description No Stopwords")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Description No Stopwords':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Description No Stopwords")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Description No Stopwords':  ", test_accuracy)

# # Generate bigrams for removed stopwords and stemmed
# print("Generating bigrams for removed stopwords and stemmed...")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Title No Stopwords Stemmed")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Title No Stopwords Stemmed")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Description No Stopwords Stemmed")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Description No Stopwords Stemmed")
# df_train = model.generate_bigrams(df_train, text_col="Tokenized Title Description No Stopwords Stemmed")
# df_test = model.generate_bigrams(df_test, text_col="Tokenized Title Description No Stopwords Stemmed")

# # Training bigrams over simple data
# print("Training bigrams over simple data...")
# params = model.fit(df_train, smoothening=0.1, text_col="Bigrams Tokenized Description No Stopwords Stemmed")
# priors = params[0]
# estimates = params[1]
# df_train= model.predict(df_train, text_col="Bigrams Tokenized Description No Stopwords Stemmed")
# train_accuracy = model.accuracy(df_train)
# print("Train accuracy for 'Bigrams Tokenized Description No Stopwords Stemmed':  ", train_accuracy)
# df_test = model.predict(df_test, text_col="Bigrams Tokenized Description No Stopwords Stemmed")
# test_accuracy = model.accuracy(df_test)
# print("Test accuracy for 'Bigrams Tokenized Description No Stopwords Stemmed':  ", test_accuracy)

# Combining unigrams and bigrams
print("Combining unigrams and bigrams for simple data...")
df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Title", text_col2="Bigrams Tokenized Title")
df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Title", text_col2="Bigrams Tokenized Title")
df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Description", text_col2="Bigrams Tokenized Description")
df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Description", text_col2="Bigrams Tokenized Description")
df_train = model.combine_unigram_bigram(df_train, text_col1="Tokenized Title Description", text_col2="Bigrams Tokenized Title Description")
df_test = model.combine_unigram_bigram(df_test, text_col1="Tokenized Title Description", text_col2="Bigrams Tokenized Title Description")

print(df_test.iloc[0])

# Training bigrams over combined data
print("Training bigrams over combined data...")
params = model.fit(df_train, smoothening=0.1, text_col="Tokenized Description Bigrams Tokenized Description")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Description Bigrams Tokenized Description")
train_accuracy = model.accuracy(df_train)
print("Train accuracy for 'Tokenized Description Bigrams Tokenized Description':  ", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Description Bigrams Tokenized Description")
test_accuracy = model.accuracy(df_test)
print("Test accuracy for 'Tokenized Description Bigrams Tokenized Description':  ", test_accuracy)