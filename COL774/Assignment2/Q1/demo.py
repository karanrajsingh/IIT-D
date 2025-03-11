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
df_train = model.tokenize_col(df_train, text_col="Title Description")
df_test = model.tokenize_col(df_test, text_col="Title Description")

# print(df_test.iloc[0])

# Training simple data
print("Training simple data...")
params = model.fit(df_train, smoothening=0.1)
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train)
train_accuracy = model.accuracy(df_train)
print("Train accuracy:  ", train_accuracy)
df_test = model.predict(df_test)
test_accuracy = model.accuracy(df_test)
print("Test accuracy:  ", test_accuracy)

# model.wordcloud(df_train)

# line = "The hills are abound with the sound of music!"
# clean_stopwords = model.remove_stopwords_from_text(line)
# print(clean_stopwords)

# Remove Stopwords
print("Removing stopwords...")
df_train = model.remove_stopwords(df_train, text_col="Title")
df_test = model.remove_stopwords(df_test, text_col="Title")
df_train = model.remove_stopwords(df_train, text_col="Description")
df_test = model.remove_stopwords(df_test, text_col="Description")
df_train = model.remove_stopwords(df_train, text_col="Title Description")
df_test = model.remove_stopwords(df_test, text_col="Title Description")

# Training after removing stopwords
print("Training the model after stopword removal...")
params = model.fit(df_train, smoothening=0.1, text_col="Tokenized Description No Stopwords")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Description No Stopwords")
train_accuracy = model.accuracy(df_train, )
print("Train accuracy after stopword removal  ", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Description No Stopwords")
test_accuracy = model.accuracy(df_test)
print("Test accuracy after stopword removal:  ", test_accuracy)

# line = "work worked working"
# stemmed_line = model.stemming_the_text(line)
# print(stemmed_line)

# Stemming
print("Stemming...")
df_train = model.stem(df_train, text_col="Tokenized Title No Stopwords")
df_test = model.stem(df_test, text_col="Tokenized Title No Stopwords")
df_train = model.stem(df_train, text_col="Tokenized Description No Stopwords")
df_test = model.stem(df_test, text_col="Tokenized Description No Stopwords")
df_train = model.stem(df_train, text_col="Tokenized Title Description No Stopwords")
df_test = model.stem(df_test, text_col="Tokenized Title Description No Stopwords")

# print(df_train.columns)
# print(df_test.columns)


# Training after stemming
print("Training after stemming...")
params = model.fit(df_train, smoothening=0.1, text_col="Tokenized Description No Stopwords Stemmed")
priors = params[0]
estimates = params[1]
df_train= model.predict(df_train, text_col="Tokenized Description No Stopwords Stemmed")
train_accuracy = model.accuracy(df_train)
print("Train accuracy after stemming:  ", train_accuracy)
df_test = model.predict(df_test, text_col="Tokenized Description No Stopwords Stemmed")
test_accuracy = model.accuracy(df_test)
print("Test accuracy after stemming:  ", test_accuracy)


