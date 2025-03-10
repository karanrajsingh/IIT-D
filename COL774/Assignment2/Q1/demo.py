import naive_bayes

model = naive_bayes.NaiveBayes()

'''
PREPROCESSING
'''
# Read dataframe...
path_train = '../data/Q1/train.csv'
path_test = '../data/Q1/test.csv'
df_train = model.read(path_train)
df_test = model.read(path_test)

# Remove punctuations
df_train = model.clean_punctuation(df_train, text_col="Title")
df_test = model.clean_punctuation(df_test, text_col="Title")
df_train = model.clean_punctuation(df_train, text_col="Description")
df_test = model.clean_punctuation(df_test, text_col="Description")
df_train = model.merge_col(df_train, text_col1="Title", text_col2="Description")
df_test = model.merge_col(df_test, text_col1="Title", text_col2="Description")

# Tokenize description for Part 1
df_train = model.tokenize_col(df_train, text_col="Title")
df_test = model.tokenize_col(df_test, text_col="Title")
df_train = model.tokenize_col(df_train, text_col="Description")
df_test = model.tokenize_col(df_test, text_col="Description")
df_train = model.tokenize_col(df_train, text_col="Title Description")
df_test = model.tokenize_col(df_test, text_col="Title Description")

# print(df_test.iloc[0])

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