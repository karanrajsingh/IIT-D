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
df_train = model.clean_punctuation(df_train, text_col="Description")
df_test = model.clean_punctuation(df_test, text_col="Description")

# Tokenize description for Part 1
df_train = model.tokenize_col(df_train, text_col="Description")
print(df_train)