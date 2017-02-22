"""
    Train a logistic regresion model for document classification.

    Search this file for the keyword "Hint" for possible areas of
    improvement.  There are of course others.
"""

import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import re
import string
from pprint import pprint
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Hint: These are not actually used in the current
# pipeline, but would be used in an alternative
# tokenizer such as PorterStemming.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

"""
    This is a very basic tokenization strategy.

    Hint: Perhaps implement others such as PorterStemming
    Hint: Is this even used?  Where would you place it?
"""
def tokenizer(text):
    return text.split()

"""
    Preprocessor for cleaning input data. Removes HTML tags.
"""
def preprocessor(text):
    regex_html = re.compile(r'<.*?>')
    regex_punc = re.compile(r'[\.!\?,"/\\\[\]]')
    # regex_punc = re.compile('[%s]' % re.escape(string.punctuation))
    return regex_punc.sub('', regex_html.sub(' ', text))


# Read in the dataset and store in a pandas dataframe
df = pd.read_csv('./training_movie_data.csv')

# Split your data into training and test sets.
# Allows you to train the model, and then perform
# validation to get a sense of performance.
#
# Hint: This might be an area to change the size
# of your training and test sets for improved
# predictive performance.
#training_size = 35000     # 34129 optimal for some reason
# training_size = int(sys.argv[1])

#get the x and y values for k fold
X_values = df.loc[:,'review'].values
y_values = df.loc[:,'sentiment'].values

#X_train = df.loc[:training_size, 'review'].values
#y_train = df.loc[:training_size, 'sentiment'].values
#X_test = df.loc[training_size:, 'review'].values
#y_test = df.loc[training_size:, 'sentiment'].values

# print('Before preprocessing:\n %s\n' % X_train[0])
# print('After preprocessing:\n %s\n' % preprocessor(X_train[0]))

# print('Before tokenization:\n %s\n' % X_train[0])
# print('After tokenization:\n')
# print(', '.join(tokenizer(preprocessor(X_train[0]))))

# Perform feature extraction on the text.
# Hint: Perhaps there are different preprocessors to
# test?
# tfidf = TfidfVectorizer(strip_accents=None,
#                         lowercase=False,
#                         preprocessor=None)
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=True,
                        preprocessor=preprocessor,
                        analyzer='word',
                        # tokenizer=tokenizer,
                        stop_words=None,
			max_df=.75,
			max_features=50000,
			ngram_range=(1,2),)

# Hint: There are methods to perform parameter sweeps to find the
# best combination of parameters.  Look towards GridSearchCV in
# sklearn or other model selection strategies.
#tfidf._validate_vocabulary()
# Create a pipeline to vectorize the data and then perform regression.
# Hint: Are there other options to add to this process?
# Look to documentation on Regression or similar methods for hints.
# Possibly investigate alternative classifiers for text/sentiment.
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(C=11.5,fit_intercept=False,penalty='l2'))])
# Train the pipline using the training set.
#lr_tfidf.fit(X_train, y_train)
kf = KFold(n_splits=5)
accurace = 0.0
for train_index, test_index in kf.split(X_values):
	X_train = df.loc[train_index,'review'].values
	y_train = df.loc[train_index,'sentiment'].values
	X_test = df.loc[test_index, 'review'].values
	y_test = df.loc[test_index, 'sentiment'].values
	lr_tfidf.fit(X_train,y_train)
	score = lr_tfidf.score(X_test, y_test)
	accurace = float(accurace) + score
	print('Test Accuracy: %.4f' % score)
# Print the Test Accuracy
print('Test Accuracy: %.4f' % (float(accurace)/5))

# Save the classifier for use later.
pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
