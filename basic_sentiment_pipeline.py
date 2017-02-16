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
from nltk.stem import *

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
def snowball(text):
    tokens = text.split()
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    return [stemmer.stem(token) for token in tokens]


"""
    There is a bug in nltk 3.2.2 (default version) that breaks PorterStemmer.
    Revert to 3.2.1 or just stick with SnowballStemmer
"""
def porter(text):
    tokens = text.split()
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


"""
    Preprocessor for cleaning input data. Removes HTML tags.
"""
def preprocessor(text):
    regex_html = re.compile(r'<.*?>')
    regex_punc = re.compile(r'[\.!\?,"/\\\[\]]')
    # regex_punc = re.compile(r'[^\w\s\d\']')
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
training_size = 35000     # 34129 optimal for some reason
# training_size = int(sys.argv[1])
X_train = df.loc[:training_size, 'review'].values
y_train = df.loc[:training_size, 'sentiment'].values
X_test = df.loc[training_size:, 'review'].values
y_test = df.loc[training_size:, 'sentiment'].values

# print('Before preprocessing:\n %s\n' % X_train[0])
# print('After preprocessing:\n %s\n' % preprocessor(X_train[0]))

# print('Before tokenization:\n %s\n' % X_train[0])
# print('After tokenization:\n')
# print(', '.join(porter(preprocessor(X_train[0]))))

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
                        tokenizer=None,
                        stop_words=None)

# Hint: There are methods to perform parameter sweeps to find the
# best combination of parameters.  Look towards GridSearchCV in
# sklearn or other model selection strategies.

# Create a pipeline to vectorize the data and then perform regression.
# Hint: Are there other options to add to this process?
# Look to documentation on Regression or similar methods for hints.
# Possibly investigate alternative classifiers for text/sentiment.
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(C=4.5,fit_intercept=False,penalty='l2',random_state=0))])

# Train the pipline using the training set.
lr_tfidf.fit(X_train, y_train)

# Print the Test Accuracy
print('Test Accuracy: %.4f' % lr_tfidf.score(X_test, y_test))

# Save the classifier for use later.
pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
