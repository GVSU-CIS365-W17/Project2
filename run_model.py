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
df = pd.read_csv(sys.argv[1])

# training_size = int(sys.argv[1])
X_test = df.loc[:, 'review'].values
y_test = df.loc[:, 'sentiment'].values

lr_tfidf = pickle.load(open("saved_model.sav", "rb"))

# Train the pipline using the training set.
# lr_tfidf.fit(X_train, y_train)

# Print the Test Accuracy
print('Test Accuracy: %.4f' % lr_tfidf.score(X_test, y_test))

# Save the classifier for use later.
pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
