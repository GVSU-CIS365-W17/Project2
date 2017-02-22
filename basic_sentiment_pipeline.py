"""
CIS 365 W17
Project 2: Sentiment Analysis
Team LowDash: Mark Jannenga, Tanner Gibson, Sierra Ellison

Trains a logistic regression model for document classification.
Significant changes made to the starter code are:
    -preprocessor: defined at line 72, used at line 158
    -GridSearchCV: defined at line 88, used at line 171
    -k-fold cross validation: defined and used at line 186

Other changes include changes to parameters at lines 156 and 177,
and stemmers defined at lines 40 and 56 but unused
"""

import pandas as pd
import pickle
from pprint import pprint
from time import time
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem import *
from sklearn.model_selection import KFold

# Hint: These are not actually used in the current
# pipeline, but would be used in an alternative
# tokenizer such as PorterStemming.
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')


def snowball(text):
    """
    The SnowballStemmer is an algortihm from the nltk module that
    attempts to find the base word for each token in the text (e.g.
    'testing' becomes 'test'. It takes a string as input and outputs
    a list of base word tokens. It would be used as the tokenizer
    by TfidfVectorizer.

    :param text: The text to tokenize
    :return: A list of base word tokens
    """
    tokens = text.split()
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    return [stemmer.stem(token) for token in tokens]


def porter(text):
    """
    PorterStemmer works similarly to SnowballStemmer, and could also
    be used as a tokenizer by TfidfVectorizer. Note: there is a bug
    in nltk 3.2.2 (the current default version) that causes this
    implementation to cause an exception. nltk 3.2.1 runs it just
    fine.

    :param text: The text to tokenize
    :return: A list of base word tokens
    """
    tokens = text.split()
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def preprocessor(text):
    """
    This function takes raw text and replaces anything formatted as
    an HTML tag with a single space, and removes certain punctuation,
    specifically .!?,/\[]
    This can be used as a preprocessor by TfidfVectorizer.

    :param text: raw text
    :return: preprocessed text without HTML tags or punctuation
    """
    regex_html = re.compile(r'<.*?>')
    regex_punc = re.compile(r'[\.!\?,"/\\\[\]]')
    # regex_punc = re.compile(r'[^\w\s\d\']')
    return regex_punc.sub('', regex_html.sub(' ', text))


def do_gridsearch(X_train, y_train):
    """
    This functions performs a parameter sweep using GridSearchCV.
    Even multithreaded, it takes several hours to run.

    :param X_train: The training data
    :param y_train: The tags for the training data
    :return:
    """
    lr_tfidf = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', LogisticRegression()),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__C': (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5),
        'clf__penalty': ('l2', 'l1'),
        # 'clf__dual': (True, False),
        'clf__fit_intercept': (True, False),
    }

    grid_search = GridSearchCV(lr_tfidf, parameters, n_jobs=-1, verbose=0)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in lr_tfidf.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


# Read in the dataset and store in a pandas dataframe
df = pd.read_csv('./training_movie_data.csv')

# Split your data into training and test sets.
# Allows you to train the model, and then perform
# validation to get a sense of performance.
#
# Hint: This might be an area to change the size
# of your training and test sets for improved
# predictive performance.

# Static training set for use with GridSearchCV
training_size = 35000
X_train = df.loc[:training_size, 'review'].values
y_train = df.loc[:training_size, 'sentiment'].values
X_test = df.loc[training_size:, 'review'].values
y_test = df.loc[training_size:, 'sentiment'].values

# Full set to be used with k-fold cross validation
X_values = df.loc[:, 'review'].values
y_values = df.loc[:, 'sentiment'].values

# Perform feature extraction on the text.
# Hint: Perhaps there are different preprocessors to
# test?
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=True,
                        preprocessor=preprocessor,
                        analyzer='word',
                        # tokenizer=SnowballStemmer,
                        stop_words=None,
                        max_df=0.75,
                        max_features=50000,
                        ngram_range=(1, 2), )

# Hint: There are methods to perform parameter sweeps to find the
# best combination of parameters.  Look towards GridSearchCV in
# sklearn or other model selection strategies.

# Uncomment this and comment out all following code to run GridSearchCV
# do_gridsearch(X_train, y_train)

# Create a pipeline to vectorize the data and then perform regression.
# Hint: Are there other options to add to this process?
# Look to documentation on Regression or similar methods for hints.
# Possibly investigate alternative classifiers for text/sentiment.
lr_tfidf = Pipeline([
    ('vect', tfidf),
    # ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(C=11,
                               fit_intercept=False,
                               penalty='l2')),
])

# Train the pipline using the training set.
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X_values):
    print("train_index: ", train_index, "test_index: ", test_index)
    X_train = df.loc[train_index, 'review'].values
    y_train = df.loc[train_index, 'sentiment'].values
    X_test = df.loc[test_index, 'review'].values
    y_test = df.loc[test_index, 'sentiment'].values
    lr_tfidf.fit(X_train, y_train)
    print('Test Accuracy: %.4f' % lr_tfidf.score(X_test, y_test))

# Print the Test Accuracy
print('Test Accuracy: %.4f' % lr_tfidf.score(X_test, y_test))

# Save the classifier for use later.
pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
