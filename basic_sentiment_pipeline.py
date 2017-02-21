"""
    Train a logistic regresion model for document classification.

    Search this file for the keyword "Hint" for possible areas of
    improvement.  There are of course others.
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
import sys
import re
import string
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
training_size = 35000     # 34129 optimal for some reason
# training_size = int(sys.argv[1])
#X_train = df.loc[:training_size, 'review'].values
#y_train = df.loc[:training_size, 'sentiment'].values
#X_test = df.loc[training_size:, 'review'].values
#y_test = df.loc[training_size:, 'sentiment'].values

X_values = df.loc[:,'review'].values
y_values = df.loc[:,'sentiment'].values

#xtrain = open("xtrain.txt",'w')
#ytrain = open("ytrain.txt",'w')
#xtest = open("xtest.txt", 'w')
#ytest = open("ytest.txt",'w')

#pprint(X_train, stream=xtrain)
#pprint(y_train, stream=ytrain)
#pprint(X_test, stream=xtest)
#pprint(y_test, stream=ytest)
#exit()
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
			max_df=0.75,
			max_features=50000,
			ngram_range=(1,2),)

# Hint: There are methods to perform parameter sweeps to find the
# best combination of parameters.  Look towards GridSearchCV in
# sklearn or other model selection strategies.
#tuned_parameters = 
# Create a pipeline to vectorize the data and then perform regression.
# Hint: Are there other options to add to this process?
# Look to documentation on Regression or similar methods for hints.
# Possibly investigate alternative classifiers for text/sentiment.
#lr_tfidf = Pipeline([('vect', tfidf),
#                     ('clf', LogisticRegression(C=4.5,fit_intercept=False,penalty='l2',random_state=0))])


lr_tfidf = Pipeline([
    ('vect', tfidf),
    #('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(C=5,
				fit_intercept=False,
				penalty='l2')),
])

#parameters = {
#    'vect__max_df': (0.5, 0.75, 1.0),
#    'vect__max_features': (None, 5000, 10000, 50000),
#    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
#    'clf__C': (0.5,1,1.5,2,2.5,3,3.5,4,4.5,5),
#    'clf__penalty': ('l2', 'l1'),
    #'clf__dual': (True, False),
#    'clf__fit_intercept': (True, False),
#}


#grid_search = GridSearchCV(lr_tfidf, parameters, n_jobs=-1, verbose=0)

#print("Performing grid search...")
#print("pipeline:", [name for name, _ in lr_tfidf.steps])
#print("parameters:")
#pprint(parameters)
#t0 = time()
#grid_search.fit(X_train, y_train)
#print("done in %0.3fs" % (time() - t0))
#print()

#print("Best parameters set:")
#best_parameters = grid_search.best_estimator_.get_params()
#for param_name in sorted(parameters.keys()):
#    print("\t%s: %r" % (param_name, best_parameters[param_name]))


#gscv = GridSearchCV(lr_tfidf, tuned_parameters, cv=5, scoring=pipScore()
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

def pipScore(pipeline):
	return pipeline.score(X_test, y_test)

# Print the Test Accuracy
print('Test Accuracy: %.4f' % lr_tfidf.score(X_test, y_test))

# Save the classifier for use later.
pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
