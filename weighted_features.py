# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def read_corpus(corpus_file, use_bias=False):
    """read input document and return the textual articles
    and either the bias or hyperpartisan label"""
    #with open(corpus_file) as json_file:
    #    data = json.load(json_file)
    
    df = pd.read_csv("small_train_balanced_publishers.tsv",sep='\t')
    #df = pd.DataFrame(data)
    data = df[['text', 'title']]
    if use_bias:
        target = df.bias
    else:
        target = data.hyperp

    return data, target

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key. """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text)}
                for text in posts]


pipeline = Pipeline([

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ('title', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('tfidf', TfidfVectorizer(min_df=50)),
            ])),

            # Pipeline for standard bag-of-words model for body
            ('body_bow', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('tfidf', TfidfVectorizer()),
                #('best', TruncatedSVD(n_components=50)),
            ])),

            # Pipeline for pulling ad hoc features from post's body
            ('body_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
            # Pipeline for pulling ad hoc features from post's body
            ('title_stats', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

        ],

        # weight components in FeatureUnion
        #transformer_weights={
        #    'title': 0.3,
        #    'body_bow': 1.0,
        #    'body_stats': 0.0,
        #    'title_stats':0.1,
        #},
    )),

    # Use a SVC classifier on the combined features
    ('svc', SVC(kernel='linear')),
])


data, target = read_corpus('tokenised.json', use_bias=False)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.8, stratify=target)

#print(data)

pipeline.fit(X_train, Y_train)
y = pipeline.predict(X_test)
print(classification_report(y, Y_test))
