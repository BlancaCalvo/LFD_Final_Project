from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.multiclass import unique_labels
import numpy as np, pandas as pd, math
from sklearn.utils import shuffle
import time
import nltk, json


np.random.seed(1612)


class Adj_Adv_counter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _get_features(self, doc):
        pos_doc = nltk.pos_tag(doc)
        print(pos_doc)
        return {"superlatives": len([itm[0] for itm in pos_doc if itm[1] in ["RBS", "JJS"]]),
                "comparatives": len([itm[0] for itm in pos_doc if itm[1] in ["RBR", "JJR"]])}

    def transform(self, documents):
        return [self._get_features(doc) for doc in documents]



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


# Dummy function
def identity(x):
    return x


# Divides the data into hyperpartisan and non-hyperpartisan articles
def true_false_subsets(data):
    true = data[data.hyperp == True]
    false = data[data.hyperp == False]
    return true, false


# Infers the multiclass classification from the binary classification results and prints
# classification reports for all classification tasks (binary, multiclass, and combined labels)
def two_step_classification(train, test):

    true, false = true_false_subsets(train)


    print("Building the first classifier...")
    pipeline = Pipeline([


        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling features from the post's subject line
                ('title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('tfidf', TfidfVectorizer(preprocessor=identity, tokenizer=identity))
                ])),

                # Pipeline for standard bag-of-words model for body
                ('body_bow', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('tfidf', TfidfVectorizer(preprocessor=identity, tokenizer=identity))
                    # ('best', TruncatedSVD(n_components=50)),
                ])),

                ('adj_adv_vec', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('adj_adv', Adj_Adv_counter()),
                    ('vec', DictVectorizer())
                 ])),

                ('adj_adv_vec_title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('adj_adv', Adj_Adv_counter()),
                    ('vec', DictVectorizer())
                ])),

            ],
        )),

        # Use an SVC classifier on the combined features
        ('svc', LinearSVC(max_iter=10000)),
    ])

    # Fit classifier on the whole dataset and make predictions
    t0 = time.time()
    pipeline.fit(train, train.hyperp)
    t1 = time.time()
    print("Fit time: ", t1-t0)
    new_test = test.reset_index()

    print("Predicting hyperpartisanship...")
    hyperp_predictions = pipeline.predict(new_test)
    hyperp_dict = {v: k for v, k in zip(new_test.index.values, hyperp_predictions)}


    # Based on the first classifier's predictions, the instances that were classified as True are passed to a classifier
    # to further determine if they're left or right
    print("Building the second classifier...")
    clf_true = Pipeline([

        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling features from the post's subject line
                ('title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('tfidf', TfidfVectorizer(min_df=50, preprocessor=identity, tokenizer=identity)),
                ])),

                # Pipeline for standard bag-of-words model for body
                ('body_bow', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('tfidf', TfidfVectorizer(preprocessor=identity, tokenizer=identity)),
                    # ('best', TruncatedSVD(n_components=50)),
                ])),

                ('adj_adv_vec', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('adj_adv', Adj_Adv_counter()),
                    ('vec', DictVectorizer())
                ])),

                ('adj_adv_vec_title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('adj_adv', Adj_Adv_counter()),
                    ('vec', DictVectorizer())
                ])),

            ],
        )),

        # Use an SVC classifier on the combined features
        ('svc', LinearSVC(max_iter=10000)),
    ])

    t2 = time.time()
    clf_true.fit(true, true.bias)
    t3 = time.time()
    print("Fit time: ", t3 - t2)

    # Based on the first classifier's predictions, the instances that were classified as False
    # are passed to a classifier to further predict if they're left-center, right-center, or least.
    print("Building the third classifier...")
    clf_false = Pipeline([

        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling features from the post's subject line
                ('title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('tfidf', TfidfVectorizer(min_df=50, preprocessor=identity, tokenizer=identity)),
                ])),

                # Pipeline for standard bag-of-words model for body
                ('body_bow', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('tfidf', TfidfVectorizer(preprocessor=identity, tokenizer=identity)),
                    # ('best', TruncatedSVD(n_components=50)),
                ])),

                ('adj_adv_vec', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('adj_adv', Adj_Adv_counter()),
                    ('vec', DictVectorizer())
                ])),

                ('adj_adv_vec_title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('adj_adv', Adj_Adv_counter()),
                    ('vec', DictVectorizer())
                ])),

            ],
        )),

        # Use an SVC classifier on the combined features
        ('svc', LinearSVC(max_iter=10000)),
    ])

    t4 = time.time()
    clf_false.fit(false, false.bias)
    t5 = time.time()
    print("Fit time: ", t5 - t4)

    true_test_i = [i for i,v in hyperp_dict.items() if v == True]
    false_test_i = [i for i,v in hyperp_dict.items() if v == False]

    true_test = new_test.iloc[true_test_i]
    false_test = new_test.iloc[false_test_i]

    new_test['hyperp_pred'] = np.zeros(len(new_test))
    for i, v in hyperp_dict.items():
        new_test.loc[new_test.index == i, 'hyperp_pred'] = v

    print("Classifying hyperpartisan articles as left or right...")
    true_predictions = clf_true.predict(true_test)
    true_pred_dict = {i: v for i, v in zip(true_test.index.values, true_predictions)}

    false_predictions = clf_false.predict(false_test)
    false_pred_dict = {i:v for i,v in zip(false_test.index.values, false_predictions)}


    print("Classifying non-hyperpartisan articles as left-center, right-center, or least...")
    false_pred_dict.update(true_pred_dict)
    bias_pred_dict = false_pred_dict

    new_test['bias_pred'] = np.zeros(len(new_test))
    for i, v in bias_pred_dict.items():
        new_test.loc[new_test.index == i, 'bias_pred'] = v

    print("Combining classifications and getting scores...")
    # Get the combined predicted and combined true labels in order to score the classifications
    new_test['combined_pred'] = new_test['hyperp_pred'].astype('str') + ' ' + new_test['bias_pred']
    new_test['combined_true'] = new_test['hyperp'].astype('str') + ' ' + new_test['bias']

    comb_pred = new_test.combined_pred
    comb_true = new_test.combined_true


    # Print classification reports
    print(classification_report(new_test.hyperp, hyperp_predictions))
    print(classification_report(new_test.bias, new_test.bias_pred))
    print(classification_report(comb_true, comb_pred))