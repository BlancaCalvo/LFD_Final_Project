# Description: Building classifiers and finding the most predictive features
# Author: Marion
# Date: 18-10-19

import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing.text import text_to_word_sequence


def read_corpus(corpus_file, use_bias=False):
    """read input document and return the textual articles
    and either the bias or hyperpartisan label"""

    data = pd.read_csv(corpus_file, sep='\t')
    documents = data.text

    if use_bias:
        labels = data.bias
    else:
        labels = data.hyperp

    return documents, labels


def plot_coefficients(classifier, vectorizer, top_n_features=30):
    # get the coefficients of the classifier and the vocabulary and feature names of the vectorizer
    coef = classifier.coef_
    vocab = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names()

    # placeholder for the word weights
    weights = dict()

    # associate each feature/word+POS tag to its weight
    for word in feature_names:
        weights[word] = coef[0, vocab[word]]

    # convert dictionary into a list of tuples
    weight_word = list(zip(weights.values(), weights.keys()))
    # sort the features by weights
    weight_word.sort(key=lambda elem: elem[0])

    # get the most informative features in both pos and neg directions
    top_neg_coef = weight_word[:top_n_features]
    top_pos_coef = weight_word[-top_n_features:]

    # make arrays of top coefficient values and features (labels)
    top_values = np.asarray([val for (val, word) in top_neg_coef + top_pos_coef])
    words = np.asarray([word for (val, word) in top_neg_coef + top_pos_coef])

    # plot the results
    plt.figure()
    colors = ['blue' if v < 0 else 'red' for v in top_values]
    ypos = np.arange(len(top_values))
    plt.bar(ypos, top_values, color=colors)
    plt.xticks(ypos, words, rotation='vertical')
    plt.xlabel("most informative features")
    plt.ylabel('coefficient value')
    plt.title('Feature contribution of the model')
    plt.show()

    return


if __name__ == '__main__':
    # get documents and labels
    X, Y = read_corpus('small_train_balanced.tsv', use_bias=False)

    # split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y)

    # possible classifiers
    nb = MultinomialNB()
    svm = SVC(kernel='linear')

    # vectorizer
    # at the preprocessing step, the text could benefit from some clearning,
    # i.e. filtering out weird numbers

    vec = TfidfVectorizer(max_features=10000,
                          stop_words='english',
                          tokenizer=text_to_word_sequence,
                          ngram_range=(3, 3))

    # build pipeline
    pipeline = Pipeline([('vec', vec),
                         ('clf', svm)])

    # train the model
    model = pipeline.fit(X_train, Y_train)

    # predict
    y_pred = model.predict(X_test)

    # print the results
    print(classification_report(y_pred, Y_test))

    # get the fitted vectorizer and classifier objects
    cls_object = pipeline.named_steps['clf']
    vec_object = pipeline.named_steps['vec']

    # plot most informative coefficients
    plot_coefficients(cls_object, vec_object)