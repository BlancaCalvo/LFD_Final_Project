from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.multiclass import unique_labels
import numpy as np, pandas as pd, math
from sklearn.utils import shuffle
import pickle
import json
import nltk

np.random.seed(1612)


def identity(x):
   return x

class Adj_Adv_counter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def _get_features(self, doc):
        n_sup = 0
        n_comp = 0
        n_punct = 0
        pos_doc = nltk.pos_tag(doc)
        for word,tag in pos_doc:
            if (tag == 'RBS' or tag == 'JJS'):
                n_sup += 1
            if (tag == 'RBR' or tag == 'JJR'):
                n_comp += 1
            if (word == '?' or word == '!' or word == '.' or word == ','):
               n_punct += 1
        len_doc = len(doc)
        return {"superlatives": n_sup, "comparatives": n_comp}#,"length": len_doc}#, "punctuation": n_punct}
    # ,, "person": n_per, "location": n_loc, "organisation": n_org,
    def transform(self, documents):
        return [self._get_features(doc) for doc in documents]

def true_false_subsets(data):
    true = data[data.hyperp == True]
    false = data[data.hyperp == False]
    return true, false

def get_pipe_vec():
    tfidf_vec = TfidfVectorizer(max_features=10000,
                                tokenizer=identity,
                                preprocessor=identity,
                                ngram_range=(1, 4))

    adj_adv_vec = Pipeline([('adj_adv', Adj_Adv_counter()),
                            ('vec', DictVectorizer())])

    vec = FeatureUnion([('tfidf', tfidf_vec), ('textstats', adj_adv_vec)])
    return vec

# Infers the multiclass classification from the binary classification results and prints
# classification reports for all classification tasks (binary, multiclass, and combined labels)
def two_step_classification(train, test):
    pd.set_option('max_colwidth', 20)
    pd.set_option('display.max_columns', None)

    true, false = true_false_subsets(train)

    print('Build First Classifier..')
    vec = get_pipe_vec()
    # Fit classifier on the whole dataset and make predictions
    clf_bin = Pipeline([('vec', vec),
                        ('clf', SVC(kernel='linear'))])
    clf_bin.fit(train.text, train.hyperp)
    new_test = test.reset_index()
    hyperp_predictions = clf_bin.predict(new_test.text)
    hyperp_dict = {v: k for v, k in zip(new_test.index.values, hyperp_predictions)}

    print('Build Second Classifier..')
    # Based on the first classifier's predictions, the instances that were classified as True are passed to a classifier
    # to further determine if they're left or right
    clf_true = Pipeline([('vec', TfidfVectorizer(ngram_range=(1,3),
                                                 max_features=10000,
                                                tokenizer=identity,
                                                preprocessor=identity)),
                        ('clf', SVC(kernel='linear'))])
    clf_true.fit(true.text, true.bias)

    print('Build Third Classifier..')
    # Based on the first classifier's predictions, the instances that were classified as False are passed to a classifier
    # to further predict if they're left-center, right-center, or least.
    clf_false = Pipeline([('vec', TfidfVectorizer(ngram_range=(1,3),
                                                  max_features=10000,
                                                tokenizer=identity,
                                                preprocessor=identity)),
                        ('clf', SVC(kernel='linear'))])
    clf_false.fit(false.text, false.bias)

    true_test_origin, false_test_origin = true_false_subsets(new_test)

    true_test_i = [i for i,v in hyperp_dict.items() if v == True]
    false_test_i = [i for i,v in hyperp_dict.items() if v == False]

    true_test = new_test.iloc[true_test_i]
    false_test = new_test.iloc[false_test_i]

    new_test['hyperp_pred'] = np.zeros(len(new_test))
    for i, v in hyperp_dict.items():
        new_test.loc[new_test.index == i, 'hyperp_pred'] = v

    false_predictions = clf_false.predict(false_test.text)
    false_pred_dict = {i:v for i,v in zip(false_test.index.values, false_predictions)}

    true_predictions = clf_true.predict(true_test.text)
    true_pred_dict = {i: v for i, v in zip(true_test.index.values, true_predictions)}

    false_pred_dict.update(true_pred_dict)
    bias_pred_dict = false_pred_dict

    new_test['bias_pred'] = np.zeros(len(new_test))
    for i, v in bias_pred_dict.items():
        new_test.loc[new_test.index == i, 'bias_pred'] = v

    # Get the combined predicted and combined true labels in order to score the classifications
    new_test['combined_pred'] = new_test['hyperp_pred'].astype('str') + ' ' + new_test['bias_pred']
    new_test['combined_true'] = new_test['hyperp'].astype('str') + ' ' + new_test['bias']

    comb_pred = new_test.combined_pred
    comb_true = new_test.combined_true

    #print(new_test)

    # Print classification reports
    print(classification_report(new_test.hyperp, hyperp_predictions))
    print(classification_report(new_test.bias, new_test.bias_pred))
    print(classification_report(comb_true, comb_pred))


    #

    # (maybe in here somewhere you add a confidence threshold?? and then pass the istances to a votingclassifier?? )
    #
    # divide predictions into subsets: True and False
    # fit new model on True data
    # predict bias labels on True data
    #
    # fit model on False data
    # predict bias labels on False data
    return 0

def load_model(filename):
    return pickle.load(open(filename, 'rb'))


def create_confusion_matrix(true, pred):
    lab = unique_labels(true, pred)
    cm = confusion_matrix(true, pred)
    cm_df = pd.DataFrame(cm, lab, lab)
    print(cm_df)


def read_corpus(corpus_file):
    """read input document and return the textual articles
    and either the bias or hyperpartisan label"""

    #data = pd.read_csv(corpus_file, sep='\t') #compression='xz',
    #     encoding='utf-8',
    #     index_col=0).dropna()
    with open(corpus_file) as json_file:
        data = json.load(json_file)

    data = pd.DataFrame(data)
    data['text'] = data['title']+data['text']

    train, test = train_test_split(data, train_size=0.8)


    return train, test

def get_combined_labels(labels_array):
    combined_labels = []
    for label in labels_array:
        combined_labels.append(" ".join(label))
    return combined_labels


def get_single_labels(labels_array, bias):
    labels = []
    if bias:
        for label in labels_array:
            labels.append(label[1])
    else:
        for label in labels_array:
            labels.append(label[0])
    return labels


def main():
    print("Opening data...")
    corpus_file = 'data/tokenized_with_NUM.json'

    train, test = read_corpus(corpus_file)


    two_step_classification(train, test)


    #
    #
    # train_outputs = np.column_stack((y_train_hyperp, y_train_bias))
    #
    # test_outputs = np.column_stack((y_test_hyperp, y_test_bias))
    #
    # #y_train_both_labels = [y_train_bias, train_outputs]
    #
    # model = pipeline.fit(x_train, train_outputs)
    # predictions = model.predict(x_test)
    #
    # bias_predictions = get_single_labels(predictions, bias=True)
    # hyperp_predictions = get_single_labels(predictions, bias=False)
    #
    # combined_predictions = get_combined_labels(predictions)
    # combined_test = get_combined_labels(test_outputs)
    #
    # print(classification_report(y_test_bias, bias_predictions))
    # print(classification_report(y_test_hyperp, hyperp_predictions))
    # print(classification_report(combined_test, combined_predictions))

    # model_hyperp = pipeline.fit(x_train, y_train_hyperp)
    # y_pred_hyperp = model_hyperp.predict(x_test)
    # print(classification_report(y_test_hyperp, y_pred_hyperp))
    #
    # model_bias = pipeline.fit(x_train, y_train_bias)
    # y_pred_bias = model_bias.predict(x_test)
    # print(classification_report(y_test_bias, y_pred_bias))


if __name__ == '__main__':
    main()