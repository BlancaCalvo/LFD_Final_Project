# Description: create word vectors from data
# Author: Blanca
# Date: 21-10-19


import json
import pandas as pd
import gensim
#from sklearn.pipeline import Pipeline
#from sklearn.ensemble import ExtraTreesClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

with open('tokenised.json') as json_file:
    data = json.load(json_file)

data = pd.DataFrame(data)
#print(data.columns.values)

X = data['sentences'].tolist()

model = gensim.models.Word2Vec(X, size=100)
#w2v = dict(zip(model.wv.index2word, model.wv.syn0))
words = list(model.wv.vocab)
vocab_length = len(words)
#model.save('model.bin')

model = Sequential()
model.add(Embedding(vocab_length, 100))#, input_length=length_long_sentence))
#model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

