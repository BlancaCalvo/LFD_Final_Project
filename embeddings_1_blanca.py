
# Description: create word vectors from the full data
# Author: Blanca
# Date: 21-10-19


import json
import pandas as pd
import gensim


with open('tokenised_all.json') as json_file:
    data = json.load(json_file)

data = pd.DataFrame(data)
#print(data.columns.values)

X = data['sentences'].tolist()

model = gensim.models.Word2Vec(X, size=100)
#w2v = dict(zip(model.wv.index2word, model.wv.syn0))
words = list(model.wv.vocab)
vocab_length = len(words)
model.save('model_all.bin')
