import pandas as pd

train = pd.read_csv('../data/hyperp-training-grouped.csv.xz',
	compression='xz',
	sep='\t',
	encoding='utf-8',
	index_col=0).dropna()

train.to_csv('data.csv')