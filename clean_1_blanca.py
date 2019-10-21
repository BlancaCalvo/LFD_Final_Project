import pandas as pd
from nltk import word_tokenize

def import_data():
	#train = pd.read_csv('../data/hyperp-training-grouped.csv.xz',
	#	compression='xz',
	#	sep='\t',
	#	encoding='utf-8',
	#	index_col=0).dropna()
	train = pd.read_csv("data/small_train_balanced.tsv",sep='\t')
	return train

def replaceMultiple(mainString, toBeReplaces, newString):
	for elem in toBeReplaces :    # Iterate over the strings to be replaced
		if elem in mainString :         # Check if string is in the main string
			mainString = mainString.replace(elem, newString) # Replace the string
	return  mainString

def take_media_names(data):
	publishers = set()
	for row in data:
		row = replaceMultiple(row, ["https://", "http://", ".com/", ".us/", ".org/"], "")
		publishers.add(row)
	return publishers

def clean_data(data, publishers):
	tokenized_line = word_tokenize(data) #tokenize
	tokenized_line = [word.lower() for word in tokenized_line]
	tokenized_line = [word for word in tokenized_line if word not in '..........'] #exclude wierd points and publisher names
	tokenized_line = [word for word in tokenized_line if word not in publishers]
	return tokenized_line

if __name__ == '__main__':	
	pd.options.display.max_colwidth = 100 # limits print() of a string to 100 characters max.
	print('Importing data...')
	train = import_data()
	print('Cleaning data...')
	publishers = take_media_names(train.publisher)
	train['sentences'] = train.text.apply(lambda x: clean_data(x, publishers))
	print(train.sentences.head(30))



