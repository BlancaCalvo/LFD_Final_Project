# Description: Clean the data from the small dataset, tokenise and save to json
# Author: Blanca
# Date: 21-10-19

import pandas as pd
from nltk import word_tokenize
import json 

def import_data():
	#train = pd.read_csv('../data/hyperp-training-grouped.csv.xz',
	#	compression='xz',
	#	sep='\t',
	#	encoding='utf-8',
	#	index_col=0).dropna()
	train = pd.read_csv("data/small_train_balanced.tsv",sep='\t')
	return train

def replaceMultiple(mainString, toBeReplaces, newString): #replace more than one string
	for elem in toBeReplaces :    # Iterate over the strings to be replaced
		if elem in mainString :         # Check if string is in the main string
			mainString = mainString.replace(elem, newString) # Replace the string
	return  mainString

def take_media_names(data): # gets the names of the publishers from the url provided
	publishers = set() # save it as unique value
	for row in data:
		row = replaceMultiple(row, ["https://", "http://", ".com/", ".us/", ".org/"], "")
		publishers.add(row)
	return publishers

def clean_data(data, publishers):
	tokenized_line = word_tokenize(data) #tokenize
	tokenized_line = [word.lower() for word in tokenized_line]
	tokenized_line = [word for word in tokenized_line if word not in '..........'] #exclude wierd points 
	tokenized_line = [word for word in tokenized_line if word not in publishers] # and publisher names
	return tokenized_line

if __name__ == '__main__':	
	pd.options.display.max_colwidth = 100 # limits print() of a string to 100 characters max.
	print('Importing data...')
	train = import_data()
	print('Cleaning data...')
	publishers = take_media_names(train.publisher)
	train['sentences'] = train.text.apply(lambda x: clean_data(x, publishers))

	train = train.to_dict('dict')

	with open('tokenised.json', 'w') as json_file:
		json.dump(train, json_file) 





