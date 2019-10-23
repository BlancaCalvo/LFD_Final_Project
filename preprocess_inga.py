import pandas as pd
from nltk import word_tokenize
import json
import re

def import_data():
    # train = pd.read_csv('hyperp-training-grouped.csv.xz',
    #     compression='xz',
    #     sep='\t',
    #     encoding='utf-8',
    #     index_col=0).dropna()
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

# Replace numbers with the placeholder NUM
def replace_numbers(text):
    result = re.sub(r'\d+', 'NUM', text)
    return result

def clean_data(data, publishers):
    data = replace_numbers(data.lower())
    # exclude weird points and publisher names
    tokenized_line = [word for word in word_tokenize(data) if word not in '..........' and word not in publishers]
    return tokenized_line

def restructure_data(data):
    num_rows = len(data.index)
    new_dict = {}
    for i in range (0, num_rows):
        new_dict[i] = [int(data.iloc[i]['id']), data.iloc[i]['bias'], str(data.iloc[i]['hyperp']), data.iloc[i]['text']]
    return new_dict

if __name__ == '__main__':
    pd.options.display.max_colwidth = 100 # limits print() of a string to 100 characters max.
    print('Importing data...')
    train = import_data()
    print('Cleaning data...')
    publishers = take_media_names(train.publisher)
    publishers.add('nbc')
    publishers.add('fox')
    publishers.add('daily')
    publishers.add('albuquerque')
    train['text'] = train.text.apply(lambda x: clean_data(x, publishers))

    dict = train.to_dict('dict')

    with open('tokenized_with_NUM.json', 'w') as json_file:
        json.dump(dict, json_file)
