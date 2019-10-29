import pandas as pd
import nltk
import json
import re, math
from collections import Counter
from itertools import chain
import collections


def ngrams(text, n):
    text = text.split()
    return zip(*[text[i:] for i in range(n)])


def replaceMultiple(main_string, to_be_replaced, new_string): #replace more than one string
    for elem in to_be_replaced :    # Iterate over the strings to be replaced
        if elem in main_string :         # Check if string is in the main string
            main_string = main_string.replace(elem, new_string) # Replace the string
    return  main_string


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

# Removes all publishers, given a list of publishers, from the text
def remove_publishers(text, publishers):
    publ_regex = re.compile('\\b(%s)\\W' % ('|'.join(map(re.escape, publishers))), re.I)
    new_text = publ_regex.sub('', text)
    return new_text

# Removes puncuation, except for question marks and exclamation points
def remove_punctuation(text):
    return re.sub(r'[^\w\s\\?\\!]','', text)
    # result = re.sub(r'\s+[?]+\s+', ' ', text)
    # return result

def clean_data(data, publishers, title):
    data = replace_numbers(data)
    data = remove_punctuation(data)
    if not title:
        data = remove_publishers(data, publishers)
    return nltk.tokenize.word_tokenize(data)


def get_even_distribution(data, bias_label, max_instances):
    sample = data[data.bias == bias_label]
    publ_list = []
    cnt_publ = Counter(sample.publisher)


    for k in cnt_publ.keys():
        publ_list.append(cnt_publ[k])

    goal_list = [0]*len(publ_list)
    result = calculate_distribution(publ_list, goal_list, max_instances, max_instances)

    i = 0
    appended_data = []
    for k in cnt_publ.keys():
        publ_sample = sample[sample.publisher == k]
        appended_data.append(publ_sample[:result[i]])
        i += 1
    partial_set = pd.concat(appended_data)

    return partial_set


def calculate_distribution(publishers, goal_list, ultimate_goal, goal):
    publ_counter = 0

    for p in publishers:
        if p != 0:
            publ_counter += 1

    num_per_publ = math.floor(goal/publ_counter)
    rem = goal - (num_per_publ*publ_counter)

    if sum(goal_list) == goal:
        return goal_list

    for i in range(0, len(publishers)):
        if publishers[i] == 0:
            continue
        if publishers[i] <= num_per_publ:
            goal_list[i] += publishers[i]
            rem = num_per_publ - publishers[i]
            publishers[i] -= publishers[i]
            if sum(goal_list) == ultimate_goal:
                return goal_list
        elif publishers[i] > num_per_publ:
            goal_list[i] += num_per_publ
            publishers[i] -= num_per_publ
            if sum(goal_list) == ultimate_goal:
                return goal_list

    rem = rem + (ultimate_goal-sum(goal_list))
    goal_list = calculate_distribution(publishers, goal_list, ultimate_goal, rem)

    return goal_list


if __name__ == '__main__':
    pd.options.display.max_colwidth = 501  # limits print() of a string to 100 characters max.

    print('Importing data...')

    data = pd.read_csv('hyperp-training-grouped.csv.xz',
                       compression='xz', sep='\t',
                       encoding='utf-8', index_col=0).dropna()


    print('Cleaning data...')
    publishers = take_media_names(data.publisher)
    publishers.add('NBC')
    publishers.add('Fox')
    publishers.add('Daily')
    publishers.add('Albuquerque')
    publishers.add('Washington Post') ## THIS IS PROBLEMATIC?? Washington could also just be Washington
    ## USE REGEX for the list of publishers. Just take the same old list and make a function that replaces
    # the names on the list with an empty string or something
    data['text'] = data.text.apply(lambda x: clean_data(x, publishers, title=False))
    data['title'] = data.title = data.title.apply(lambda x: clean_data(x, publishers, title=True))

    print("Removing short articles...")
    short_articles = data[data['text'].map(len) < 100]
    new_data = data[data['text'].map(len) >= 100]

    print("Creating balanced dataset...")
    cnt_hyper = Counter(new_data.hyperp)
    cnt_bias = Counter(new_data.bias)


    # get desired number of items for both binary classes
    bias_amount = min(cnt_bias.values())
    hyperp_amount = math.floor((bias_amount * 3) / 2)

    # make a small training set (balanced labels)
    small_set = pd.DataFrame()
    for label in cnt_bias.keys():
        if label == 'left-center' or label == 'right-center' or label == 'least':
            small_set = small_set.append(get_even_distribution(new_data, label, bias_amount), ignore_index=True)
        else:
            small_set = small_set.append(get_even_distribution(new_data, label, hyperp_amount), ignore_index=True)


    # only get ID, hyperp, bias, publisher (website) + text
    cols = ['id', 'hyperp', 'bias', 'publisher', 'title', 'text']
    small_set_relevant = small_set.loc[:, small_set.columns.isin(cols)]

    # save the small dataset to tsv file
    # print("Saving dataset...")
    # small_set_relevant.to_csv(path_or_buf='small_train_balanced_publishers_new.tsv',
    #                           index=False, header=cols, sep='\t')


    # save the small dataset to json
    print("Saving data to JSON...")
    dict = small_set_relevant.to_dict('dict')
    with open('tokenized_with_NUM_new.json', 'w') as json_file:
         json.dump(dict, json_file)