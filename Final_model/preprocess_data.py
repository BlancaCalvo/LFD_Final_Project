import re


def replaceMultiple(main_string, to_be_replaced, new_string): #replace more than one string
    for elem in to_be_replaced :    # Iterate over the strings to be replaced
        if elem in main_string :         # Check if string is in the main string
            main_string = main_string.replace(elem, new_string) # Replace the string
    return  main_string


# Gets the names of the publishers from the url provided
def take_media_names(data):
    publishers = set() # to save publishers as unique values
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


# Cleans the data by removing punctuation. Removes publisher names only for text, not for titles.
# Returns tokenized, cleaned data.
def clean_data(data, publishers, title):
    data = replace_numbers(data)
    data = remove_punctuation(data)
    if not title:
        return [word for word in data.split() if word not in publishers]
    else: return [word for word in data.split()]


# Preprocesses the data. If the data is training data, also removes additional publishers.
# Removes articles shorter than 100 words.
def preprocess(data, train):
    publishers = take_media_names(data.publisher)
    if train:
        publishers.update(['NBC', 'Fox', 'Daily', 'Albuquerque', 'Advertisement',
                           'ADVERTISEMENT', 'Continue', 'Reading', 'Below'])

    print('Cleaning data...')
    data['text'] = data.text.apply(lambda x: clean_data(x, publishers, title=False))
    data['title'] = data.title = data.title.apply(lambda x: clean_data(x, publishers, title=True))

    print("Removing short articles...")
    new_data = data[data['text'].map(len) >= 100]

    return new_data


