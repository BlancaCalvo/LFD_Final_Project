# Description: Create a small balanced multi-class dataset
# Author: Marion
# Date: 17-10-19


from collections import Counter
import pandas as pd
from sklearn.utils import shuffle


if __name__ == '__main__':
    data = pd.read_csv('hyperp-training-grouped.csv.xz',
                       compression='xz', sep='\t',
                       encoding='utf-8', index_col=0).dropna()

    # inspect class distributions

    Y_hyper = data.hyperp
    Y_bias = data.bias

    cnt_hyper = Counter()
    cnt_bias = Counter()

    for label in Y_hyper:
        cnt_hyper[label] += 1

    for label in Y_bias:
        cnt_bias[label] += 1

    # show both class distributions
    print(cnt_hyper)
    print(cnt_bias)

    # get number of items in minority class
    minority = min(cnt_bias.values())

    # make a small training set (balanced labels)
    small_set = pd.DataFrame()
    for label in cnt_bias.keys():
        sample = data[data.bias == label]
        sample = shuffle(sample)
        small_set = small_set.append(sample[:minority], ignore_index=True)

    # check that class distribution is now actually balanced
    # and also the distribution of the hyperp labels (unfortunately skewed now)
    cnt_check = Counter()
    for bias, hyper in zip(small_set.bias, small_set.hyperp):
        cnt_check[bias] += 1
        cnt_check[hyper] += 1

    print(cnt_check)

    # only get ID, hyperp, bias labels, publisher (website) + text
    cols = ['id', 'hyperp', 'bias', 'publisher', 'text']
    small_set_relevant = small_set.loc[:, small_set.columns.isin(cols)]

    # save the small dataset to tsv file
    small_set_relevant.to_csv(path_or_buf='small_train_balanced.tsv',
                              index=False,
                              header=cols,
                              sep='\t')

