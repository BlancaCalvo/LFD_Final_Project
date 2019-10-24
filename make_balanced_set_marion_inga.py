# Description: Create a small balanced multi-class dataset
# Author: Marion
# Date: 17-10-19


from collections import Counter
import pandas as pd
from sklearn.utils import shuffle
import math


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



def get_even_distribution(data, bias_label, max_instances):
    sample = data[data.bias == bias_label]
    publ_list = []
    cnt_publ = Counter(sample.publisher)


    for k in cnt_publ.keys():
        publ_list.append(cnt_publ[k])

    goal_list = [0]*len(publ_list)
    result = calculate_distribution(publ_list, goal_list, max_instances, max_instances)
    print(result)

    i = 0
    appended_data = []
    for k in cnt_publ.keys():
        publ_sample = sample[sample.publisher == k]
        appended_data.append(publ_sample[:result[i]])
        i += 1
    partial_set = pd.concat(appended_data)

    return partial_set



if __name__ == '__main__':
    data = pd.read_csv('hyperp-training-grouped.csv.xz',
                       compression='xz', sep='\t',
                       encoding='utf-8', index_col=0).dropna()

    # inspect class distributions

    cnt_hyper = Counter(data.hyperp)
    cnt_bias = Counter(data.bias)


    # show both class distributions
    # print(cnt_hyper)
    # print(cnt_bias)

    # get desired number of items for both binary classes
    bias_amount = min(cnt_bias.values())
    hyperp_amount = math.floor((bias_amount*3)/2)

    # make a small training set (balanced labels)
    small_set = pd.DataFrame()
    for label in cnt_bias.keys():
        if label == 'left-center' or label == 'right-center' or label == 'least':
            small_set = small_set.append(get_even_distribution(data, label, bias_amount), ignore_index=True)
        else:
            small_set = small_set.append(get_even_distribution(data, label, hyperp_amount), ignore_index=True)

    # check that class distribution is now actually balanced
    # and also the distribution of the hyperp labels (unfortunately skewed now)
    # cnt_check = Counter()
    # for bias, hyper in zip(small_set.bias, small_set.hyperp):
    #     cnt_check[bias] += 1
    #     cnt_check[hyper] += 1

    #print(cnt_check)

    # only get ID, hyperp, bias, publisher (website) + text
    cols = ['id', 'hyperp', 'bias', 'publisher', 'text']
    small_set_relevant = small_set.loc[:, small_set.columns.isin(cols)]

    # save the small dataset to tsv file
    small_set_relevant.to_csv(path_or_buf='small_train_balanced_publishers.tsv',
                              index=False,
                              header=cols,
                              sep='\t')


