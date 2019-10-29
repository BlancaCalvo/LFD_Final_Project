from collections import Counter
import math
import pandas as pd


# Creates a partial dataset consisting of one bias class with the correct distribution of publishers
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


# Calculates the distribution that a class should get with respect to number of instances per publisher
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


# Returns a complete, balanced dataset with balanced classes and a balanced
# (as balanced as possible) number of instances from each publisher in every class
def balance_data(data):
    print("Creating balanced dataset...")
    cnt_bias = Counter(data.bias)

    # get desired number of items for both binary classes
    bias_amount = min(cnt_bias.values())
    hyperp_amount = math.floor((bias_amount * 3) / 2)

    # make a small training set (balanced labels)
    small_set = pd.DataFrame()
    for label in cnt_bias.keys():
        if label == 'left-center' or label == 'right-center' or label == 'least':
            small_set = small_set.append(get_even_distribution(data, label, bias_amount), ignore_index=True)
        else:
            small_set = small_set.append(get_even_distribution(data, label, hyperp_amount), ignore_index=True)

    # only get ID, hyperp, bias, publisher (website) + text
    cols = ['id', 'hyperp', 'bias', 'publisher', 'title', 'text']
    small_set_relevant = small_set.loc[:, small_set.columns.isin(cols)]

    return small_set_relevant