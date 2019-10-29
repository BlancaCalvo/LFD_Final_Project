import sys
import time
import pandas as pd
import preprocess_data, balance_training_data, classification

# Loads the data
def load_data(filename):
    data = pd.read_csv(filename,
                       compression='infer', sep='\t',
                       encoding='utf-8', index_col=0).dropna()
    return data


def main():
    if len(sys.argv) == 3:
        start = time.time()
        print('Preprocessing the training data..')
        train = load_data(sys.argv[1])
        print(train.head(3))
        train = preprocess_data.preprocess(train, train=True)
        train = balance_training_data.balance_data(train)

        print('Preprocessing the test data..')
        test = load_data(sys.argv[2])
        print(test.head(3))
        test = preprocess_data.preprocess(test, train=False)

        print('Building the model..')
        classification.two_step_classification(train, test)
        end = time.time()

        print("The whole thing took {0:.2f} minutes".format((end - start) / 60))
    else:
        print("You need to provide two files: one set of training data and one set of test data.")


if __name__ == '__main__':
    main()