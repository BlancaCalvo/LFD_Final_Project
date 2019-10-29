import sys
import pandas as pd
import Final_model.preprocess_data, \
    Final_model.balance_training_data, Final_model.classification

# Loads the data
def load_data(filename):
    data = pd.read_csv(filename,
                       compression='infer', sep='\t',
                       encoding='utf-8', index_col=0).dropna()
    return data


def main():
    if len(sys.argv) == 3:
        train = load_data(sys.argv[1])
        print(train.head(3))
        train = Final_model.preprocess_data.preprocess(train, train=True)
        train = Final_model.balance_training_data.balance_data(train)

        test = load_data(sys.argv[2])
        print(test.head(3))
        test = Final_model.preprocess_data.preprocess(test, train=False)

        Final_model.classification.two_step_classification(train, test)

    else:
        print("You need to provide two files: one set of training data and one set of test data.")


if __name__ == '__main__':
    main()