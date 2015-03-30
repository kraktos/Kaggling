import pandas as pd


def main():
    # read in  data, parse into training and target sets

    # TRAINING ##############
    df_train = pd.read_csv('data/train.csv')

    # TESTING ##############
    df_test = pd.read_csv('data/test.csv')

    # shape of the inputs
    print df_train.shape, df_test.shape

    # columns in the frame
    print df_train.columns.values

    print df_train.head(5)

    train_X = df_train[0:df_train.shape[0]].as_matrix()
    test_X = df_test[df_test.shape[0]::].as_matrix()
    train_y = df_train['TARGET']


if __name__ == "__main__":
    main()
