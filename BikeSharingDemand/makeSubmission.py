from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import datetime
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor

# https://www.kaggle.com/c/bike-sharing-demand


def main():
    # read in  data, parse into training and target sets

    # TESTING ##############
    df_test_orig = pd.read_csv('Data/test.csv', header=0)
    df_test_temp = pd.read_csv('Data/test.csv', header=0)

    df_test_temp['datetime'] = df_test_temp.apply(lambda row: 12-datetime.datetime.strptime(row['datetime'],
                                                                                            '%Y-%m-%d %H:%M:%S').hour, axis=1)

    # TRAINING ##############
    df_train = pd.read_csv('Data/train.csv',  header=0)

    df_train['datetime'] = df_train.apply(lambda row: 12-datetime.datetime.strptime(row['datetime'],
                                                                                    '%Y-%m-%d %H:%M:%S').hour, axis=1)

    print("DF Dim = " + str(df_train.shape))

    training = df_train.loc[0:, "datetime":"windspeed"]
    target = df_train.loc[0:, 'count']
    print("training Dim = " + str(training.shape))
    print("target Dim = " + str(target.shape))

    test = df_test_temp.loc[0:, "datetime":"windspeed"]
    print("Test Dim = " + str(test.shape))

    # create and train the random forest
    # multi-core CPUs can use:
    # regress = DecisionTreeRegressor(max_depth=7)
    regress = RandomForestRegressor(n_estimators=80, max_depth=9)

    # regress = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=0, loss='ls')

    regress.fit(training, target)

    # output for submission
    np.savetxt("Data/submission.csv", np.column_stack((df_test_orig['datetime'], regress.predict(test))),
               delimiter="\t", fmt='%s,%f', header='datetime,count')



if __name__ == "__main__":
    main()
