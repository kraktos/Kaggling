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

    # get the independant and dependant variables from training
    training = df_train.loc[0:, "datetime":"windspeed"]
    target_registered = df_train.loc[0:, 'registered']
    target_casual = df_train.loc[0:, 'casual']

    # get the independant variables from testing
    test = df_test_temp.loc[0:, "datetime":"windspeed"]

    print("DataFrame Dimension = " + str(df_train.shape))
    print("Training Dimension = " + str(training.shape))
    print("Target Dimension = " + str(target_casual.shape))
    print("Test Dimension = " + str(test.shape))

    # create and train the random forest
    # multi-core CPUs can use:
    # regress = DecisionTreeRegressor(max_depth=7)
    regress_casual = RandomForestRegressor(n_estimators=80, max_depth=9)
    regress_registered = RandomForestRegressor(n_estimators=80, max_depth=9)
    regress_casual.fit(training, target_casual)
    regress_registered.fit(training, target_registered)

    # output for submission
    np.savetxt("Data/submission.csv", np.column_stack((df_test_orig['datetime'],
                            regress_casual.predict(test) + regress_registered.predict(test))),
               delimiter="\t", fmt='%s,%f', header='datetime,count')


if __name__ == "__main__":
    main()
