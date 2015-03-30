from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import datetime
from sklearn.ensemble.forest import RandomForestRegressor
import logloss
import numpy as np


def main():
    # read in  data, parse into training and target sets

    # TESTING ##############
    df_test_temp = pd.read_csv('Data/test.csv', header=0)
    df_test_temp['datetime'] = df_test_temp.apply(lambda row: 12-datetime.datetime.strptime(row['datetime'],
                                                                                            '%Y-%m-%d %H:%M:%S').hour, axis=1)
    # TRAINING ##############
    df_train = pd.read_csv('Data/train.csv',  header=0)
    df_train['datetime'] = df_train.apply(lambda row: 12-datetime.datetime.strptime(row['datetime'],
                                                                                   '%Y-%m-%d %H:%M:%S').hour, axis=1)
    training = df_train.loc[0:, "datetime":"windspeed"]
    target = df_train.loc[0:, 'count']
    test = df_test_temp.loc[0:, "datetime":"windspeed"]

    print("DF Dim = " + str(df_train.shape))
    print("training Dim = " + str(training.shape))
    print("target Dim = " + str(target.shape))
    print("Test Dim = " + str(test.shape))

    # multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    # regress = DecisionTreeRegressor(max_depth=8)
    regress = RandomForestRegressor(n_estimators=90, max_depth=9)
    # regress = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=0, loss='ls')

    # Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(training), 6)

    # iterate through the training and test cross validation segments and
    # run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        # regress model on 'registered counts'
        regress.fit(df_train.loc[traincv, "datetime":"windspeed"], df_train.loc[traincv, 'registered'])
        counts_registered = regress.predict(df_train.loc[testcv, "datetime":"windspeed"])

        # regress model on 'casual counts'
        regress.fit(df_train.loc[traincv, "datetime":"windspeed"], df_train.loc[traincv, 'casual'])
        counts_casual = regress.predict(df_train.loc[testcv, "datetime":"windspeed"])

        # regress model on 'sum of them'
        results.append(logloss.llfun(df_train.loc[testcv, 'count'], counts_casual + counts_registered))

    # print out the mean of the cross-validated results
    print("Results: " + str(np.array(results).mean()))


if __name__ == "__main__":
    main()