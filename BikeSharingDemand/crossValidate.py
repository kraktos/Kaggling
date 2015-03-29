from sklearn import cross_validation
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from numpy import genfromtxt
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import datetime


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

    # print(df_train.loc[0:30, 'datetime'])
    print("DF Dim = " + str(df_train.shape))

    training = df_train.loc[0:, ("atemp", "humidity", "windspeed")]
    target = df_train.loc[0:, 'count']
    print("training Dim = " + str(training.shape))
    print("target Dim = " + str(target.shape))

    test = df_test_temp.loc[0:, ("atemp", "humidity", "windspeed")]
    print("Test Dim = " + str(test.shape))

    # multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    regress = DecisionTreeRegressor(max_depth=9)
    # regress = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=0, loss='ls')

    # Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(training), 5)

    scores = cross_validation.cross_val_score(regress, training, target, scoring='mean_absolute_error', cv=cv)

    print(scores.mean())

if __name__ == "__main__":
    main()