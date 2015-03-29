from numpy import genfromtxt, savetxt
from sklearn import linear_model

# https://www.kaggle.com/c/bioresponse


def main():
    # create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv', 'r'), delimiter=',', dtype='f8')[1:]
    target = [x[11] for x in dataset]
    train = [x[1:9] for x in dataset]
    test = genfromtxt(open('Data/test.csv', 'r'), delimiter=',', dtype='f8')[1:, 1:]
    print(test.shape)

    # create and train the random forest
    # multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    regress = linear_model.LinearRegression(fit_intercept=False)
    regress.fit(train, target)
    print(regress.coef_)

    Y_test = regress.predict(test)
    predicted_counts = [x[1] for x in enumerate(regress.predict(test))]

    savetxt('Data/submission.csv', predicted_counts, delimiter=',', fmt='%f',
            header='count', comments='')


if __name__ == "__main__":
    main()
