from sklearn import cross_validation
from sklearn import linear_model
from numpy import genfromtxt


def main():
    # read in  data, parse into training and target sets
    dataset = genfromtxt(open('Data/train.csv', 'r'), delimiter=',', dtype='f8')[1:]
    target = [x[11] for x in dataset]
    train = [x[1:9] for x in dataset]
    test = genfromtxt(open('Data/test.csv', 'r'), delimiter=',', dtype='f8')[1:, 1:]
    print(test.shape)

    # multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    regress = linear_model.LinearRegression(fit_intercept=False)

    # Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(train), 5)

    scores = cross_validation.cross_val_score(regress, train, target, scoring='r2', cv=cv)

    print("Results: " + scores)


if __name__ == "__main__":
    main()