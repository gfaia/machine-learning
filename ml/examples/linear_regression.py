import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

from ml.regression import LinearRegression
from ml.utils import train_test_split


def LinearR_classify_random_data():
    """Linear regression."""
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target

    X_train, y_train, X_test, y_test = train_test_split(X, y)
    model = LinearRegression().train(X_train, y_train)

    predicted = model.predict(X_test)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, predicted))


if __name__ == "__main__":
    LinearR_classify_random_data()
