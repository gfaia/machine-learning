import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

from ml.classification import PerceptronModel
from ml.utils import train_test_split
from ml.validate import accuracy_score


def perceptron_classify_random_data():
    """Basic perceptron model,
    Generalization performance will drops dramatically
    When feature space is inseparable.
    """
    X, y = make_classification(
        n_samples=1000, n_features=2, n_redundant=0, n_informative=1, random_state=1, n_clusters_per_class=1
    )

    X_train, y_train, X_test, y_test = train_test_split(X, y)
    model = PerceptronModel().train(X_train, y_train)

    predicted = model.predict(X_test)
    print("Classification accuracy: %.2f" % accuracy_score(y_test, predicted))


if __name__ == "__main__":
    perceptron_classify_random_data()
