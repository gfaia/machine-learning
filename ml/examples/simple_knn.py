import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

from ml.classification import SimpleKNNModel
from ml.utils import train_test_split
from ml.validate import accuracy_score


def knn_classify_random_data():
    """Classify the randomly generated dataset."""

    # Modify the size of dataset.
    X, y = make_classification(
        n_samples=1000, n_features=2, n_redundant=0, n_informative=1, random_state=1, n_clusters_per_class=1
    )

    X_train, y_train, X_test, y_test = train_test_split(X, y)
    model = SimpleKNNModel(k_value=3).train(X_train, y_train)

    predicted = model.predict(X_test)
    print("CLassification accuracy: %.2f" % accuracy_score(y_test, predicted))


if __name__ == "__main__":
    knn_classify_random_data()
