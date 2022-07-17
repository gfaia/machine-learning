import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from ml.classification import NaiveBayesBModel
from ml.utils import train_test_split


def niave_bayes_example():
    """An example from lihang.sml"""

    X = np.array(
        [
            [1, 1],
            [1, 2],
            [1, 2],
            [1, 1],
            [1, 1],
            [2, 1],
            [2, 2],
            [2, 2],
            [2, 3],
            [2, 3],
            [3, 3],
            [3, 2],
            [3, 2],
            [3, 3],
            [3, 3],
        ]
    )
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    model = NaiveBayesBModel().train(X, y)
    label = model.predict(np.array([[2, 1], [2, 5]]))
    print(label)


if __name__ == "__main__":
    niave_bayes_example()
