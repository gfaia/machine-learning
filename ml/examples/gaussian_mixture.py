import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from ml.cluster import GaussianMixtureModel


def GaussianMix_random_dataset():
    """Kmeans ++ classify random dataset."""

    X, y = make_blobs(n_features=2, centers=3)
    model = GaussianMixtureModel(max_iter=100).train(X)
    centers = model.mu
    predicted = model.predict(X)

    # Plot the predicted results.
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=predicted, s=25, edgecolor="k")
    plt.scatter(centers[:, 0], centers[:, 1], marker="*")
    plt.show()


if __name__ == "__main__":
    GaussianMix_random_dataset()
