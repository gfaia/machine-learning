import numpy as np
import random

from ml.base import ClusterModel
from ml.distance import get_distance_metric


def _k_init(X, n_clusters, distance_metric):
    """Init n_clusters seed according to k-means++."""

    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Select first center randomly.
    center_id = random.randint(0, n_samples - 1)
    centers[0] = X[center_id]
    X_copy = X.copy()
    X_copy = np.delete(X_copy, center_id, 0)

    for i in range(1, n_clusters):
        distances = []
        for j in range(X_copy.shape[0]):
            nearest_distance = np.min([distance_metric(X_copy[j], centers[k]) for k in range(i)])
            distances.append(nearest_distance**2)
        distances_sum = np.sum(distances)
        select_p = [distances[k] / distances_sum for k in range(len(distances))]
        accumulate = [sum(select_p[: k + 1]) for k in range(len(select_p))]

        # select one center.
        r = random.random()
        select_center = 0
        for j in range(len(accumulate)):
            if r < accumulate[j]:
                select_center = j
                break

        centers[i] = X_copy[select_center]
        X_copy = np.delete(X_copy, select_center, 0)

    return centers


def _gaussian_function(x, mu, sigma):
    """Gaussian density function.

    x, (1, n_features) array
    mu, mean vector
    sigma, (n_features, n_features) covariance matrix
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(mu, list) == list:
        mu = np.array(mu)
    if isinstance(sigma, list) == list:
        sigma = np.array(sigma)
    assert isinstance(x, np.ndarray) and isinstance(mu, np.ndarray) and isinstance(sigma, np.ndarray)

    n_features = x.shape[0]
    if mu.shape[0] != n_features:
        raise Exception("Parameters error, the shape of `mu` is not %s" % n_features)
    if sigma.shape != (n_features, n_features):
        raise Exception("Parameters error, the shape of `delta` is not (%s, %s)" % (n_features, n_features))

    det_sigma = np.abs(np.linalg.det(sigma))
    subtract = x - mu
    inv_sigma = np.linalg.inv(sigma)
    mul = np.inner(np.inner(subtract, inv_sigma), subtract)

    return (1 / np.sqrt((2 * np.pi) ** n_features * det_sigma)) * np.exp(-0.5 * mul)


def _random_generate_normal_coefficients(k):
    """Generate k numbers match the sum of them is one."""
    normal_sum = 1
    coef = []
    for i in range(k - 1):

        # while r = random.random() < normal_sum:
        while True:
            r = random.random()
            if r < normal_sum:
                coef.append(r)
                normal_sum -= r
                break
    coef.append(normal_sum)
    return coef


class KMeansCluster(ClusterModel):
    """The KMeans algorithm clusters data by trying to separate
    in n groups of equal variance, minimizing a criterion known
    as the inertia or within-cluster sum-of-squares. This algorithm
    only is fitted to the linear separable data."""

    def __init__(self, max_iter=10, n_clusters=3, dist="euclidean"):

        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.distance_metric = get_distance_metric(dist)

    def _train(self, X):
        """Cluster data into `n_clusters` groups."""

        self.n_samples, self.n_features = n_samples, n_features = X.shape
        n_clusters = self.n_clusters
        metric = self.distance_metric
        max_iter = self.max_iter

        # use k-means++ algorithms to initiate efficiently.
        centers = _k_init(X, n_clusters, metric)

        for i in range(max_iter):
            # generate n_clusters clusters.
            label_set = []
            for j in range(n_samples):
                dist, label = np.inf, 0
                for k in range(n_clusters):
                    temp_dist = metric(centers[k], X[j])
                    if temp_dist < dist:
                        dist, label = temp_dist, k
                label_set.append(label)

            for j in range(n_clusters):
                point_sum, count = 0, 0
                for k in range(n_samples):
                    if label_set[k] == j:
                        point_sum += X[k]
                        count += 1
                centers[j] = point_sum / count

        self.centers = centers
        return self

    def _predict(self, X):
        """Use self.centers to determine the samples'labels.
        The label is the index of center in centers."""

        centers = self.centers
        y_ = []
        samples = X.shape[0]
        dist = self.distance_metric

        for i in range(samples):
            distance = [dist(X[i], center) for center in centers]
            nearest = np.argmin(distance)
            y_.append(nearest)

        return y_


class GaussianMixtureModel(ClusterModel):
    """The Gaussian mixture model is one kind of probability model.
    Use EM algorithm to optimize the parameters."""

    def __init__(self, max_iter=100, n_clusters=3):
        """i_mu i_delta, the parameters of gaussian distribution."""

        self.max_iter = max_iter
        self.n_clusters = n_clusters

    def _train(self, X):
        """Cluster data into `n_clusters` groups."""

        n_samples, n_features = self.n_samples, self.n_features = X.shape
        n_clusters = self.n_clusters
        max_iter = self.max_iter

        # Initialize parameters mu,delta and the coefficients of k
        # classes in the model.Initialize the `mu` parameters with
        # random numbers and `sigma` as identity matrix.
        mu = np.array([[random.random() for _ in range(n_features)] for _ in range(n_clusters)])
        sigma = np.array([np.identity(n_features) for _ in range(n_clusters)])
        coef = _random_generate_normal_coefficients(n_clusters)
        matrix = np.ndarray(shape=(n_samples, n_clusters))

        for i in range(max_iter):

            # Expectation step
            for j in range(n_samples):
                temp_list = [coef[_] * _gaussian_function(X[j], mu[_], sigma[_]) for _ in range(n_clusters)]
                for k in range(n_clusters):
                    matrix[j][k] = temp_list[k] / np.sum(temp_list)

            # Maximize step
            for k in range(n_clusters):
                sub_sum = np.sum([matrix[j][k] for j in range(n_samples)])
                mu[k] = np.sum([matrix[j][k] * X[j] for j in range(n_samples)], axis=0) / sub_sum
                sigma[k] = (
                    np.sum([matrix[j][k] * np.outer(X[j] - mu[k], X[j] - mu[k]) for j in range(n_samples)], axis=0)
                    / sub_sum
                )
                coef[k] = sub_sum / n_samples

        self.mu, self.sigma, self.coef = mu, sigma, coef
        return self

    def _predict(self, X):
        """Use train model to predict model."""

        samples = X.shape[0]
        n_clusters = self.n_clusters
        mu, sigma, coef = self.mu, self.sigma, self.coef
        matrix = np.ndarray(shape=(samples, n_clusters))
        y_ = []

        for j in range(samples):
            temp_list = [coef[_] * _gaussian_function(X[j], mu[_], sigma[_]) for _ in range(n_clusters)]
            for k in range(n_clusters):
                matrix[j][k] = temp_list[k] / np.sum(temp_list)

            y_.append(np.argmax(matrix[j]))

        return y_
