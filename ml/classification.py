import numpy as np
from collections import Counter

from ml.distance import get_distance_metric
from ml.base import BaseClassifyModel, LinearModel


class SimpleKNNModel(BaseClassifyModel):
    """KNN algorithm use distance between samples to define
    similarity. Utilize the most proportion labels in the
    nearest k's samples to label the unknown sample.
    """

    def __init__(self, dist="euclidean", k_value=5):

        self.distance_metric = get_distance_metric(dist)
        self.k = k_value

    def _train(self, X, y):
        """The simple KNN algorithm nearly no need train operation.
        The information is contained in data set.

        X, 2-d numpy.arrays
        y, 1-d numpy.arrays
        """
        self.n_samples, self.n_features = X.shape
        if self.n_samples == 0:
            raise Exception("The number of samples cannot be zero.")

        self.X, self.y = X, y
        return self

    def _predict(self, X):
        """
        X, 2-d numpy.array

        return unpredicted samples' labels.
        """
        n_samples = self.n_samples
        X_model, y_model = self.X, self.y
        samples = X.shape[0]
        k_value = self.k
        distance_metric = self.distance_metric
        y_ = []

        for i in range(samples):
            nearest_k = [distance_metric(X_model[j], X[i]) for j in range(n_samples)]

            arg_index = np.argsort(nearest_k)[:k_value]
            labels = [y_model[index] for index in arg_index]
            max_labels = -1
            counter = Counter(labels)

            for key in counter.keys():
                if counter.get(key) > max_labels:
                    max_labels = key

            y_.append(max_labels)

        return y_


class LogisticRegressionModel(LinearModel):
    """Binary logistic regression, a binary classifier.

    formula，
        p(y = 1 | x) = logistic(w * x)
        w = [bias, w1, w2, ..., wn], n is n_features
    """

    def __init__(self, learn_rate=0.1, init_coef=0.0, init_bias=0.0, max_iter=100, threshold=0.5):

        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.bias = init_bias
        self.coef = init_coef
        self.threshold = threshold

    def check_transform_X_y(self, X, y):
        """Check and transform (X, y) for Logistic algorithm."""
        if set(y) != {0, 1}:
            raise Exception("For binary logistic regression model, set of y should be {0, 1}")

        return X, y

    @staticmethod
    def logistic(x, coef, bias):
        """Logistic function: 1 / (1 + exp(-b-w*x)"""
        return 1 / (1 + np.exp(-np.dot(x, coef) - bias))

    @staticmethod
    def error(output, y, coef, bias):
        """Example Error: -y * log(y') - (1 - y) * np.log(1 - y')"""
        return -y * np.log(output) - (1 - y) * np.log(1 - output)

    def _train(self, X, y):
        """The loss function is optimized by stochastic gradient descent (SGD).

        X, 2-d numpy.arrays
        y, 1-d numpy.arrays
        """
        self.n_samples, self.n_features = n_samples, n_features = X.shape
        max_iter = self.max_iter
        learn_rate = self.learn_rate
        X, y = self.check_transform_X_y(X, y)
        _bias, _coef = self.bias, np.array([self.coef] * n_features)

        # Stochastic gradient descent.
        for i in range(max_iter):
            for j in range(n_samples):
                _coef += learn_rate * (y[j] - self.logistic(X[j], _coef, _bias)) * X[j]
                _bias += learn_rate * (y[j] - self.logistic(X[j], _coef, _bias))

        self.bias, self.coef = _bias, _coef
        return self

    def _predict(self, X):
        """Use logistic linear model to predict labels."""
        _bias, _coef = self.bias, self.coef
        t = self.threshold
        samples = X.shape[0]
        return [1 if p > t else 0 for p in self.logistic(X, _coef, _bias)]


class PerceptronModel(LinearModel):
    """Perceptron is a simple linear model to classify samples.
    PerceptronModel is the original version.

    formula,
        y = sign(w * x + b) = sign(w0 + w1*x1 + ... + wn*xn)
        if sign(z) > 0: y = +1
        else: y = -1
    """

    def __init__(self, max_iter=100, init_coef=0.0, init_bias=0.0, learn_rate=0.1):

        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.bias = init_bias
        self.coef = init_coef

    def check_transform_X_y(self, X, y):
        """Check and transform (X, y) for perceptron algorithm."""

        # The value of y should be -1 or +1
        # Always, the set of y will be {0, 1}, transform 0 to -1
        if set(y) == {0, 1}:
            for i in range(self.n_samples):
                if y[i] == 0:
                    y[i] = -1

        return X.astype(np.float64), y.astype(np.float64)

    def _train(self, X, y):
        """Train the weights w and bias b, parameters of model.

        X, 2-d numpy.arrays
        y, 1-d numpy.arrays and the range of y is {+1, -1}
        """
        self.n_samples, self.n_features = n_samples, n_features = X.shape
        _bias, _coef = self.bias, np.array([self.coef] * n_features)
        max_iter = self.max_iter
        learn_rate = self.learn_rate
        X, y = self.check_transform_X_y(X, y)

        for i in range(max_iter):
            for j in range(n_samples):
                if np.dot(X[j], _coef) <= 0:
                    _bias += np.float64(learn_rate) * y[j]
                    _coef += np.float64(learn_rate) * y[j] * X[j]

        self.bias, self.coef = _bias, _coef
        return self

    def _predict(self, X):
        """
        X, 2-d numpy.array

        return unpredicted samples' labels.
        """
        _bias, _coef = self.bias, self.coef
        samples = X.shape[0]
        X = X.astype(np.float64)
        return [1 if s > 0 else 0 for s in (_bias + np.dot(X, _coef))]


class NaiveBayesBModel(BaseClassifyModel):
    """naive Bayes classifiers are a family of simple probabilistic
    classifiers based on applying Bayes' theorem with strong (naive)
    independence assumptions between the features.

    The multinomial naive bayes version can only train discrete features.

    formula: use Laplace smoothing to estimate probability.
    """

    def __init__(self, lambda_value=1):

        self.lambda_value = lambda_value

    def _train(self, X, y):
        """naive Bayes learning probability from data set.

        X, 2-d numpy.arrays
        y, 1-d numpy.arrays
        """
        self.n_samples, self.n_features = n_samples, n_features = X.shape
        lambda_value = self.lambda_value

        # calculate prior probability and condition probability
        self.labels_counter = labels_counter = Counter(y)
        self.labels_set = labels_set = labels_counter.keys()
        prior_probability = {}

        for l in labels_set:
            prior_probability[l] = (labels_counter.get(l) + lambda_value) / (
                n_samples + len(labels_counter) * lambda_value
            )

        x_split = {l: [] for l in labels_set}

        for i in range(n_samples):
            x_split[y[i]].append(list(X[i]))

        for l in x_split.keys():
            x_split[l] = np.array(x_split[l])

        condition_probability = {l: {} for l in labels_set}
        x_total_set = []

        for i in range(n_features):
            x_total_set.append(len(set(X[:, i])))

        for l in labels_set:
            for i in range(n_features):
                x_set = condition_probability[l][i + 1] = {}
                x_counter = Counter(x_split[l][:, i])
                for x in x_counter.keys():
                    x_set[x] = (x_counter[x] + lambda_value) / (labels_counter[l] + x_total_set[i] * lambda_value)

        self.prior_probability = prior_probability
        self.condition_probability = condition_probability
        self.x_total_set = x_total_set
        return self

    def _predict(self, X):
        """Use probability to predict unlabeled samples.

        X, 2-d numpy.array

        return unpredicted samples' labels.
        """
        n_features = self.n_features
        prior_probability = self.prior_probability
        condition_probability = self.condition_probability
        samples = X.shape[0]
        y_ = []
        labels_set = self.labels_set
        labels_counter = self.labels_counter
        x_total_set = self.x_total_set
        lambda_value = self.lambda_value

        for i in range(samples):
            max_p = 0
            max_label = None
            sample = X[i]
            for l in labels_set:
                p = prior_probability[l]
                for j in range(n_features):
                    try:
                        sub = condition_probability[l][j + 1][sample[j]]
                    except KeyError:
                        sub = lambda_value / (labels_counter[l] + x_total_set[j] * lambda_value)
                    p = p * sub
                if p > max_p:
                    max_p = p
                    max_label = l
            y_.append(max_label)

        return y_
