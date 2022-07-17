import numpy as np
from ml.base import LinearModel


class LinearRegression(LinearModel):
    """Linear regression algorithm,
    Loss function =  || X * w - Y || ^ 2
    """

    def __init__(self, learn_rate=0.1, init_coef=0.0, init_bias=0.0, max_iter=100):

        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.bias = init_bias
        self.coef = init_coef

    @staticmethod
    def linear_func(x, bias, coef):
        """Linear function: y = w * x + b."""
        return bias + np.dot(x, coef)

    def _train(self, X, y):
        """Minimize the cost function by using gradient descent algorithm."""
        self.n_samples, self.n_features = n_samples, n_features = X.shape
        _bias, _coef = self.bias, np.array([self.coef] * n_features)
        max_iter = self.max_iter
        learn_rate = self.learn_rate

        for i in range(max_iter):
            for j in range(n_samples):
                _bias += learn_rate * (y[j] - self.linear_func(X[j], _bias, _coef))
                _coef += learn_rate * (y[j] - self.linear_func(X[j], _bias, _coef)) * X[j]

        self.bias, self.coef = _bias, _coef
        return self

    def _predict(self, X):
        """Predict samples."""
        _bias, _coef = self.bias, self.coef
        return [self.linear_func(x, _bias, _coef) for x in X]
