"""Utils for data science."""


def train_test_split(X, y, p=0.2):
    """Split the total dataset into train and test parts.
    Args:
        X, Features.
        y, Labels.
        p, Split ratio.
    Return:
        train_X, train_y, test_X, test_y
    """
    n_samples, n_features = X.shape
    splited = int(n_samples * p)

    return X[splited:], y[splited:], X[:splited], y[:splited]
