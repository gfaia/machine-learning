import numpy as np
from ml.base import check_two_arrays


def get_distance_metric(dist="euclidean"):
    """Distance between two samples.

    x, 1-d numpy.arrays
    y, 1-d numpy.arrays
    """
    try:
        distance_metric = DISTANCE_MAP[dist]
    except KeyError:
        raise Exception("Distance {0} dose not existed.".format(dist))

    if not callable(distance_metric):
        raise Exception("Distance {0} isn't callable.".format(dist))

    return distance_metric


def euclidean_distance_metric(x, y):
    """Euclidean distance
    formula: distance = sqrt(sum(xi - yi))
    """
    check_two_arrays(x, y)
    return np.sqrt(np.sum(np.square(np.subtract(x, y))))


def manhattan_distance_metric(x, y):
    """Manhattan distance
    formula: distance = sum(abs(xi - yi))
    """
    check_two_arrays(x, y)
    return np.sum(np.abs(np.subtract(x, y)))


DISTANCE_MAP = {"euclidean": euclidean_distance_metric, "manhattan": manhattan_distance_metric}
