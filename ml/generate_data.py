import random
import numpy as np


def generate_random_dataset(n_samples=50, n_features=2, n_labels=2, value_range=10):
    """Randomly generate data set (x, y).

    n_samples, the number of samples
    n_features, the number of features
    n_labels, the number of labels, y ~ range(n_labels)
    range, the range of element, every element of sample
           -range <= xi <= range

    return
    x: 2-d numpy.arrays
    y: 1-d numpy.arrays
    """
    x, y = [], []
    labels = list(range(n_labels))
    _labels = [i * (1 / n_labels) + (1 / n_labels) for i in labels]

    for i in range(n_samples):
        x_sub = []
        for j in range(n_features):
            x_sub.append((2 * value_range * random.random() - value_range))
        x.append(x_sub)

        # Randomly select labels
        r = random.random()
        label = 0
        for k in range(n_labels):
            if r < _labels[k]:
                label = k
                break
        y.append(label)

    x = np.array(x)
    y = np.array(y)

    return x, y


def _parameters_validate(n_samples, range_a, range_b):
    """
    n_samples, bigger than zero.
    range_a, range_b, range_a < range_b
    """
    if n_samples < 0:
        raise Exception("The number of samples cannot smaller than zero.")
    if range_a > range_b:
        raise Exception("The range of sample wrong!")


def generate_random_float_list(n_samples=50, range_a=0, range_b=1):
    """Randomly generate a list of random float numbers.
    range_a range_b, the range of sample range_a ~ range_b

    return  x: 1-d list
    """
    _parameters_validate(n_samples, range_a, range_b)
    float_list = []

    for i in range(n_samples):
        float_list.append((range_b - range_a) * random.random() + range_a)

    return float_list


def generate_random_integer_list(n_samples=50, range_a=0, range_b=10):
    """Randomly generate a list of random integer numbers.

    range_a range_b, the range of sample range_a ~ range_b
                     range_a and range_b are integer

    return  x: 1-d list
    """
    _parameters_validate(n_samples, range_a, range_b)
    integer_list = []

    for i in range(n_samples):
        integer_list.append(random.randint(range_a, range_b))

    return integer_list
