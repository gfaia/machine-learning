"""Metrics for machine learning."""


def accuracy_score(true, pred):
    """The accuracy of predict, the similarity of two lists."""
    assert len(true) == len(pred)

    length = len(true)
    correct = 0

    for i in range(length):
        if true[i] == pred[i]:
            correct += 1

    return correct / length
