import numpy as np


def calc_accuracy(output, targets):
    """
    Get index of maximum value in output
    Subtract actual values from estimate values
    Count number of 0 values (correct values)
    Calculate accuracy
    """
    estimates = output.argmax(1)
    actual = targets[0]
    differences = np.array(np.subtract(estimates, actual))
    num_correct = len(differences[np.where(differences == 0)])
    accuracy = 100 * (num_correct / len(actual))
    return accuracy
