import numpy as np


def ade(trace1, trace2):
    error = 0
    length = min(trace1.shape[0], trace2.shape[0])
    for index in range(length):
        error += np.sum((trace1[index,:] - trace2[index,:]) ** 2)
    return error / length


def fde(trace1, trace2):
    length = min(trace1.shape[0], trace2.shape[0])
    error = np.sum((trace1[length-1,:] - trace2[length-1,:]) ** 2) ** 0.5
    return error