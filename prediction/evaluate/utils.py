import numpy as np
import json


def ade(trace1, trace2):
    error = 0
    length = min(trace1.shape[0], trace2.shape[0])
    for index in range(length):
        error += np.sum((trace1[index,:] - trace2[index,:]) ** 2) ** 0.5
    return error / length


def fde(trace1, trace2):
    length = min(trace1.shape[0], trace2.shape[0])
    error = np.sum((trace1[length-1,:] - trace2[length-1,:]) ** 2) ** 0.5
    return error


def store_report(report, filepath):
    with open(filepath, 'w') as f:
        json.dump(report, f)


def report_mean(metric_report):
    report_data = []
    for _, d in metric_report.items():
        report_data += d
    return np.mean(np.array(report_data))