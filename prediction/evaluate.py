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

def evaluate_error(algorithm_interface):
    api = algorithm_interface
    ade_list = []
    fde_list = []
    for input_data in api.data():
        output_data = api.run(input_data)
        for _, obj in output_data["objects"].items():
            ade_list.append(ade(obj["future_trace"], obj["predict_trace"]))
            fde_list.append(fde(obj["future_trace"], obj["predict_trace"]))
    return ade_list, fde_list