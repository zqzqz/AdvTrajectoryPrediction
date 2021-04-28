import os
import numpy as np
import json
import copy


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


def output_data_online_generator(api):
    index = 0
    for input_data in api.data():
        output_data = api.run(input_data)
        yield index, output_data
        index += 1


def output_data_offline_generator(data_dir):
    count = len(os.listdir(data_dir))
    for index in range(count):
        file_path = os.path.join(data_dir, "{}.json".format(index))
        with open(file_path, "r") as f:
            output_data = json_to_data(json.load(f))
            yield index, output_data


def output_data_offline_by_index(data_dir, index):
    file_path = os.path.join(data_dir, "{}.json".format(index))
    with open(file_path, "r") as f:
        output_data = json_to_data(json.load(f))
        return output_data


def evaluate_error(IN, online=True):
    ade_list = []
    fde_list = []

    if online:
        generator = output_data_online_generator(IN)
    else:
        generator = output_data_offline_generator(IN)
    
    for _, output_data in generator:
        for _, obj in output_data["objects"].items():
            ade_list.append(ade(obj["future_trace"], obj["predict_trace"]))
            fde_list.append(fde(obj["future_trace"], obj["predict_trace"]))

    return ade_list, fde_list
    