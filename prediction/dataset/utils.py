import copy
import json
import numpy as np


def get_dict_values(data):
    stack = [(data, [])]
    while len(stack) > 0:
        (d, k) = stack.pop()
        if isinstance(d, dict):
            for key in d:
                if not isinstance(d[key], dict):
                    yield d, key
                else:
                    stack.append((d[key], k + [key]))
        else:
            yield k


def data_to_json(data):
    json_data = copy.deepcopy(data)
    for d, k in get_dict_values(json_data):
        if isinstance(d[k], np.ndarray):
            d[k] = d[k].tolist()
    return json_data


def json_to_data(json_data):
    data = copy.deepcopy(json_data)
    for d, k in get_dict_values(data):
        if isinstance(d[k], list):
            d[k] = np.array(d[k])
    return data


def load_data(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    if isinstance(json_data, dict):
        data = json_to_data(json_data)
    elif isinstance(json_data, list):
        data = []
        for x in json_data:
            data.append(json_to_data(x))
    else:
        raise Exception("Wrong format!")
    return data


def store_data(data, file_path):
    if isinstance(data, dict):
        json_data = data_to_json(data)
    elif isinstance(data, list):
        json_data = []
        for x in data:
            json_data.append(data_to_json(x))
    else:
        raise Exception("Wrong format!")

    with open(file_path, "w") as f:
        json.dump(json_data, f)
        