import os
import numpy as np
import json
import copy

from .utils import json_to_data


def output_data_online_generator(api):
    index = 0
    for input_data in api.data():
        output_data = api.run(input_data)
        yield index, output_data
        index += 1


def output_data_offline_generator(data_dir):
    for filename in os.listdir(data_dir):
        name, extension = os.path.splitext(filename)
        if extension != ".json":
            continue
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r") as f:
            output_data = json_to_data(json.load(f))
            yield name, output_data


def output_data_offline_by_name(data_dir, name):
    file_path = os.path.join(data_dir, "{}.json".format(name))
    with open(file_path, "r") as f:
        output_data = json_to_data(json.load(f))
        return output_data


def input_data_by_attack_step(data, obs_length, pred_length, attack_step):
    input_data = {
        "observe_length": obs_length,
        "predict_length": pred_length,
        "objects": {}
    }
    k = attack_step
    for _obj_id, obj in data["objects"].items():
        feature = np.concatenate((obj["observe_feature"], obj["future_feature"]), axis=0)
        observe_feature = copy.deepcopy(feature[k:k+obs_length,:])
        future_feature = copy.deepcopy(feature[k+obs_length:k+obs_length+pred_length,:])
        trace = np.concatenate((obj["observe_trace"], obj["future_trace"]), axis=0)
        observe_trace = copy.deepcopy(trace[k:k+obs_length,:])
        future_trace = copy.deepcopy(trace[k+obs_length:k+obs_length+pred_length,:])
        new_obj = {
            "type": int(obj["type"]),
            "observe_feature": observe_feature,
            "future_feature": future_feature,
            "observe_trace": observe_trace,
            "future_trace": future_trace,
            "predict_trace": np.zeros((pred_length,2)),
        }
        input_data["objects"][_obj_id] = new_obj

    return input_data