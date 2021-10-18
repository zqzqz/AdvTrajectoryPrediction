import os
import numpy as np
import json
import copy

from .utils import json_to_data


def add_flags(data):
    delete_obj_ids = []
    for obj_id, obj in data["objects"].items():
        obj["observe_mask"] = (obj["observe_trace"][:,0] > 0).astype(np.int64)
        obj["future_mask"] = (obj["future_trace"][:,0] > 0).astype(np.int64)
        
        if np.sum(obj["observe_mask"]) == 0:
            delete_obj_ids.append(obj_id)
            continue

        if np.min(np.concatenate((obj["observe_mask"], obj["future_mask"]), axis=0)) <= 0:
            obj["complete"] = False
        else:
            obj["complete"] = True

        if np.min(obj["observe_mask"][-1]) <= 0:
            obj["visible"] = False
        else:
            obj["visible"] = True

        obj["static"] = False
        trace = obj["observe_trace"][obj["observe_mask"]>0,:]
        if trace.shape[0] > 1:
            v = trace[1:,:] - trace[:-1,:]
            v = np.sum(v ** 2, axis=1)
            if np.sum(v > 0) < v.shape[0]:
                obj["static"] = True

    for obj_id in delete_obj_ids:
        del data["objects"][obj_id]
    
    return data


# deprecated
def output_data_online_generator(api):
    index = 0
    for input_data in api.data():
        output_data = api.run(input_data)
        yield index, output_data
        index += 1


def data_offline_generator(data_dir, sample=-1):
    file_list = os.listdir(data_dir)
    file_id_list = []
    for filename in file_list:
        name, extension = os.path.splitext(filename)
        if extension == ".json":
            file_id_list.append(int(name))
    file_id_list.sort()
    if sample > 0:
        skip_step = len(file_id_list) // sample
        file_id_list = file_id_list[::skip_step][:sample]

    for file_id in file_id_list:
        file_path = os.path.join(data_dir, "{}.json".format(file_id))
        with open(file_path, "r") as f:
            output_data = json_to_data(json.load(f))
            yield str(file_id), output_data


def data_offline_by_name(data_dir, name):
    file_path = os.path.join(data_dir, "{}.json".format(name))
    with open(file_path, "r") as f:
        output_data = json_to_data(json.load(f))
        return output_data


def input_data_by_attack_step(data, obs_length, pred_length, attack_step):
    input_data = {"objects": {}}
    for key, value in data.items():
        if key != "objects":
            input_data[key] = value
    input_data["observe_length"] = obs_length
    input_data["predict_length"] = pred_length

    k = attack_step
    for _obj_id, obj in data["objects"].items():
        feature = obj["observe_feature"]
        observe_feature = copy.deepcopy(feature[k:k+obs_length,:])
        future_feature = copy.deepcopy(feature[k+obs_length:k+obs_length+pred_length,:])
        trace = obj["observe_trace"]
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
    
    input_data = add_flags(input_data)
    return input_data