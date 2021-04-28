import copy
import json
import numpy as np

def data_to_json(data):
    json_data = copy.deepcopy(data)
    for obj_id in data["objects"]:
        json_data["objects"][obj_id]["observe_trace"] = data["objects"][obj_id]["observe_trace"].tolist()
        json_data["objects"][obj_id]["future_trace"] = data["objects"][obj_id]["future_trace"].tolist()
        json_data["objects"][obj_id]["predict_trace"] = data["objects"][obj_id]["predict_trace"].tolist()
        json_data["objects"][obj_id]["observe_full_trace"] = data["objects"][obj_id]["observe_full_trace"].tolist()
        json_data["objects"][obj_id]["future_full_trace"] = data["objects"][obj_id]["future_full_trace"].tolist()
    return json_data


def json_to_data(json_data):
    data = copy.deepcopy(json_data)
    for obj_id in json_data["objects"]:
        data["objects"][obj_id]["observe_trace"] = np.array(json_data["objects"][obj_id]["observe_trace"])
        data["objects"][obj_id]["future_trace"] = np.array(json_data["objects"][obj_id]["future_trace"])
        data["objects"][obj_id]["predict_trace"] = np.array(json_data["objects"][obj_id]["predict_trace"])
        data["objects"][obj_id]["observe_full_trace"] = np.array(json_data["objects"][obj_id]["observe_full_trace"])
        data["objects"][obj_id]["future_full_trace"] = np.array(json_data["objects"][obj_id]["future_full_trace"])
    return data


def load_data(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    data = json_to_data(json_data)
    return data


def store_data(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data_to_json(data), f)