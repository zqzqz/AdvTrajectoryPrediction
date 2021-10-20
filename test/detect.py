import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.attack.constraint import get_metrics, hard_constraint
from prediction.dataset.utils import load_data
from test import datasets, models

import numpy as np
import torch
import json
import matplotlib.pyplot as plt


mode = "multi_frame"
dataset_name = "apolloscape"
attack_goals = ["ade", "fde", "left", "right", "front", "rear"]


def get_unit_vector(vectors):
    scale = np.sum(vectors ** 2, axis=1) ** 0.5 + 0.001
    result = np.zeros(vectors.shape)
    result[:,0] = vectors[:,0] / scale
    result[:,1] = vectors[:,1] / scale
    return result


def get_acceleration(trace_array):
    v = trace_array[1:,:] - trace_array[:-1,:]
    a = v[1:,:] - v[:-1,:]
    aa = a[1:,:] - a[:-1,:]

    direction = get_unit_vector(v)
    direction_r = np.concatenate((direction[:,1].reshape(direction.shape[0],1), 
                                -direction[:,0].reshape(direction.shape[0],1)), axis=1)

    long_a = np.sum(direction[:-1,:] * a, axis=1)
    lat_a = np.sum(direction_r[:-1,:] * a, axis=1)

    return long_a, lat_a


def CUSUM(trace_array, opts):
    long_a, lat_a = get_acceleration(trace_array)
    long_opts, lat_opts = opts["long"], opts["lat"]
    
    result = False
    for opts, a in [(long_opts, long_a), (lat_opts, lat_a)]:
        s = 0
        last_m = 0
        for m in a.tolist():
            if m * last_m < 0:
                s = max(0, s + abs(m-last_m)/opts["scale"] - opts["d"])
            last_m = m
        # print(s)
        if s > opts["t"]:
            result = True
    
    return result


def fit_model(dataset_name):
    data_dir = "data/dataset/{}/multi_frame/raw".format(dataset_name)
    observe_length = datasets[dataset_name]["obs_length"]
    physical_bounds = datasets[dataset_name]["instance"].bounds

    if os.path.isfile("detect/traces.npy") and os.path.isfile("detect/labels.npy"):
        traces = np.load("detect/traces.npy")
        labels = np.load("detect/labels.npy")
    else:
        normal_traces = []
        attack_traces = []
        for filename in os.listdir(data_dir):
            input_data = load_data(os.path.join(data_dir, filename))
            for _, obj in input_data["objects"].items():
                if obj["type"] not in [1, 2]:
                    continue
                if np.sum(obj["observe_mask"][:observe_length] > 0) != observe_length:
                    continue
                observe_trace = obj["observe_trace"][:observe_length]
                normal_traces.append(observe_trace)

                perturbation = np.random.rand(observe_length, 2)
                perturbation = hard_constraint(observe_trace, perturbation, 1, physical_bounds)
                attack_trace = observe_trace + perturbation
                attack_traces.append(attack_trace)
        traces = np.array(normal_traces + attack_traces)
        labels = np.array([0 for i in range(len(normal_traces))] + [1 for i in range(len(attack_traces))])
        np.save("detect/traces.npy", traces)
        np.save("detect/labels.npy", labels)

    opts = {
        "long": {
            "d": 0.01,
            "t": 0.1,
            "scale": physical_bounds["linear_a"]
        },
        "lat": {
            "d": 0.01,
            "t": 0.1,
            "scale": physical_bounds["rotate_a"]
        },
    }

    best_score = 0
    best_parameter = None
    for d in [di * 0.02 for di in range(1, 20)]:
        for t in [ti * 0.05 for ti in range(1, 20)]:
            opts["long"]["d"] = d
            opts["lat"]["d"] = d
            opts["long"]["t"] = t
            opts["lat"]["t"] = t
            results = np.zeros(traces.shape[0])
            for i in range(traces.shape[0]):
                results[i] = CUSUM(traces[i], opts)

            TP = np.sum((results > 0) * (labels > 0)) / np.sum(labels > 0)
            FP = np.sum((results > 0) * (labels <= 0)) / np.sum(labels <= 0)
            print(d, t, TP, FP)
            
            score = TP*(1-FP)
            if score > best_score:
                best_score = score
                best_parameter = (d, t)
    
    print(best_parameter, best_score)


def test(model_name, dataset_name, mode, opts):
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode == "multi_frame" else 1
    TP, FP, TN, FN = 0, 0, 0, 0

    for name, obj_id in samples:
        normal_file = os.path.join("data/{}_{}/{}/normal/original/raw/{}-{}.json".format(model_name, dataset_name, mode, name, obj_id))
        normal_data = load_data(normal_file)
        for i in range(attack_length):
            observe_trace = normal_data["output_data"][str(i)]["objects"][str(obj_id)]["observe_trace"]
            result = CUSUM(observe_trace, opts)
            if result:
                FP += 1
            else:
                TN += 1

        for attack_goal in attack_goals:
            attack_file = normal_file = os.path.join("data/{}_{}/{}/attack/original/raw/{}-{}-{}.json".format(model_name, dataset_name, mode, name, obj_id, attack_goal))
            attack_data = load_data(attack_file)
            for i in range(attack_length):
                observe_trace = attack_data["output_data"][str(i)]["objects"][str(obj_id)]["observe_trace"]
                result = CUSUM(observe_trace, opts)
                if result:
                    TP += 1
                else:
                    FN += 1
    
    print(TP, FP, TN, FN)


physical_bounds = datasets[dataset_name]["instance"].bounds
# fit_model(dataset_name)
opts = {
    "long": {
        "d": 0.36,
        "t": 0.15,
        "scale": physical_bounds["linear_a"]
    },
    "lat": {
        "d": 0.36,
        "t": 0.15,
        "scale": physical_bounds["rotate_a"]
    },
}
# with open("detect/opts.json", 'w') as f:
#     json.dump(opts, f)
test("grip", "apolloscape", "multi_frame", opts)

