import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.attack.constraint import get_metrics, hard_constraint
from prediction.dataset.utils import load_data
from test import datasets, models

import numpy as np
import torch
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import svm


translate = {
    "models": {
        "grip": "GRIP++",
        "fqa": "FQA",
        "trajectron": "Trajectron++"
    },
    "detect": {
        "svm": "SVM model",
        "variance": "Threshold on variance of acceleration"
    }
}

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


def get_delta_accelerate(trace_array):
    v = trace_array[1:,:] - trace_array[:-1,:]
    a = v[1:,:] - v[:-1,:]
    aa = a[1:,:] - a[:-1,:]
    aa = np.sum(aa ** 2, axis=1) ** 0.5
    scale = np.sum(a[:-1,:] ** 2, axis=1) ** 0.5
    return aa / scale


def CUSUM(trace_array, opts):
    long_a, lat_a = get_acceleration(trace_array)
    long_opts, lat_opts = opts["long"], opts["lat"]
    
    result = False
    s = 0
    for opts, a in [(long_opts, long_a), (lat_opts, lat_a)]:
        last_m = 0
        for m in a.tolist():
            if m * last_m < 0:
                s = max(0, s + abs(m-last_m)/opts["scale"] - opts["d"])
            last_m = m
        # print(s)
        if s > opts["t"]:
            result = True
    
    return result


def variance_based_detect(trace_array, thres):
    v = trace_array[1:,:] - trace_array[:-1,:]
    a = v[1:,:] - v[:-1,:]
    mean_a = np.mean(a, axis=0)
    dist_a = a - np.tile(mean_a, (a.shape[0],1))
    var_a = np.sum(np.sum(dist_a ** 2, axis=1)) / a.shape[0]
    # mean_scalar_v = np.sum(v ** 2) / v.shape[0]
    # var_a_rescale = var_a / mean_scalar_v
    return var_a > thres


def preprocess(trace_array):
    v = trace_array[1:,:] - trace_array[:-1,:]
    scalar_v = np.sum(v**2, axis=1) ** 0.5
    scalar_v = scalar_v - np.mean(scalar_v)
    delta_scalar_v = scalar_v[1:] - scalar_v[:-1]
    # delta_scalar_v = (delta_scalar_v - np.min(delta_scalar_v)) / (np.max(delta_scalar_v) - np.min(delta_scalar_v))
    direction = get_unit_vector(v)
    direction = np.arctan(direction[:,0] / (direction[:,1] + 0.001))
    direction = direction - np.mean(direction)
    delta_direction = direction[1:] - direction[:-1]
    # delta_direction = (delta_direction - np.min(delta_direction)) / (np.max(delta_direction) - np.min(delta_direction))
    feature = np.vstack([delta_scalar_v, delta_direction]).reshape(-1)
    return feature 


def evaluate(results, labels):
    report = {}
    report["accuracy"] = np.sum(results == labels) / labels.shape[0]
    report["TP"] = np.sum((results == 1) * (labels == 1))
    report["FP"] = np.sum((results == 1) * (labels == 0))
    report["TN"] = np.sum((results == 0) * (labels == 0))
    report["FN"] = np.sum((results == 0) * (labels == 1))
    report["TPR"] = report["TP"] / (report["TP"] + report["FN"])
    report["FPR"] = report["FP"] / (report["FP"] + report["TN"])
    return report


def generate_train_data(dataset_name="apolloscape", model_name="grip", data_label="train", mode="single_frame"):
    data_dir = "data/dataset/{}/multi_frame/raw".format(dataset_name)
    observe_length = datasets[dataset_name]["obs_length"]
    physical_bounds = datasets[dataset_name]["instance"].bounds
    attack_length = datasets[dataset_name]["attack_length"] if mode == "multi_frame" else 1
    samples = datasets[dataset_name]["samples"]

    normal_traces = []
    attack_traces = []
    for case_id in range(len(os.listdir(data_dir))):
        input_data = load_data(os.path.join(data_dir, "{}.json".format(case_id)))
        for obj_id, obj in input_data["objects"].items():
            if (int(case_id), int(obj_id)) in samples:
                continue
            if obj["type"] not in [1, 2]:
                continue
            if np.sum(obj["observe_mask"][:observe_length] > 0) != observe_length:
                continue
            observe_trace = obj["observe_trace"][:observe_length]
            normal_traces.append(observe_trace)

            for k in range(1):
                perturbation = np.random.rand(observe_length, 2) * 1.5 - 0.75
                perturbation = hard_constraint(observe_trace, perturbation, 1, physical_bounds)
                attack_trace = observe_trace + perturbation
                attack_traces.append(attack_trace)
    
    # data_dir = "data/{}_{}/{}/attack/original/raw/".format(model_name, dataset_name, "multi_frame")
    # for filename in os.listdir(data_dir):
    #     case_id = filename.split('-')[0]
    #     obj_id = filename.split('-')[1]
    #     if (int(case_id), int(obj_id)) in samples:
    #         continue
    #     attack_data = load_data(os.path.join(data_dir, filename))
    #     for i in range(attack_length):
    #         observe_trace = attack_data["output_data"][str(i)]["objects"][attack_data["obj_id"]]["observe_trace"][:observe_length]
    #         attack_traces.append(observe_trace)

    traces = np.array(normal_traces + attack_traces)
    labels = np.array([0 for i in range(len(normal_traces))] + [1 for i in range(len(attack_traces))])
    np.save("detect/traces_{}.npy".format(data_label), traces)
    np.save("detect/labels_{}.npy".format(data_label), labels)


def generate_test_data(dataset_name="apolloscape", model_name="grip", data_label="test", mode="single_frame"):
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode == "multi_frame" else 1

    normal_traces = []
    attack_traces = []
    for name, obj_id in samples:
        normal_file = os.path.join("data/{}_{}/{}/normal/original/raw/{}-{}.json".format(model_name, dataset_name, mode, name, obj_id))
        normal_data = load_data(normal_file)
        observe_traces = []
        for i in range(attack_length):
            observe_trace = normal_data["output_data"][str(i)]["objects"][str(obj_id)]["observe_trace"]
            observe_traces.append(observe_trace)
            normal_traces.append(observe_trace)

        for attack_goal in attack_goals:
            attack_file = normal_file = os.path.join("data/{}_{}/{}/attack/original/raw/{}-{}-{}.json".format(model_name, dataset_name, mode, name, obj_id, attack_goal))
            attack_data = load_data(attack_file)
            perturbation = attack_data["perturbation"][str(obj_id)]
            for i in range(attack_length):
                observe_trace = observe_traces[i] + perturbation[i:i+datasets[dataset_name]["obs_length"]]
                attack_traces.append(observe_trace)

    traces = np.array(normal_traces + attack_traces)
    labels = np.array([0 for i in range(len(normal_traces))] + [1 for i in range(len(attack_traces))])
    np.save("detect/traces_{}.npy".format(data_label), traces)
    np.save("detect/labels_{}.npy".format(data_label), labels)


def fit_model(data_label="", mode="svm"):
    if os.path.isfile("detect/traces_{}.npy".format(data_label)) and os.path.isfile("detect/labels_{}.npy".format(data_label)):
        traces = np.load("detect/traces_{}.npy".format(data_label))
        labels = np.load("detect/labels_{}.npy".format(data_label))
    else:
        raise Exception("data not ready")

    if mode == "svm":
        features = []
        for i in range(traces.shape[0]):
            features.append(preprocess(traces[i]))
        features = np.array(features)

        clf = svm.SVC()
        clf.fit(features, labels)
        pickle.dump(clf, open("detect/model_svm.sav", 'wb'))

        results = clf.predict(features)
        print("Train acc", evaluate(results, labels))

    elif mode == "variance":
        best_acc = 0
        best_t = None
        best_results = None
        for t in [ti * 0.01 for ti in range(1, 100)]:
            results = np.zeros(traces.shape[0])
            for i in range(traces.shape[0]):
                results[i] = variance_based_detect(traces[i], t)
            report = evaluate(results, labels)
            print(t, report)
            acc = report["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_t = t
                best_results = np.copy(results)
        print("Best threshold", best_t, best_acc)
        print(evaluate(best_results, labels))
        with open("detect/variance.txt", 'w') as f:
            f.write(str(best_t))
            
    elif mode == "cusum":
        opts = {
            "d": 0,
            "t": 0
        }

        best_score = 0
        best_parameter = None
        for d in [di * 0.1 for di in range(1, 20)]:
            for t in [ti * 0.1 for ti in range(1, 20)]:
                opts["d"] = d
                opts["t"] = t
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


def test_model(data_label="", mode="svm"):
    if os.path.isfile("detect/traces_{}.npy".format(data_label)) and os.path.isfile("detect/labels_{}.npy".format(data_label)):
        traces = np.load("detect/traces_{}.npy".format(data_label))
        labels = np.load("detect/labels_{}.npy".format(data_label))
    else:
        raise Exception("data not ready")

    if mode == "svm":
        features = []
        for i in range(traces.shape[0]):
            features.append(preprocess(traces[i]))
        features = np.array(features)

        from sklearn import svm
        clf = pickle.load(open("detect/model_svm.sav", 'rb'))
        results = clf.predict(features)

        y_test_pred = clf.decision_function(features)
        test_fpr, test_tpr, te_thresholds = roc_curve(labels, y_test_pred)

    elif mode == "variance":
        with open("detect/variance.txt", 'r') as f:
            t = float(f.read())

        test_fpr, test_tpr, te_thresholds = np.zeros(300), np.zeros(300), np.zeros(300)
        for i, tmp_t in enumerate([0.015 * k for k in range(300)]):
            results = np.zeros(traces.shape[0])
            for k in range(traces.shape[0]):
                results[k] = variance_based_detect(traces[k], tmp_t)
            report = evaluate(results, labels)
            test_fpr[i] = report["FPR"]
            test_tpr[i] = report["TPR"]
            te_thresholds[i] = tmp_t

        results = np.zeros(traces.shape[0])
        for i in range(traces.shape[0]):
            results[i] = variance_based_detect(traces[i], t)

    print(evaluate(results, labels))
    return test_fpr, test_tpr, te_thresholds


def tmp():
    model_name = "grip"
    dataset_name = "apolloscape"
    mode = "single_frame"
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode == "multi_frame" else 1

    for name, obj_id in samples:
        print(name, obj_id)
        normal_traces = []
        attack_traces = []

        normal_file = os.path.join("data/{}_{}/{}/normal/original/raw/{}-{}.json".format(model_name, dataset_name, mode, name, obj_id))
        normal_data = load_data(normal_file)
        observe_traces = []
        for i in range(attack_length):
            observe_trace = normal_data["output_data"][str(i)]["objects"][str(obj_id)]["observe_trace"]
            observe_traces.append(observe_trace)
            normal_traces.append(observe_trace)

        for attack_goal in attack_goals:
            attack_file = normal_file = os.path.join("data/{}_{}/{}/attack/original/raw/{}-{}-{}.json".format(model_name, dataset_name, mode, name, obj_id, attack_goal))
            attack_data = load_data(attack_file)
            perturbation = attack_data["perturbation"][str(obj_id)]
            for i in range(attack_length):
                observe_trace = observe_traces[i] + perturbation[i:i+datasets[dataset_name]["obs_length"]]
                attack_traces.append(observe_trace)
        
        traces = np.array(normal_traces + attack_traces)
        labels = np.array([0 for i in range(len(normal_traces))] + [1 for i in range(len(attack_traces))])

        with open("detect/variance.txt", 'r') as f:
            t = float(f.read())
        results = np.zeros(traces.shape[0])
        for i in range(traces.shape[0]):
            results[i] = variance_based_detect(traces[i], t)

        print(results)


def draw_roc():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    for i, mode in enumerate(["svm", "variance"]):
        for model_name in ["fqa", "grip", "trajectron"]:
            fpr, tpr, thres = test_model(data_label="test_"+model_name, mode=mode)
            np.save(open("detect/{}-{}-{}.npy".format(mode, model_name, "fpr"), "wb"), fpr)
            np.save(open("detect/{}-{}-{}.npy".format(mode, model_name, "tpr"), "wb"), tpr)
            np.save(open("detect/{}-{}-{}.npy".format(mode, model_name, "thres"), "wb"), thres)
            ax[i].plot(fpr, tpr, label="AUC {}={:4f}".format(translate["models"][model_name], auc(fpr, tpr)))
        ax[i].plot([0,1],[0,1],'g--')
        ax[i].legend()
        ax[i].set_xlabel("False Positive Rate")
        ax[i].set_ylabel("True Positive Rate")
        ax[i].set_title(translate["detect"][mode])
    fig.savefig("figures/detect.pdf")



# generate_train_data(model_name="", data_label="train")
# generate_test_data(model_name="grip", data_label="test_grip")
# generate_test_data(model_name="fqa", data_label="test_fqa")
# generate_test_data(model_name="trajectron", data_label="test_trajectron")

# mode = "svm"
# fit_model(data_label="train", mode=mode)
# test_model(data_label="test_grip", mode=mode)
# test_model(data_label="test_fqa", mode=mode)
# test_model(data_label="test_trajectron", mode=mode)


draw_roc()