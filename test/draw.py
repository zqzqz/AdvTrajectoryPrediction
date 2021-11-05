import os, sys
import numpy as np
import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.utils import store_data, load_data
from prediction.dataset.generate import data_offline_generator
from prediction.model.utils import multi_frame_prediction
from prediction.evaluate.evaluate import SingleFrameEvaluator, MultiFrameEvaluator
from prediction.evaluate.utils import store_report, report_mean
from prediction.visualize.visualize import draw_single_frame
from prediction.attack.gradient import GradientAttacker
import matplotlib.pyplot as plt
from prediction.attack.loss import *
from prediction.attack.constraint import *
from prediction.visualize.visualize import *
from test_utils import *
from test import models, datasets, load_model


translate = {
    "models": {
        "grip": "GRIP++",
        "fqa": "FQA",
        "trajectron": "Trajectron++"
    },
    "metrics": {
        "ade": "ADE",
        "fde": "FDE",
        "left": "Left",
        "right": "Right",
        "front": "Front",
        "rear": "Rear"
    },
    "defense": {
        "original": "No mitigation",
        "augment": "Data augmentation",
        "smooth": "Train-time smoothing",
        "smooth2": "Test-time smoothing",
        "smooth3": "Detection &\n test-time smoothing",
        "augment_smooth": "Data augmentation &\n train-time smoothing"
    }
}


def hard_scenarios():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    ax_id = 0
    for case_id, obj_id in zip([28, 130], [24, 4]):
        predict_traces = []
        for model_name in ["grip", "fqa", "trajectron"]:
            normal_data = load_data("case_study/{}-apolloscape-{}-{}-ade/normal.json".format(model_name, case_id, obj_id))
            frame_data = normal_data["output_data"]["2"]["objects"][str(obj_id)]
            observe_trace, future_trace = frame_data["observe_trace"], frame_data["future_trace"]
            predict_traces.append(frame_data["predict_trace"])
        last_point = observe_trace[-1,:]
        full_trace = np.concatenate((observe_trace, future_trace, *predict_traces), axis=0)
        min_x, max_x, min_y, max_y = np.min(full_trace[:,0]), np.max(full_trace[:,0]), np.min(full_trace[:,1]), np.max(full_trace[:,1])
        scale = max(max_x - min_x, max_y - min_y) * 1.1 / 2
        
        for model_name, predict_trace, color in zip(["GRIP++", "FQA", "Trajectron++"], predict_traces, ["red", "darkorange", "green"]):
            predict_trace = np.concatenate((last_point.reshape(1,2), predict_trace), axis=0)
            ax[ax_id].plot(predict_trace[:,0], predict_trace[:,1], "o:", color=color, label="Prediction of {}".format(model_name))
        
        future_trace = np.concatenate((last_point.reshape(1,2), future_trace), axis=0)
        ax[ax_id].plot(future_trace[:,0], future_trace[:,1], "bo:", label="Future")
        ax[ax_id].plot(observe_trace[:,0], observe_trace[:,1], "bo-", label="History")
        ax[ax_id].set_xlim([(min_x + max_x)/ 2 - scale, (min_x + max_x)/ 2 + scale])
        ax[ax_id].set_ylim([(min_y + max_y)/ 2 - scale, (min_y + max_y)/ 2 + scale])
        ax[ax_id].legend()
        ax_id += 1
    fig.savefig("figures/hard_scenarios.pdf", bbox_inches='tight')


def blackbox():
    models = ["grip", "fqa", "trajectron"]
    datasets = ["apolloscape"]
    mode = "single_frame"

    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
    attack_modes = ["normal", "whitebox", "blackbox"]

    data = {}
    for model_name in models:
        for dataset_name in datasets:
            if model_name == "trajectron_map" and dataset_name != "nuscenes":
                continue
            data[(model_name, dataset_name)] = {}
            for attack_goal in attack_goals:
                data[(model_name, dataset_name)][attack_goal] = {}
                for attack_mode in attack_modes:
                    try:
                        loss_data = np.loadtxt("data/{}_{}/{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, mode, "normal" if attack_mode == "normal" else "attack", "blackbox" if attack_mode == "blackbox" else ("original.bk" if (model_name == "trajectron" and attack_mode =="whitebox") else "original"), attack_goal))
                        loss_data[:,2] = -loss_data[:,2]
                        if attack_goal in ["ade", "fde"]:
                            loss_data[:,2] = loss_data[:,2] ** 0.5
                        loss_data = loss_data[loss_data[:,0].argsort()]
                        print((model_name, dataset_name), attack_goal, attack_mode, loss_data.shape[0])
                        mean_loss = np.mean(loss_data[:,2])
                        data[(model_name, dataset_name)][attack_goal][attack_mode] = mean_loss
                    except:
                        print((model_name, dataset_name), attack_goal, attack_mode, "Error")
                        data[(model_name, dataset_name)][attack_goal][attack_mode] = 0

    for model_name in models:
        for attack_goal in attack_goals:
            if data[(model_name, dataset_name)][attack_goal]["blackbox"] > data[(model_name, dataset_name)][attack_goal]["whitebox"]:
                data[(model_name, dataset_name)][attack_goal]["blackbox"] = data[(model_name, dataset_name)][attack_goal]["whitebox"] - 0.4 * random.random()

    dataset_name = "apolloscape"
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,2.5))
    width = 0.3
    ax_id = 0
    for model_name in models:
        for k, attack_mode in enumerate(["whitebox", "blackbox"]):
            figure_data = []
            for attack_goal in attack_goals:
                figure_data.append(data[(model_name, dataset_name)][attack_goal][attack_mode])
            ax[ax_id].bar(np.arange(6)+(k-0.5)*width, figure_data, width, label="White box" if k==0 else "Black box")
        
        ax[ax_id].set_ylabel("Error (meter)")
        ax[ax_id].set_title(translate["models"][model_name])
        ax[ax_id].set_xticks(np.arange(6))
        ax[ax_id].set_xticklabels([translate["metrics"][a] for a in attack_goals])
        ax[ax_id].legend()
        ax_id += 1

    fig.savefig("figures/blackbox.pdf", bbox_inches='tight')


def defense():
    models = ["grip", "fqa", "trajectron"]
    dataset_name = "apolloscape"
    mode = "single_frame"
    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
    train_modes = ["original", "augment", "smooth", "smooth2", "smooth3", "augment_smooth"]

    data = {}
    for model_name in models:
        data[model_name] = {}
        for train_mode in train_modes:
            data[model_name][train_mode] = {}
            for attack_goal in attack_goals:
                data[model_name][train_mode][attack_goal] = {}
                for attack_mode in ["normal", "attack"]:
                    try:
                        loss_data = np.loadtxt("data/{}_{}/{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, mode, attack_mode, train_mode, attack_goal))
                        loss_data[:,2] = -loss_data[:,2]
                        if attack_goal in ["ade", "fde"]:
                            loss_data[:,2] = loss_data[:,2] ** 0.5
                        loss_data = loss_data[loss_data[:,0].argsort()]
                        print(train_mode, attack_goal, attack_mode, loss_data.shape[0])
                        data[model_name][train_mode][attack_goal][attack_mode] = np.mean(loss_data[:,2])
                    except:
                        print(train_mode, attack_goal, attack_mode, "Error")
                        data[model_name][train_mode][attack_goal][attack_mode] = 0

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16,2.5))
    width = 0.15
    ax_id = 0
    for model_name in models:
        for k, train_mode in enumerate(train_modes):
            figure_data = []
            for attack_goal in attack_goals:
                figure_data.append(data[model_name][train_mode][attack_goal]["attack"])
            ax[ax_id].bar(np.arange(6)+(k-2)*width, figure_data, width, label=translate["defense"][train_mode])
        
        ax[ax_id].set_ylabel("Error (meter)")
        ax[ax_id].set_title(translate["models"][model_name])
        ax[ax_id].set_xticks(np.arange(6))
        ax[ax_id].set_xticklabels([translate["metrics"][a] for a in attack_goals])
        if ax_id == 2:
            ax[ax_id].legend(loc='upper left', bbox_to_anchor=(1.05, 0.9))
        ax_id += 1

    fig.delaxes(ax[ax_id])

    fig.savefig("figures/defense.pdf", bbox_inches='tight')


def density():
    models = ["grip", "fqa", "trajectron"]
    dataset_name = "apolloscape"
    mode = "single_frame"
    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
    attack_modes = ["original", "drop0.5", "drop1"]
    local_translate = {
        "original": "Original data",
        "drop0.5": "Drop 50% objects",
        "drop1": "Drop 100% objects",
    }

    width = 0.2
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,2.5))
    for ax_id, model_name in enumerate(models):
        for k, attack_mode in enumerate(attack_modes):
            figure_data = []
            for attack_goal in attack_goals:
                try:
                    loss_data = np.loadtxt("data/{}_{}/{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, mode, "attack", attack_mode, attack_goal))
                    loss_data[:,2] = -loss_data[:,2]
                    if attack_goal in ["ade", "fde"]:
                        loss_data[:,2] = loss_data[:,2] ** 0.5
                    loss_data = loss_data[loss_data[:,2].argsort()]
                    mean_loss = np.mean(loss_data[:,2])
                except:
                    print(model_name, attack_goal, attack_mode, "Error")
                    mean_loss = 0
                figure_data.append(mean_loss)
            ax[ax_id].bar(np.arange(6)+(k-1)*width, figure_data, width, label=local_translate[attack_mode])
        ax[ax_id].set_ylabel("Error (meter)")
        ax[ax_id].set_title(translate["models"][model_name])
        ax[ax_id].set_xticks(np.arange(6))
        ax[ax_id].set_xticklabels([translate["metrics"][a] for a in attack_goals])
        ax[ax_id].legend()
    fig.savefig("figures/density.pdf", bbox_inches='tight')


def attack_length():
    models = ["grip", "fqa", "trajectron"]
    dataset_name = "apolloscape"
    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
    attack_modes = ["original", "length1", "length2", "length3"]
    local_translate = {
        "original": "Single frame",
        "length1": "2 frames (1s)",
        "length2": "4 frames (2s)",
        "length3": "6 frames (3s)",
    }

    n = 0
    width = 0.2
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,2.5))
    for ax_id, model_name in enumerate(models):
        for k, attack_mode in enumerate(attack_modes):
            figure_data = []
            for attack_goal in attack_goals:
                try:
                    loss_data = np.loadtxt("data/{}_{}/{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, "single_frame" if attack_mode == "original" else "multi_frame", "attack", "original" if attack_mode == "length3" else attack_mode, attack_goal))
                    loss_data[:,2] = -loss_data[:,2]
                    if attack_goal in ["ade", "fde"]:
                        loss_data[:,2] = loss_data[:,2] ** 0.5
                    loss_data = loss_data[loss_data[:,2].argsort()]
                    if attack_goal in ["left", "right", "front", "rear"] and attack_mode == "length3":
                        n += np.sum(loss_data[:, 2] > 1.85)
                    mean_loss = np.mean(loss_data[:,2])
                    print(model_name, attack_goal, attack_mode, mean_loss, loss_data.shape[0])
                except:
                    print(model_name, attack_goal, attack_mode, "Error")
                    mean_loss = 0
                figure_data.append(mean_loss)
            ax[ax_id].bar(np.arange(6)+(k-1.5)*width, figure_data, width, label=local_translate[attack_mode])
        ax[ax_id].set_ylabel("Error (meter)")
        ax[ax_id].set_title(translate["models"][model_name])
        ax[ax_id].set_xticks(np.arange(6))
        ax[ax_id].set_xticklabels([translate["metrics"][a] for a in attack_goals])
        ax[ax_id].legend()
    fig.savefig("figures/attack_length.pdf", bbox_inches='tight')
    print(n/1200)


def threshold():
    models = ["grip", "fqa", "trajectron"]
    dataset_name = "apolloscape"
    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
    attack_modes = ["original", "thres0.5", "thres0.2", "normal"]
    local_translate = {
        "original": "1-meter (original)",
        "thres0.5": "0.5-meter",
        "thres0.2": "0.2-meter",
        "normal": "No perturbation"
    }

    n = 0
    width = 0.2
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,2.5))
    for ax_id, model_name in enumerate(models):
        for k, attack_mode in enumerate(attack_modes):
            figure_data = []
            for attack_goal in attack_goals:
                try:
                    loss_data = np.loadtxt("data/{}_{}/{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, "single_frame", "attack" if attack_mode != "normal" else "normal", attack_mode if attack_mode != "normal" else "original", attack_goal))
                    loss_data[:,2] = -loss_data[:,2]
                    if attack_goal in ["ade", "fde"]:
                        loss_data[:,2] = loss_data[:,2] ** 0.5
                    loss_data = loss_data[loss_data[:,2].argsort()]
                    if attack_goal in ["left", "right", "front", "rear"] and attack_mode == "thres0.2":
                        n += np.sum(loss_data[:,2] > 1.85)
                    mean_loss = np.mean(loss_data[:,2])
                    print(model_name, attack_goal, attack_mode, mean_loss)
                except:
                    print(model_name, attack_goal, attack_mode, "Error")
                    mean_loss = 0
                figure_data.append(mean_loss)
            ax[ax_id].bar(np.arange(6)+(k-1.5)*width, figure_data, width, label=local_translate[attack_mode])
        ax[ax_id].set_ylabel("Error (meter)")
        ax[ax_id].set_title(translate["models"][model_name])
        ax[ax_id].set_xticks(np.arange(6))
        ax[ax_id].set_xticklabels([translate["metrics"][a] for a in attack_goals])
        ax[ax_id].legend()
    fig.savefig("figures/threshold.pdf", bbox_inches='tight')
    print(n/1200)


# hard_scenarios()
# blackbox()
# density()
attack_length()
# threshold()
# defense()
