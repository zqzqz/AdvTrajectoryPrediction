import os, sys
import numpy as np
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


def draw():
    input_data = load_data("data/dataset/apolloscape/multi_frame/raw/121.json")
    input_data2 = load_data("data/dataset/apolloscape/multi_frame/raw/124.json")
    normal_result_data = load_data("data/grip_apolloscape/multi_frame/normal/original/raw/121-4.json")
    attack_result_data = load_data("data/grip_apolloscape/multi_frame/attack/original/raw/121-4-left.json")

    av = input_data2["objects"]["1"]["observe_trace"]
    gt = input_data["objects"]["4"]["observe_trace"]
    perturbation = attack_result_data["perturbation"]["4"]
    normal_preds = [np.concatenate((normal_result_data["output_data"][str(i)]["objects"]["4"]["observe_trace"][-1,:].reshape(1,2), normal_result_data["output_data"][str(i)]["objects"]["4"]["predict_trace"]), axis=0) for i in range(6)]
    attack_preds = [np.concatenate((attack_result_data["output_data"][str(i)]["objects"]["4"]["observe_trace"][-1,:].reshape(1,2), attack_result_data["output_data"][str(i)]["objects"]["4"]["predict_trace"]), axis=0) for i in range(6)]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
    for index, pred in enumerate(normal_preds):
        ax[0].plot(pred[:,0], pred[:,1], 'o:', color=[0.4 + 0.09 * index for i in range(3)], label="Prediction at time 0-5" if index == 0 else None)
    ax[0].plot(gt[:,0], gt[:,1], 'ko-', label="Ground truth trajectory")
    ax[0].plot(av[:,0], av[:,1], 'bo-', label="AV trajectory")
    ax[0].legend()
    ax[0].set_title("Prediction w/o perturbation, no lane changing.")
    for index, pred in enumerate(attack_preds):
        ax[1].plot(pred[:,0], pred[:,1], 'o:', color=[0.4 + 0.09 * index for i in range(3)], label="Prediction at time 0-5" if index == 0 else None)
    ax[1].plot([gt[10,0]+perturbation[10,0]] + gt[11:,0].tolist(), [gt[10,1]+perturbation[10,1]] + gt[11:,1].tolist(), 'ko-', label="Ground truth trajectory")
    ax[1].plot(gt[:11,0]+perturbation[:11,0], gt[:11,1]+perturbation[:11,1], 'ro-', label="Perturbed trajectory")
    ax[1].plot(av[:,0], av[:,1], 'bo-', label="AV trajectory")
    ax[1].legend()
    ax[1].set_title("Prediction w/ perturbation towards left deviation, fake lane changing.")

    plt.savefig("case-study.pdf")


case_id = 121
obj_id = 4
model_name = "grip"
dataset_name = "apolloscape"
DATASET_DIR = "data/dataset/apolloscape"
attack_length = datasets[dataset_name]["attack_length"]
physical_bounds = datasets[dataset_name]["instance"].bounds
attack_goal = "ade"
augment = False
smooth = True
output_dir = "case_study"

api = load_model(model_name, dataset_name, augment=augment, smooth=smooth, models=models)
attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=10, iter_num=100, physical_bounds=physical_bounds)

tag = "{}{}".format("_augment" if augment else "", "_smooth" if smooth else "")
test_sample(api, DATASET_DIR, case_id, obj_id, attack_length, "{}/normal{}.json".format(output_dir, tag), "{}/normal{}.png".format(output_dir, tag))
attack_sample(attacker, DATASET_DIR, case_id, obj_id, attack_goal, "{}/attack{}.json".format(output_dir, tag), "{}/attack{}.png".format(output_dir, tag))