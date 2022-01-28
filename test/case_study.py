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


def rotate(traj, theta):
    return np.matmul(
                np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]),
                traj.T).T


def draw():
    ov_obj_id = 3
    ov_case_id = 122
    av_obj_id = 1
    av_case_id = 122
    attack_goal = "right"
    defense = "augment"
    x_min, x_max = 420, 560
    y_min, y_max = 100, 140
    theta = 0
    attack_length = 6

    av_input_data = load_data("data/dataset/apolloscape/multi_frame/raw/{}.json".format(av_case_id))
    ov_input_data = load_data("data/dataset/apolloscape/multi_frame/raw/{}.json".format(ov_case_id))
    normal_result_data = load_data("case_study/grip-apolloscape-{}-{}-{}/normal.json".format(ov_case_id, ov_obj_id, attack_goal))
    attack_result_data = load_data("case_study/grip-apolloscape-{}-{}-{}/attack.json".format(ov_case_id, ov_obj_id, attack_goal))
    defense_result_data = load_data("case_study/grip-apolloscape-{}-{}-{}/attack_{}.json".format(ov_case_id, ov_obj_id, attack_goal, defense))

    print(normal_result_data["loss"]["left"]/6, attack_result_data["loss"]/6, defense_result_data["loss"]/6)

    av = av_input_data["objects"][str(av_obj_id)]["observe_trace"][2:]
    gt = ov_input_data["objects"][str(ov_obj_id)]["observe_trace"]
    perturbation = attack_result_data["perturbation"][str(ov_obj_id)]
    normal_preds = [np.concatenate((normal_result_data["output_data"][str(i)]["objects"][str(ov_obj_id)]["observe_trace"][-1,:].reshape(1,2), normal_result_data["output_data"][str(i)]["objects"][str(ov_obj_id)]["predict_trace"]), axis=0) for i in range(attack_length)]
    attack_preds = [np.concatenate((attack_result_data["output_data"][str(i)]["objects"][str(ov_obj_id)]["observe_trace"][-1,:].reshape(1,2), attack_result_data["output_data"][str(i)]["objects"][str(ov_obj_id)]["predict_trace"]), axis=0) for i in range(attack_length)]
    defense_preds = [np.concatenate((defense_result_data["output_data"][str(i)]["objects"][str(ov_obj_id)]["observe_trace"][-1,:].reshape(1,2), defense_result_data["output_data"][str(i)]["objects"][str(ov_obj_id)]["predict_trace"]), axis=0) for i in range(attack_length)]

    theta = -theta/180*np.pi
    center = np.array([(x_min+x_max)/2, (y_min+y_max)/2])
    av = rotate(av-center, theta)
    gt = rotate(gt-center, theta)
    for i in range(attack_length):
        normal_preds[i] = rotate(normal_preds[i]-center, theta)
        attack_preds[i] = rotate(attack_preds[i]-center, theta)
        defense_preds[i] = rotate(defense_preds[i]-center, theta)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,8))
    for index, pred in enumerate(normal_preds):
        ax[0].plot(pred[:,0], pred[:,1], 'o:', color=[0.4 + 0.09 * index for i in range(3)], label="Prediction at time 0-5" if index == 0 else None)
    ax[0].plot(gt[:,0], gt[:,1], 'ko-', label="Ground truth trajectory")
    ax[0].plot(av[:,0], av[:,1], 'bo-', label="AV trajectory")
    ax[0].legend()
    ax[0].set_title("No attack")
    ax[0].set_xlim([(x_min-x_max)/2, (x_max-x_min)/2])
    ax[0].set_ylim([(y_min-y_max)/2, (y_max-y_min)/2])

    for index, pred in enumerate(attack_preds):
        ax[1].plot(pred[:,0], pred[:,1], 'o:', color=[0.4 + 0.09 * index for i in range(3)], label="Prediction at time 0-5" if index == 0 else None)
    ax[1].plot([gt[10,0]+perturbation[10,0]] + gt[11:,0].tolist(), [gt[10,1]+perturbation[10,1]] + gt[11:,1].tolist(), 'ko-', label="Ground truth trajectory")
    ax[1].plot(gt[:11,0]+perturbation[:11,0], gt[:11,1]+perturbation[:11,1], 'ro-', label="Perturbed trajectory")
    ax[1].plot(av[:,0], av[:,1], 'bo-', label="AV trajectory")
    ax[1].legend()
    ax[1].set_title("After attack")
    ax[1].set_xlim([(x_min-x_max)/2, (x_max-x_min)/2])
    ax[1].set_ylim([(y_min-y_max)/2, (y_max-y_min)/2])

    for index, pred in enumerate(defense_preds):
        ax[2].plot(pred[:,0], pred[:,1], 'o:', color=[0.4 + 0.09 * index for i in range(3)], label="Prediction at time 0-5" if index == 0 else None)
    ax[2].plot([gt[10,0]+perturbation[10,0]] + gt[11:,0].tolist(), [gt[10,1]+perturbation[10,1]] + gt[11:,1].tolist(), 'ko-', label="Ground truth trajectory")
    ax[2].plot(gt[:11,0]+perturbation[:11,0], gt[:11,1]+perturbation[:11,1], 'ro-', label="Perturbed trajectory")
    ax[2].plot(av[:,0], av[:,1], 'bo-', label="AV trajectory")
    ax[2].legend()
    ax[2].set_title("After defense")
    ax[2].set_xlim([(x_min-x_max)/2, (x_max-x_min)/2])
    ax[2].set_ylim([(y_min-y_max)/2, (y_max-y_min)/2])

    plt.savefig("figures/case-study-2.pdf", bbox_inches='tight')


def run():
    case_id = 121
    obj_id = 4
    model_name = "grip"
    dataset_name = "apolloscape"
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    attack_length = datasets[dataset_name]["attack_length"]
    physical_bounds = datasets[dataset_name]["instance"].bounds
    attack_goal = "left"
    output_dir = "case_study/{}-{}-{}-{}-{}-n".format(model_name, dataset_name, case_id, obj_id, attack_goal)

    def task(augment, smooth):
        os.makedirs(output_dir, exist_ok=True)
        tag = "{}{}".format("_augment" if augment else "", "_smooth"+str(smooth) if smooth > 0 else "")
        sys.stdout = open(output_dir+"/log"+tag+".log", "w")
        sys.stderr = open(output_dir+"/log"+tag+".log", "w")

        api = load_model(model_name, "apolloscape", augment=augment, smooth=smooth, models=models)
        attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=10, iter_num=200, physical_bounds=physical_bounds, bound=0.5)

        test_sample(api, DATASET_DIR, case_id, obj_id, attack_length, "{}/normal{}.json".format(output_dir, tag), "{}/normal{}.png".format(output_dir, tag))
        attack_sample(attacker, DATASET_DIR, case_id, obj_id, attack_goal, "{}/attack{}.json".format(output_dir, tag), "{}/attack{}.png".format(output_dir, tag))

    for augment, smooth in zip([False, True, False, False, False, True], [0, 0, 1, 2, 3, 1]):
        task(augment, smooth)


def draw_acceleration():
    fig, axes = plt.subplots(2, 2, figsize=(10,3))

    def draw_one_acceleration(ax, traj, theta=0, xlim=None, ylim=None, color="k", title=""): 
        if xlim is None:
            xlim = [np.min(traj[:,0]), np.max(traj[:,0])]
        if ylim is None:
            ylim = [np.min(traj[:,1]), np.max(traj[:,1])]
        theta = theta/180*np.pi
        center = np.array([np.mean(xlim), np.mean(ylim)])
        traj = rotate(traj-center, theta)
        v = (traj[1:] - traj[:-1])
        a = (v[1:] - v[:-1])
        print(a)
        ax.plot(traj[:,0], traj[:,1], 'o:', color=color)
        for i in range(1, traj.shape[0]-1):
            ax.arrow(traj[i,0], traj[i,1], a[i-1,0], a[i-1,1], width=0.05, head_width=0.4, edgecolor="b")
        xlim = [min(np.min(traj[:,0]), np.min(a[:,0])), max(np.max(traj[:,0]), np.max(a[:,0]))]
        ylim = [min(np.min(traj[:,1]), np.min(a[:,1])), max(np.max(traj[:,1]), np.max(a[:,1]))]
        xlim[0] -= 0.02 * (xlim[1] - xlim[0])
        xlim[1] += 0.02 * (xlim[1] - xlim[0])
        if xlim[1] - xlim[0] < 15:
            xlim[0] -= 5
            xlim[1] += 5
        pad = ((xlim[1]-xlim[0]) / 6 - (ylim[1] - ylim[0])) / 2
        ylim[0] -= pad
        ylim[1] += pad
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')

    draw_one_acceleration(axes[0][0], load_data("data/dataset/apolloscape/multi_frame/raw/0.json")["objects"]["12"]["observe_trace"], color="k", theta=85, title="Normal trajectory")
    draw_one_acceleration(axes[0][1], load_data("data/dataset/apolloscape/multi_frame/raw/296.json")["objects"]["290"]["observe_trace"][3:11], color="k", theta=-15, title="Normal trajectory (falsely classified to be malicious)")
    draw_one_acceleration(axes[1][0], load_data("data/dataset/apolloscape/multi_frame/raw/121.json")["objects"]["4"]["observe_trace"][:9] + load_data("case_study/trajectron-apolloscape-121-4-left/attack.json")["perturbation"]["4"], color="r", theta=-193, title="Adversarial trajectory (deviation to left)")
    draw_one_acceleration(axes[1][1], load_data("data/dataset/apolloscape/multi_frame/raw/28.json")["objects"]["24"]["observe_trace"][:11] + load_data("case_study/grip-apolloscape-28-24-ade/attack.json")["perturbation"]["24"], color="r", theta=-95, title="Adversarial trajectory (maximizing ADE)")

    plt.tight_layout(pad=0.1, h_pad=-2)
    plt.savefig("figures/acceleration.pdf")


draw_acceleration()