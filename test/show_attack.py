import numpy as np
import torch
import matplotlib.pyplot as plt

models = ["grip", "fqa", "trajectron", "trajectron_map"]
datasets = ["apolloscape", "ngsim", "nuscenes"]
mode = "multi_frame"

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
                    loss_data = np.loadtxt("data/{}_{}/{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, mode, "normal" if attack_mode == "normal" else "attack", "blackbox" if attack_mode == "blackbox" else "original", attack_goal))
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
                

print(",".join([""] + [attack_goal+",," for attack_goal in attack_goals]))
print(",".join([""] + [attack_mode for attack_mode in attack_modes] * 6))
for model_name in models:
    for dataset_name in datasets:
        if model_name == "trajectron_map" and dataset_name != "nuscenes":
            continue
        print(",".join(["{}-{}".format(model_name, dataset_name)] + [",".join(["{:4f}".format(data[(model_name, dataset_name)][attack_goal][attack_mode]) for attack_mode in attack_modes]) for attack_goal in attack_goals]))
