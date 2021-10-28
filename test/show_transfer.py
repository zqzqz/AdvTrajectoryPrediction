import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

mode = "multi_frame"

models = ["grip", "fqa", "trajectron"]
datasets = ["apolloscape"]
attack_goals = ["ade", "fde", "left", "right", "front", "rear"]


data = OrderedDict()
for model_name in models:
    for dataset_name in datasets:
        print(model_name, dataset_name)
        data[(model_name, dataset_name)] = OrderedDict()
        for T in ["normal/original", "attack/original"] + ["transfer/"+m for m in models]:
            if T == "transfer/"+model_name:
                continue
            data[(model_name, dataset_name)][T] = OrderedDict()
            for attack_goal in attack_goals:
                try:
                    loss_data = np.loadtxt("data/{}_{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, mode, T, attack_goal))
                    ave_loss = -np.mean(loss_data[:,2])
                    if attack_goal in ["ade", "fde"]:
                        ave_loss = ave_loss ** 0.5
                    data[(model_name, dataset_name)][T][attack_goal] = ave_loss
                    print(T, attack_goal, loss_data.shape[0])
                except:
                    data[(model_name, dataset_name)][T][attack_goal] = 0
                    print(T, attack_goal, "Empty")



for model_name in models:
    for dataset_name in datasets:
        print()
        print(','.join([model_name+"-"+dataset_name] + attack_goals))
        for T in data[(model_name, dataset_name)]:
            print(','.join([T] + [str(data[(model_name, dataset_name)][T][attack_goal]) for attack_goal in attack_goals]))

