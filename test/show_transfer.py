import numpy as np
import torch
import matplotlib.pyplot as plt

mode = "multi_frame"
attack_goal = "ade"

models = ["grip", "fqa", "trajectron"]
datasets = ["apolloscape", "ngsim", "nuscenes"]

for model_name in models:
    for dataset_name in datasets:
        print(model_name, dataset_name)
        for T in ["normal_evaluate", "attack_evaluate"] + ["transfer_evaluate_"+m for m in models]:
            if T == "transfer_evaluate_"+model_name:
                continue
            loss_data = np.loadtxt("data/{}_{}/{}/{}/loss_{}.txt".format(model_name, dataset_name, mode, T, attack_goal))
            ave_loss = -np.mean(loss_data[:,2])
            if attack_goal in ["ade", "fde"]:
                ave_loss = ave_loss ** 0.5
            print(T, ave_loss)
        
