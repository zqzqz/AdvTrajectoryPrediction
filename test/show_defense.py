import numpy as np
import torch
import matplotlib.pyplot as plt

model_name = "trajectron"
dataset_name = "apolloscape"
mode = "multi_frame"

attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
train_modes = ["original", "augment", "smooth", "smooth2", "augment_smooth"]

data = {}

for train_mode in train_modes:
    data[train_mode] = {}
    for attack_goal in attack_goals:
        data[train_mode][attack_goal] = {}
        for attack_mode in ["normal", "attack"]:
            try:
                loss_data = np.loadtxt("data/{}_{}/{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, mode, attack_mode, train_mode, attack_goal))
                loss_data[:,2] = -loss_data[:,2]
                if attack_goal in ["ade", "fde"]:
                    loss_data[:,2] = loss_data[:,2] ** 0.5
                loss_data = loss_data[loss_data[:,0].argsort()]
                print(train_mode, attack_goal, attack_mode, loss_data.shape[0])
                mean_loss = np.mean(loss_data[:,2])
                data[train_mode][attack_goal][attack_mode] = mean_loss
            except:
                print(train_mode, attack_goal, attack_mode, "Error")
                data[train_mode][attack_goal][attack_mode] = 0
                

        """
        loss_length = min([l.shape[0] for l in losses])
        print(loss_length)
        for l in losses:
            ave_loss = np.mean(l[:loss_length,2])
            print(ave_loss)

        for i in range(loss_length):
            print(int(losses[0][i,0]), [l[i,2] for l in losses], losses[1][i,2] < losses[0][i,2])
        print(np.sum(losses[1][:loss_length,2] < losses[0][:loss_length,2])/loss_length)
        """

print(','.join(["{}-{}-{}".format(model_name, dataset_name, mode)] + ["normal {}".format(attack_goal)+","+"attack {}".format(attack_goal) for attack_goal in attack_goals]))
print('\n'.join([
    ','.join([train_mode] + ["{:4f},{:4f}".format(data[train_mode][attack_goal]["normal"], data[train_mode][attack_goal]["attack"]) for attack_goal in attack_goals]) for train_mode in train_modes
    ]))
