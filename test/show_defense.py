import numpy as np
import torch
import matplotlib.pyplot as plt

model_name = "trajectron"
dataset_name = "apolloscape"
mode = "multi_frame"

# data = {}
for attack_goal in ["ade", "fde", "left", "right", "front", "rear"]:
    for mode1 in ["attack"]:
        losses = []
        for mode2 in ["original", "smooth2"]:
            print(attack_goal, mode1, mode2)
            loss_data = np.loadtxt("data/{}_{}/{}/{}/{}/evaluate/loss_{}.txt".format(model_name, dataset_name, mode, mode1, mode2, attack_goal))
            loss_data[:,2] = -loss_data[:,2]
            if attack_goal in ["ade", "fde"]:
                loss_data[:,2] = loss_data[:,2] ** 0.5
            losses.append(loss_data[loss_data[:,0].argsort()])

        loss_length = min([l.shape[0] for l in losses])
        print(loss_length)
        for l in losses:
            ave_loss = np.mean(l[:loss_length,2])
            print(ave_loss)
            # data[(attack_goal, mode1, mode2)] = ave_loss

        # for i in range(loss_length):
        #     print(int(losses[0][i,0]), [l[i,2] for l in losses], losses[1][i,2] < losses[0][i,2])
        # print(np.sum(losses[1][:loss_length,2] < losses[0][:loss_length,2])/loss_length)
