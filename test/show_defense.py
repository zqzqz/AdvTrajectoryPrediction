import numpy as np
import torch
import matplotlib.pyplot as plt


dataset_name = "apolloscape"
mode = "single_frame"
attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
train_modes = ["original", "augment", "smooth", "smooth2", "smooth3", "augment_smooth"]

samples_file = "data/dataset/{}/multi_frame/samples.txt".format(dataset_name)
with open(samples_file, 'r') as f:
    lines = f.readlines()
samples = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in lines]


for model_name in ["trajectron"]:
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
                    data[train_mode][attack_goal][attack_mode] = loss_data[:,2]
                except:
                    print(train_mode, attack_goal, attack_mode, "Error")
                    data[train_mode][attack_goal][attack_mode] = np.zeros(100)

    for attack_goal in attack_goals:
        print(model_name, attack_goal)
        x = np.vstack([data[train_mode][attack_goal]["normal"]] + [data[train_mode][attack_goal]["attack"] for train_mode in train_modes]).T
        for i in range(x.shape[0]):
            print(samples[i], x[i,:])
            

    print(','.join(["{}-{}-{}".format(model_name, dataset_name, mode)] + ["normal {}".format(attack_goal)+","+"attack {}".format(attack_goal) for attack_goal in attack_goals]))
    print('\n'.join([
        ','.join([train_mode] + ["{:4f},{:4f}".format(np.mean(data[train_mode][attack_goal]["normal"]), np.mean(data[train_mode][attack_goal]["attack"])) for attack_goal in attack_goals]) for train_mode in train_modes
        ]))

    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(20,20))
    for row_id, attack_goal in enumerate(attack_goals):
        for col_id, attack_mode in enumerate(["normal", "attack"]):
            bps = []
            for train_mode_id, train_mode in enumerate(train_modes):
                bp = ax[row_id][col_id].boxplot([data[train_mode][attack_goal][attack_mode]], positions=[train_mode_id])
                bps.append(bp)
            ax[row_id][col_id].set_title("{}-{}".format(attack_goal, attack_mode))
            # ax[row_id][col_id].legend([bp["boxes"][0] for bp in bps], [train_mode for train_mode in train_modes])
    fig.savefig("figures/show_defense_{}.png".format(model_name))
