import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.generate import data_offline_generator
from prediction.attack.constraint import get_physical_constraints, get_metrics
import matplotlib.pyplot as plt

import numpy as np
import copy

# from prediction.dataset.apolloscape import ApolloscapeDataset
# dataset = ApolloscapeDataset(6, 6, 0.5)

# from prediction.dataset.ngsim import NGSIMDataset
# dataset = NGSIMDataset(15, 25, 0.2)

from prediction.dataset.nuscenes import NuScenesDataset
dataset = NuScenesDataset(4, 12, 0.5)


def xy_distribution(dataset):
    for data_type, data_path in dataset.data_path.items():
        xy_point_arrays = []
        for data_file in os.listdir(data_path):
            full_data_file_path = os.path.join(data_path, data_file)
            print(full_data_file_path)
            data = np.genfromtxt(full_data_file_path, delimiter=" ")
            xy_point_arrays.append(np.copy(data[:,3:5]))
        xy_points = np.concatenate(xy_point_arrays, axis=0)

    xy_mean = np.mean(xy_points, axis=0)
    xy_std = np.std(xy_points, axis=0)
    xy_min = np.min(xy_points, axis=0)
    xy_max = np.max(xy_points, axis=0)

    print(xy_mean, xy_std, xy_min, xy_max)


def bounds(dataset):
    vl, a1l, a2l, aa1l, aa2l = [], [], [], [], []
    for data_type, data_path in dataset.data_path.items():
        for data_file in os.listdir(data_path):
            full_data_file_path = os.path.join(data_path, data_file)
            if full_data_file_path.split('.')[-1] != "txt":
                continue
            print(full_data_file_path)
            data = np.genfromtxt(full_data_file_path, delimiter=" ")
            
            obj_ids = np.unique(data[:,1].astype(int))
            for obj_id in obj_ids:
                traj = data[data[:,1].astype(int)==obj_id,:]
                if traj.shape[0] < 4:
                    continue
                if traj[0,2] > 2.5:
                    continue
                if np.sum((traj[1:,0] - traj[:-1,0]).astype(int) == 1) != traj.shape[0]-1:
                    continue

                v, a1, a2, aa1, aa2 = get_metrics(traj[:,3:5])
                vl.append(v)
                a1l.append(a1)
                a2l.append(a2)
                aa1l.append(aa1)
                aa2l.append(aa2)

    flag = 0
    for L in [vl, a1l, a2l, aa1l, aa2l]:
        X = np.concatenate(L, axis=0)
        X[np.isnan(X)] = 0
        X = np.sort(X)
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_min = np.min(X)
        X_max = np.max(X)
        print(X_mean, X_std, X_min, X_max, X_mean+3*X_std)
        data = X
        p, x = np.histogram(data, bins=20)
        x = x[:-1] + (x[1] - x[0])/2
        plt.plot(x, p)
        plt.savefig("{}.png".format(flag))
        flag += 1


bounds(dataset)