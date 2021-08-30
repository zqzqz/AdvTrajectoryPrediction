import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from prediction.dataset.ngsim import NGSIMDataset
# from prediction.dataset.apolloscape import ApolloscapeDataset
from prediction.dataset.nuscenes import NuScenesDataset

import numpy as np
import copy

obs_length = 15
pred_length = 25
attack_length = 6
time_step = 0.2

# dataset = NGSIMDataset(obs_length, pred_length, time_step, sample_step=40)
# dataset = ApolloscapeDataset(obs_length, pred_length, time_step)
dataset = NuScenesDataset(obs_length, pred_length, time_step)


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