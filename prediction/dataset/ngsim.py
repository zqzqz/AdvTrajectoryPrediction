import os
import numpy as np
from scipy.io import loadmat
import random

from .apolloscape import ApolloscapeDataset


class NGSIMDataset(ApolloscapeDataset):
    def __init__(self, obs_length, pred_length, time_step, sample_step=5):
        super().__init__(obs_length, pred_length, time_step, sample_step)

        self.data_dir =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/NGSIM/")
        self.test_data_path = os.path.join(self.data_dir, "prediction_test2.new")
        self.val_data_path = os.path.join(self.data_dir, "prediction_val2.new")
        self.train_data_path = os.path.join(self.data_dir, "prediction_train2.new")
        self.data_path = {
            "train": self.train_data_path,
            "val": self.val_data_path,
            "test": self.test_data_path
        }
        self.data = {
            "train": [],
            "val": [],
            "test": []
        }

        self.default_time_step = 0.2
        self.skip_step = int(self.time_step / self.default_time_step)
        self.feature_dimension = 5

        self.xy_mean = np.array([10.74227619, 289.12354171])
        self.xy_std = np.array([5.97806709, 161.91972043])
        self.xy_min = np.array([0.0677, 2.225])
        self.xy_max = np.array([29.4836, 669.691])

        self.xy_distribution = {
            "mean": self.xy_mean,
            "std": self.xy_std,
            "min": self.xy_min,
            "max": self.xy_max,
        }

        self.bounds = {
            "scalar_v": 4.166,
            "linear_a": 0.291,
            "rotate_a": 0.124,
            "linear_aa": 0.391,
            "rotate_aa": 0.185
        }