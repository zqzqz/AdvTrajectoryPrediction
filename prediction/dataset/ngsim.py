import os
import numpy as np
from scipy.io import loadmat
import random

from .apolloscape import ApolloscapeDataset


class NGSIMDataset(ApolloscapeDataset):
    def __init__(self, obs_length, pred_length, time_step):
        super().__init__(obs_length, pred_length, time_step)

        self.data_dir =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/NGSIM/")
        self.test_data_path = os.path.join(self.data_dir, "prediction_test")
        self.val_data_path = os.path.join(self.data_dir, "prediction_val")
        self.train_data_path = os.path.join(self.data_dir, "prediction_train")
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

        self.default_time_step = 0.1
        self.skip_step = int(self.time_step / self.default_time_step)
        self.feature_dimension = 3