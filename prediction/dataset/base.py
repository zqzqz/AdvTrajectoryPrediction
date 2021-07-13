import numpy as np

class BaseDataset:
    def __init__(self, obs_length, pred_length, time_step):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = obs_length + pred_length
        self.time_step = time_step