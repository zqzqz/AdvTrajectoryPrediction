from prediction.base.dataloader import DataLoader
import numpy as np

class BaselineDataLoader(DataLoader):
    def __init__(self, dataset, obs_length=4, pred_length=6):
        super().__init__(dataset, obs_length, pred_length)