from prediction.dataset import BaseDataset
import torch

class Interface():
    def __init__(self, obs_length, pred_length):
        # TODO: make the trace legnth configurable
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = obs_length + pred_length
        self.dataset = None

    def set_dataset(self, dataset):
        assert(isinstance(dataset, BaseDataset))
        assert(dataset.obs_length == self.obs_length)
        assert(dataset.pred_length == self.pred_length)
        self.dataset = dataset
