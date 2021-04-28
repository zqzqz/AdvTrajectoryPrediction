from prediction.dataset import *

class Interface():
    def __init__(self, dataset_name, obs_length, pred_length):
        # TODO: make the trace legnth configurable
        self.obs_length = obs_length
        self.pred_length = pred_length

        if dataset_name == "apolloscape":
            self.dataset = ApolloscapeDataset(self.obs_length, self.pred_length)
        else:
            self.dataset = None
