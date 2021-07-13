from prediction.dataset import *

class Interface():
    def __init__(self, obs_length, pred_length):
        # TODO: make the trace legnth configurable
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.dataset = None
