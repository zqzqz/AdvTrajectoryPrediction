from prediction.base.interface import Interface

class BaselineInterface:
    def __init__(self, dataset_name, obs_length, pred_length):
        super().__init__(dataset_name, obs_length, pred_length)