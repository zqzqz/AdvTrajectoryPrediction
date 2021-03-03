class DataLoader:
    def __init__(self, dataset, obs_length, pred_length):
        self.dataset = dataset
        self.seq_length = obs_length + pred_length
        self.obs_length = obs_length
        self.pred_length = pred_length