class DataLoader:
    def __init__(self, obs_length, pred_length):
        self.seq_length = obs_length + pred_length
        self.obs_length = obs_length
        self.pred_length = pred_length