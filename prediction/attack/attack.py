# import

class BaseAttacker():
    def __init__(self, obs_length, pred_length, attack_duration, predictor):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.attack_duration = attack_duration
        self.full_trace_length = self.obs_length + self.pred_length + self.attack_duration - 1
        self.perturb_length = self.obs_length + self.attack_duration - 1

        self.predictor = predictor