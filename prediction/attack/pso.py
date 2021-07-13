import numpy as np
import logging
import copy
import pyswarms as ps

from .attack import BaseAttacker
from .loss import *
from prediction.utils import get_input_data

logger = logging.getLogger(__name__)


def objective(x, data, obj_id, predictor, loss):
    obs_length = predictor.obs_length
    pred_length = predictor.pred_length
    attack_trace_length = x.shape[1] // 2
    attack_duration = attack_trace_length-obs_length+1
    N = x.shape[0]

    loss = np.zeros(x.shape[0])

    for n in range(N):
        perturbation_value = x[n,:].reshape((attack_trace_length, 2))
        for k in range(attack_duration):
            # perturbation structure
            perturbation = {"obj_id": obj_id, "loss": loss, "ready_value": perturbation_value[k:k+obs_length,:]}
            # input data
            input_data = get_input_data(data, obj_id, obs_length, pred_length, k)
            # call predictor
            _, _loss = predictor.run(input_data, perturbation=perturbation, mode="eval")
            loss[n] += _loss
    
    return loss


class PSOAttacker(BaseAttacker):
    def __init__(self, obs_length, pred_length, attack_duration, predictor, loss=None, bound=0.5, n_particles=10, iter_num=1000, c1=0.5, c2=0.3, w=1.0):
        super().__init__(obs_length, pred_length, attack_duration, predictor)

        self.iter_num = iter_num
        self.bound = bound

        self.optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=self.perturb_length * 2, options={'c1': c1, 'c2': c2, 'w': w}, bounds=(-bound, bound), center=np.zeros(self.perturb_length * 2))

        self.loss = loss
        if self.loss is None:
            self.loss = ade_loss

    def run(self, data, obj_id):
        best_loss, best_perturb = self.optimizer.optimize(objective, iters=self.iter_num, data=data, obj_id=obj_id, predictor=self.predictor, loss=self.loss)

        best_perturb = best_perturb.reshape((self.perturb_length, 2))
        
        # repeat the prediction once to get the best output data
        best_out = []
        for k in range(self.attack_duration):
            # perturbation structure
            perturbation = {"obj_id": obj_id, "loss": self.loss, "ready_value": torch.clamp(best_perturb[k:k+self.obs_length,:], min=-self.bound, max=self.bound)}
            # input data
            input_data = get_input_data(data, obj_id, self.obs_length, self.pred_length, k)
            # call predictor
            output_data, _ = self.predictor.run(input_data, perturbation=perturbation, mode="train")
            # record results
            best_out.append(output_data)

        return best_out, best_perturb, best_loss