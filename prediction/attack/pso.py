import numpy as np
import logging
import copy
import pyswarms as ps
import torch

from .attack import BaseAttacker
from .loss import attack_loss
from .constraint import hard_constraint
from prediction.dataset.generate import input_data_by_attack_step

logger = logging.getLogger(__name__)


def objective(x, data, obj_id, predictor, loss_func, attack_opts):
    obs_length = predictor.obs_length
    pred_length = predictor.pred_length
    attack_trace_length = x.shape[1] // 2
    attack_duration = attack_trace_length-obs_length+1
    N = x.shape[0]

    loss = np.zeros(x.shape[0])

    for n in range(N):
        perturbation_value = x[n,:].reshape((attack_trace_length, 2))
        perturbation_tensor = torch.from_numpy(perturbation_value).cuda()
        observe_trace_array = data["objects"][obj_id]["observe_trace"]
        ready_perturbation_tensor = hard_constraint(observe_trace_array, perturbation_tensor, attack_opts["bound"], attack_opts["physical_bounds"])
        for k in range(attack_duration):
            # perturbation structure
            perturbation = {"obj_id": obj_id, "loss": loss_func, "ready_value": {obj_id: ready_perturbation_tensor[k:k+obs_length,:]}, "attack_opts": attack_opts}
            # input data
            input_data = input_data_by_attack_step(data, obs_length, pred_length, k)
            # call predictor
            _, _loss = predictor.run(input_data, perturbation=perturbation, backward=False)
            loss[n] += _loss
    
    return loss


class PSOAttacker(BaseAttacker):
    def __init__(self, obs_length, pred_length, attack_duration, predictor, n_particles=10, iter_num=100, c1=0.5, c2=0.3, w=1.0, bound=1, physical_bounds={}):
        super().__init__(obs_length, pred_length, attack_duration, predictor)

        self.iter_num = iter_num
        self.bound = bound
        self.physical_bounds = physical_bounds
        self.perturb_length = obs_length + attack_duration - 1
        self.loss = attack_loss
        self.options = {'c1': c1, 'c2': c2, 'w': w}
        self.n_particles = n_particles
        self.bound = bound

    def run(self, data, obj_id, **attack_opts):
        try:
            self.predictor.model.eval()
        except:
            pass

        attack_opts["bound"] = self.bound
        attack_opts["physical_bounds"] = self.physical_bounds
        

        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=self.perturb_length * 2, options=self.options, bounds=(-np.ones(self.perturb_length * 2) * self.bound, np.ones(self.perturb_length * 2) * self.bound), center=np.zeros(self.perturb_length * 2))
        best_loss, best_perturb = optimizer.optimize(objective, iters=self.iter_num, data=data, obj_id=str(obj_id), predictor=self.predictor, loss_func=self.loss, attack_opts=attack_opts)

        best_perturb = best_perturb.reshape((self.perturb_length, 2))
        
        # repeat the prediction once to get the best output data
        best_out = {}
        perturbation_tensor = torch.from_numpy(best_perturb).cuda()
        observe_trace_array = data["objects"][obj_id]["observe_trace"]
        ready_perturbation_tensor = hard_constraint(observe_trace_array, perturbation_tensor, attack_opts["bound"], attack_opts["physical_bounds"])
        for k in range(self.attack_duration):
            # perturbation structure
            perturbation = {"obj_id": obj_id, "loss": self.loss, "ready_value": {obj_id: ready_perturbation_tensor[k:k+self.obs_length,:]}, "attack_opts": attack_opts}
            # input data
            input_data = input_data_by_attack_step(data, self.obs_length, self.pred_length, k)
            # call predictor
            output_data, _ = self.predictor.run(input_data, perturbation=perturbation, backward=False)
            # record results
            best_out[str(k)] = output_data

        return {
            "output_data": best_out, 
            "perturbation": {obj_id: best_perturb},
            "loss": best_loss,
            "obj_id": obj_id,
            "attack_opts": attack_opts,
            "attack_length": self.attack_duration
        }
