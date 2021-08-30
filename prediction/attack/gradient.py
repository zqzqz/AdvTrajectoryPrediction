import numpy as np
import torch
import logging
from torch.autograd import Variable
from torch import autograd
import copy

from .attack import BaseAttacker
from .loss import attack_loss
from prediction.dataset.generate import input_data_by_attack_step

logger = logging.getLogger(__name__)


class GradientAttacker(BaseAttacker):
    def __init__(self, obs_length, pred_length, attack_duration, predictor, iter_num=100, learn_rate=0.1, bound=0.5, seed_num=10):
        super().__init__(obs_length, pred_length, attack_duration, predictor)
        self.iter_num = iter_num
        self.learn_rate = learn_rate
        self.bound = bound
        self.seed_num = seed_num
        
        self.loss = attack_loss

    def run(self, data, obj_id, **attack_opts):
        try:
            self.predictor.model.train()
        except:
            pass

        perturbation = {"obj_id": obj_id, "loss": self.loss, "value": {}, "ready_value": {}, "attack_opts": attack_opts}
        
        if "mode" in attack_opts:
            mode = attack_opts["mode"]
        else:
            mode = "single"

        if mode == "single":
            perturbation["value"][obj_id] = None
            perturbation["ready_value"][obj_id] = None
        elif mode == "all":
            for _obj_id in data["objects"]:
                perturbation["value"][_obj_id] = None
                perturbation["ready_value"][_obj_id] = None
        elif mode == "select":
            raise NotImplementedError()

        attack_opts["loss"] = self.loss.__name__

        best_iter = None
        best_loss = 0x7fffffff
        best_out = None
        best_perturb = None

        for seed in range(self.seed_num):

            for _obj_id in perturbation["value"]:
                perturbation["value"][_obj_id] = Variable(torch.rand(self.obs_length+self.attack_duration-1,2).cuda() * 2 * self.bound - self.bound, requires_grad=True)
            opt_Adam = torch.optim.Adam(list(perturbation["value"].values()), lr=self.learn_rate)

            for i in range(self.iter_num):
                total_loss = []
                total_out = {}

                for k in range(self.attack_duration):
                    # construct perturbation
                    for _obj_id in perturbation["value"]:
                        perturbation["ready_value"][_obj_id] = torch.clamp(perturbation["value"][_obj_id][k:k+self.obs_length,:], min=-self.bound, max=self.bound)
                    # construct input_data
                    input_data = input_data_by_attack_step(data, self.obs_length, self.pred_length, k)

                    # call predictor
                    output_data, loss = self.predictor.run(input_data, perturbation=perturbation, backward=True)
                    total_out[k] = output_data
                    total_loss.append(loss)

                loss = sum(total_loss)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_perturb = {_obj_id:torch.clamp(value, min=-self.bound, max=self.bound).cpu().clone().detach().numpy() for _obj_id, value in perturbation["ready_value"].items()}
                    best_iter = i
                    best_out = total_out

                opt_Adam.zero_grad()
                loss.backward()
                opt_Adam.step()

                logger.warn("Seed {} step {} finished -- loss: {}; best loss: {};".format(seed, i, loss, best_loss))

        return {
            "output_data": best_out, 
            "perturbation": best_perturb,
            "loss": best_loss,
            "obj_id": obj_id,
            "attack_opts": attack_opts,
            "attack_length": self.attack_duration
        }