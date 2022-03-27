import numpy as np
import torch
import logging
from torch.autograd import Variable
from torch import autograd
import copy
import random

from .attack import BaseAttacker
from .loss import attack_loss
from .constraint import hard_constraint
from prediction.dataset.generate import input_data_by_attack_step

logger = logging.getLogger(__name__)


class GradientAttacker(BaseAttacker):
    def __init__(self, obs_length, pred_length, attack_duration, predictor, iter_num=100, learn_rate=0.1, learn_rate_decay=20, bound=1, physical_bounds={}, smooth=0, seed_num=10):
        super().__init__(obs_length, pred_length, attack_duration, predictor)
        self.iter_num = iter_num
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.bound = bound
        self.physical_bounds = physical_bounds
        self.seed_num = seed_num
        
        self.loss = attack_loss

    def run(self, data, obj_id, **attack_opts):
        try:
            self.predictor.model.eval()
        except:
            pass

        perturbation = {"obj_id": obj_id, "loss": self.loss, "value": {}, "ready_value": {}, "attack_opts": attack_opts}
        
        lr = self.learn_rate
        if attack_opts["type"] in ["ade", "fde"]:
            lr /= 10

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
        torch.manual_seed(1)
        random.seed(1)

        for seed in range(self.seed_num):
            loss_not_improved_iter_cnt = 0

            for _obj_id in perturbation["value"]:
                perturbation["value"][_obj_id] = Variable(torch.rand(self.obs_length+self.attack_duration-1,2).cuda() * 2 * self.bound - self.bound)
                # perturbation["value"][_obj_id] = Variable(torch.zeros(self.obs_length+self.attack_duration-1,2).cuda()).detach()
            
            # opt_Adam = torch.optim.Adam(list(perturbation["value"].values()), lr=self.learn_rate/10 if perturbation["attack_opts"]["type"] in ["ade", "fde"] else self.learn_rate)

            local_best_loss = 0x7fffffff
            for i in range(self.iter_num):
                if loss_not_improved_iter_cnt > 20:
                    break
                total_loss = []
                total_out = {}

                processed_perturbation = {}
                for _obj_id in perturbation["value"]:
                    perturbation["value"][_obj_id].requires_grad = True
                    processed_perturbation[_obj_id] = hard_constraint(data["objects"][_obj_id]["observe_trace"], perturbation["value"][_obj_id], self.bound, self.physical_bounds)

                for k in range(self.attack_duration):
                    # construct perturbation
                    for _obj_id in processed_perturbation:
                        perturbation["ready_value"][_obj_id] = processed_perturbation[_obj_id][k:k+self.obs_length,:]
                    # construct input_data
                    input_data = input_data_by_attack_step(data, self.obs_length, self.pred_length, k)

                    # call predictor
                    output_data, loss = self.predictor.run(input_data, perturbation=perturbation, backward=True)
                    total_out[k] = output_data
                    total_loss.append(loss)

                loss = sum(total_loss)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_perturb = {_obj_id:value.cpu().clone().detach().numpy() for _obj_id, value in processed_perturbation.items()}
                    best_iter = i
                    best_out = total_out

                if loss.item() < local_best_loss:
                    local_best_loss = loss.item()
                    loss_not_improved_iter_cnt = 0
                else:
                    loss_not_improved_iter_cnt += 1

                # opt_Adam.zero_grad()
                # loss.backward()
                # opt_Adam.step()

                perturbation["value"][_obj_id].grad

                total_grad_sum = 0
                for _obj_id in perturbation["value"]:
                    grad = torch.autograd.grad(loss, perturbation["value"][_obj_id], retain_graph=False, create_graph=False, allow_unused=True)[0]
                    perturbation["value"][_obj_id] = perturbation["value"][_obj_id].detach() - lr * grad
                    total_grad_sum += float(torch.sum(torch.absolute(grad)).item())
                if total_grad_sum < 0.1:
                    break

                logger.warn("Seed {} step {} finished -- loss: {}; best loss: {};".format(seed, i, loss, best_loss))

        return {
            "output_data": best_out, 
            "perturbation": best_perturb,
            "loss": best_loss,
            "obj_id": obj_id,
            "attack_opts": attack_opts,
            "attack_length": self.attack_duration
        }
