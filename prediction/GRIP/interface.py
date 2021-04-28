import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP'))
import pickle
import time
import warnings
import itertools
import numpy as np
import torch
from torch.autograd import Variable
from torch import autograd
import logging
import copy

from .dataloader import GRIPDataLoader
from prediction.base.interface import Interface
from model import Model
from main import my_load_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GRIPInterface(Interface):
    def __init__(self, dataset_name, obs_length, pred_length):
        super().__init__(dataset_name, obs_length, pred_length)

        self.dataloader = GRIPDataLoader(
            self.dataset, self.obs_length, self.pred_length
        )

        self.dev = 'cuda:0' 
        graph_args={'max_hop':2, 'num_node':120}
        model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
        model.to(self.dev)
        self.model = my_load_model(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP/trained_models/model_epoch_0099.pt'))

    def data(self):
        return self.dataloader.generate_data()

    def run(self, input_data):
        self.model.eval()
        _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data = self.dataloader.preprocess(input_data, None)
        predicted = self.model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None) # (N, C, T, V)=(N, 2, 6, 120)
        output_data, _ = self.dataloader.postprocess(input_data, None, predicted, _ori_data, mean_xy, rescale_xy, no_norm_loc_data)
        return output_data

    def adv(self, data, perturbation, iter_num=100, learn_rate=0.01):
        self.model.train()
        perturbation["value"] = Variable(torch.randn(self.obs_length,2).cuda() * 0.5, requires_grad=True)
        print(perturbation["value"])
        opt_Adam = torch.optim.Adam([perturbation["value"]], lr=learn_rate)

        best_iter = None
        best_loss = 0x7fffffff
        best_out = None
        best_perturb = None

        for i in range(iter_num):
            input_data = copy.deepcopy(data)
            perturbation["clamp_value"] = torch.clamp(perturbation["value"], min=-0.5, max=0.5)

            _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data = self.dataloader.preprocess(input_data, perturbation)
            predicted = self.model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None) # (N, C, T, V)=(N, 2, 6, 120)
            output_data, loss = self.dataloader.postprocess(input_data, perturbation, predicted, _ori_data, mean_xy, rescale_xy, no_norm_loc_data)

            if loss < best_loss:
                best_loss = loss
                best_perturb = perturbation["clamp_value"].detach().cpu().clone().numpy()
                best_iter = i
                best_out = copy.deepcopy(output_data)
            
            opt_Adam.zero_grad()
            loss.backward()
            opt_Adam.step()

            logger.warn("Adv train step {} finished -- loss: {}; best loss: {};".format(i, loss, best_loss))

        return best_out, best_perturb, best_iter, best_loss

