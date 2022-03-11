import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP'))
import torch
import torch.optim as optim
import pickle
import time
import warnings
import itertools
import numpy as np
import logging
import copy
from datetime import datetime

from .dataloader import GRIPDataLoader
from prediction.model.base.interface import Interface
from model import Model
from main import my_load_model, compute_RMSE, display_result

logger = logging.getLogger(__name__)


class GRIPInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=None, max_hop=2, num_node=120, in_channels=4, rescale=[1,1], smooth=0, dataset=None):
        super().__init__(obs_length, pred_length)
        self.graph_args = {'max_hop':max_hop, 'num_node':num_node}
        self.dataloader = GRIPDataLoader(
            self.obs_length, self.pred_length, graph_args=self.graph_args, dataset=dataset
        )

        self.dev = 'cuda:0'
        if pre_load_model is not None:
            self.model = self.load_model(self.default_model(in_channels=in_channels), pre_load_model)
        else:
            self.model = None

        self.rescale = rescale
        self.smooth = smooth
        self.dataset = dataset

    def default_model(self, in_channels=4):
        model = Model(in_channels=in_channels, graph_args=self.graph_args, edge_importance_weighting=True)
        model.to(self.dev)
        return model

    def load_model(self, model, model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
        logger.warn('Successfull loaded from {}'.format(model_path))
        return model

    def save_model(self, model, model_path):
        torch.save(
            {
                'xin_graph_seq2seq_model': model.state_dict(),
            }, 
            model_path)
        logger.warn("Model saved to {}".format(model_path))

    def run(self, input_data, perturbation=None, backward=False):
        assert(self.model is not None)

        if not backward:
            self.model.eval()

        _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask, obj_index = self.dataloader.preprocess(input_data, perturbation, smooth=self.smooth, rescale_x=self.rescale[0], rescale_y=self.rescale[1])
        predicted = self.model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
        output_data, loss = self.dataloader.postprocess(input_data, perturbation, predicted, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, obj_index)

        if loss is None:
            return output_data
        else:
            return output_data, loss
