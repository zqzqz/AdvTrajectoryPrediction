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
from main import my_load_model, compute_RMSE

logger = logging.getLogger(__name__)


class GRIPInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP/trained_models/model_epoch_0099.pt')):
        super().__init__(obs_length, pred_length)

        self.dataloader = GRIPDataLoader(
            self.obs_length, self.pred_length
        )

        self.dev = 'cuda:0'
        if pre_load_model is not None:
            self.model = self.load_model(self.default_model(), pre_load_model)
        else:
            self.model = None

    def set_dataset(self, dataset):
        assert(dataset.obs_length == self.obs_length)
        assert(dataset.pred_length == self.pred_length)
        self.dataset = dataset

    def default_model(self):
        graph_args={'max_hop':2, 'num_node':120}
        model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
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
        else:
            self.model.train()
        _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask = self.dataloader.preprocess(input_data, perturbation)
        predicted = self.model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
        output_data, loss = self.dataloader.postprocess(input_data, perturbation, predicted, _ori_data, mean_xy, rescale_xy, no_norm_loc_data)

        if loss is None:
            return output_data
        else:
            return output_data, loss

    def train(self, total_epoch=200, batch_size=64, save_dir=""):
        assert(self.dataset is not None)

        model = self.default_model()
        optimizer = optim.Adam(
            [{'params':model.parameters()},],) # lr = 0.0001)
            
        for now_epoch in range(total_epoch):
            iteration = 0
            for batch in self.dataset.train_data_generator(batch_size=batch_size, enable_batch=True):
                iteration += 1

                batch_processed_data = []
                for input_data in batch:
                    _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask = self.dataloader.preprocess(input_data, None)
                    batch_processed_data.append([_input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask])
                
                _input_data = torch.cat([i[0] for i in batch_processed_data], axis=0)
                A = torch.cat([i[1] for i in batch_processed_data], axis=0)
                _ori_data = np.concatenate([i[2] for i in batch_processed_data], axis=0)
                mean_xy = np.concatenate([i[3] for i in batch_processed_data], axis=0)
                rescale_xy = torch.cat([i[4] for i in batch_processed_data], axis=0)
                no_norm_loc_data = torch.cat([i[5] for i in batch_processed_data], axis=0)
                output_loc_GT = torch.cat([i[6] for i in batch_processed_data], axis=0)
                output_mask = torch.cat([i[7] for i in batch_processed_data], axis=0)
                
                predicted = model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)

                overall_sum_time, overall_num, _ = compute_RMSE(predicted, output_loc_GT, output_mask, pra_error_order=1)
                total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num), torch.ones(1,).to(self.dev)) #(1,)

                now_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                logger.warn('|{}|Epoch:{:>4}/{:>4}|\tIteration:{:>5}|\tLoss:{:.8f}|lr: {}|'.format(datetime.now(), now_epoch, total_epoch, iteration, total_loss.data.item(), now_lr))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            
            save_path = os.path.join(save_dir, "model_epoch_{:04}.pt".format(now_epoch))
            self.save_model(model, save_path)
