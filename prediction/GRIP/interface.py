import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP'))
import pickle
import time
import warnings
import itertools
import numpy as np
import torch

from .dataloader import GRIPDataLoader
from prediction.base.interface import Interface
from model import Model
from main import my_load_model

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
        model.eval()
        self.model = my_load_model(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP/trained_models/model_epoch_0099.pt'))

    def data(self):
        return self.dataloader.generate_data()

    def run(self, input_data):    
        rescale_xy = torch.ones((1,2,1,1)).to(self.dev)
        rescale_xy[:,0] = 1.
        rescale_xy[:,1] = 1.
        _ori_data, A, mean_xy = self.dataloader.preprocess(input_data)
        ori_data = torch.from_numpy(_ori_data)
        A = torch.from_numpy(A)

        feature_id = [3, 4, 9, 10]
        no_norm_loc_data = ori_data[:,feature_id].detach()
        data = no_norm_loc_data.detach().clone()

        new_mask = (data[:, :2, 1:]!=0) * (data[:, :2, :-1]!=0) 
        data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
        data[:, :2, 0] = 0    
        # # small vehicle: 1, big vehicles: 2, pedestrian 3, bicycle: 4, others: 5
        object_type = ori_data[:,2:3]
        data = data.float().to(self.dev)
        no_norm_loc_data = no_norm_loc_data.float().to(self.dev)
        object_type = object_type.to(self.dev) #type
        data[:,:2] = data[:,:2] / rescale_xy
        # result: data, no_norm_loc_data, object_type (function main.py:preprocess_data)
        
        _input_data = data[:,:,:self.obs_length,:] # (N, C, T, V)=(N, 4, 6, 120)
        output_loc_GT = data[:,:2,self.obs_length:,:] # (N, C, T, V)=(N, 2, 6, 120)
        output_mask = data[:,-1:,self.obs_length:,:] # (N, C, T, V)=(N, 1, 6, 120)

        A = A.float().to(self.dev)
        predicted = self.model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None) # (N, C, T, V)=(N, 2, 6, 120)
        predicted = predicted *rescale_xy
        predicted = predicted.detach().cpu().numpy()

        ori_output_last_loc = no_norm_loc_data[:,:2,self.obs_length-1,:].detach().cpu().numpy()
        predicted[:,:2,0,:] = ori_output_last_loc + predicted[:,:2,0,:]
        for ind in range(1, predicted.shape[-2]):
            predicted[:,:,ind,:] = predicted[:,:,ind-1,:] + predicted[:,:,ind,:]

        return self.dataloader.postprocess(input_data, _ori_data, mean_xy, predicted)