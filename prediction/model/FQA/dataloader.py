import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FQA/src'))
import logging
import pickle
import random
import torch
from prediction.model.base.dataloader import DataLoader
import numpy as np

class FQADataLoader(DataLoader):
    def __init__(self, obs_length=6, pred_length=6):
        super().__init__(obs_length, pred_length)
        self.device = 'cuda:0' 

    def preprocess(self, input_data, xy_distribution):
        source_list = []
        mask_list = []
        for obj_id, obj in input_data["objects"].items():
            if obj["type"] not in [1, 2]:
                continue
            source = np.concatenate((obj["observe_trace"], np.zeros((self.pred_length,2))), axis=0)
            print(source)
            source = (source - xy_distribution["mean"]) / xy_distribution["std"]
            print(source)
            source_list.append(source)
            mask = np.concatenate((np.tile(obj["observe_mask"], (2,1)).T, np.zeros((self.pred_length,2))), axis=0)
            mask_list.append(mask)

        sources = torch.from_numpy(np.stack(source_list, axis=0).astype(np.float32)).to(self.device)
        masks = torch.from_numpy(np.stack(mask_list, axis=0).astype(np.float32)).to(self.device)
        sizes = [len(source_list)]

        return sources, masks, sizes

    def postprocess(self, input_data, preds, xy_distribution):
        pred_data = preds.cpu().detach().numpy()
        obj_index = 0
        for obj_id, obj in input_data["objects"].items():
            if obj["type"] not in [1, 2]:
                continue
            predict_trace = pred_data[obj_index].reshape((self.pred_length,2))
            predict_trace = predict_trace * xy_distribution["std"] + xy_distribution["mean"]
            obj["predict_trace"] = predict_trace
        return input_data

