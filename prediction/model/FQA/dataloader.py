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

    def preprocess(self, input_data, xy_distribution, target_obj_id=None):
        source_list = []
        mask_list = []
        obj_index_map = {}

        # if target_obj_id is not None:
        #     obj_id = target_obj_id
        #     obj = input_data["objects"][obj_id]
        #     source = np.concatenate(((obj["observe_trace"] - xy_distribution["mean"]) / np.max(xy_distribution["std"]), np.zeros((self.pred_length,2))), axis=0)
        #     source_list.append(source)
        #     mask = np.concatenate((np.tile(obj["observe_mask"], (2,1)).T, np.zeros((self.pred_length,2))), axis=0)
        #     obj_index_map[obj_id] = len(mask_list)
        #     mask_list.append(mask)

        for obj_id, obj in input_data["objects"].items():
            if obj["type"] not in [1, 2]:
                continue
            # if target_obj_id is not None and obj_id == target_obj_id:
            #     continue
            source = np.concatenate(((obj["observe_trace"] - xy_distribution["mean"]) / np.max(xy_distribution["std"]), np.zeros((self.pred_length,2))), axis=0)
            source_list.append(source)
            mask = np.concatenate((np.tile(obj["observe_mask"], (2,1)).T, np.zeros((self.pred_length,2))), axis=0)
            obj_index_map[obj_id] = len(mask_list)
            mask_list.append(mask)
            # if len(obj_index_map) >= 50:
            #     break

        sources = torch.from_numpy(np.stack(source_list, axis=0).astype(np.float32)).to(self.device)
        masks = torch.from_numpy(np.stack(mask_list, axis=0).astype(np.float32)).to(self.device)
        sizes = [len(source_list)]

        return sources, masks, sizes, obj_index_map

    def postprocess(self, input_data, preds, xy_distribution, obj_index_map):
        pred_data = preds.cpu().detach().numpy()
        pred_data = pred_data[:,self.obs_length-1:self.obs_length+self.pred_length-1]
        for obj_id in obj_index_map:
            obj = input_data["objects"][obj_id]
            obj_index = obj_index_map[obj_id]
            predict_trace = pred_data[obj_index].reshape((self.pred_length,2))
            predict_trace = predict_trace * np.max(xy_distribution["std"]) + xy_distribution["mean"]
            obj["predict_trace"] = predict_trace
        return input_data

