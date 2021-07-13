import os, sys
import logging
import pickle
import random
from prediction.model.base.dataloader import DataLoader
import numpy as np

class TrafficPredictDataLoader(DataLoader):
    def __init__(self, dataset, obs_length=4, pred_length=6):
        super().__init__(dataset, obs_length, pred_length)

        self.min_position_x = 10000
        self.max_position_x = -10000
        self.min_position_y = 10000
        self.max_position_y = -10000

        for data in self.dataset.raw_data_generator(self.dataset.val_data_dir, None):
            self.min_position_x = min(self.min_position_x, min(data[:, 3]))
            self.max_position_x = max(self.max_position_x, max(data[:, 3]))
            self.min_position_y = min(self.min_position_y, min(data[:, 4]))
            self.max_position_y = max(self.max_position_y, max(data[:, 4]))

    def generate_data(self):
        return self.dataset.format_data_generator(self.dataset.val_data_dir, None)

    def preprocess(self, input_data):
        x = []
        for frame_id in range(self.seq_length):
            frame_data = np.zeros((len(input_data["objects"]), 4))
            index = 0
            for obj_id, obj in input_data["objects"].items():
                frame_data[index, 0] = obj_id
                frame_data[index, 3] = self.class_objtype(obj["type"])
                if frame_id < self.obs_length:
                    pos = obj["observe_trace"][frame_id, :]
                else:
                    pos = obj["future_trace"][frame_id-self.obs_length, :]
                frame_data[index, 1] = (pos[0] - self.min_position_x) / (self.max_position_x - self.min_position_x) * 2 -1
                frame_data[index, 2] = (pos[1] - self.min_position_y) / (self.max_position_y - self.min_position_y) * 2 -1
                index += 1
            x.append(frame_data)
        return [x] # default batch_size=1

    def postprocess(self, input_data, ret_nodes):
        for frame_id in range(self.obs_length, self.seq_length):
            frame_data = ret_nodes[frame_id]
            index = 0
            for _, obj in input_data["objects"].items():
                obj["predict_trace"][frame_id-self.obs_length,0] = (frame_data[index,0] + 1) / 2 * (self.max_position_x - self.min_position_x) + self.min_position_x
                obj["predict_trace"][frame_id-self.obs_length,1] = (frame_data[index,1] + 1) / 2 * (self.max_position_y - self.min_position_y) + self.min_position_y
                index += 1
        return input_data

    def class_objtype(self, object_type):
        if object_type == 1 or object_type == 2:
            return 3
        elif object_type == 3:
            return 1
        elif object_type == 4:
            return 2
        else:
            return -1
