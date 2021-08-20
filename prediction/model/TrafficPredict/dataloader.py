import os, sys
import logging
import pickle
import random
from prediction.model.base.dataloader import DataLoader
import numpy as np

class TrafficPredictDataLoader(DataLoader):
    def __init__(self, obs_length=4, pred_length=6):
        super().__init__(obs_length, pred_length)
        self.min_position_x = 10000
        self.max_position_x = -10000
        self.min_position_y = 10000
        self.max_position_y = -10000

    def set_normalization(self, input_data_list):
        for input_data in input_data_list:
            for _, obj in input_data["objects"].items():
                self.min_position_x = min(self.min_position_x, min(obj["observe_trace"][:, 0]))
                self.max_position_x = max(self.max_position_x, max(obj["observe_trace"][:, 0]))
                self.min_position_y = min(self.min_position_y, min(obj["observe_trace"][:, 1]))
                self.max_position_y = max(self.max_position_y, max(obj["observe_trace"][:, 1]))

    def preprocess(self, input_data):
        x = []
        for frame_id in range(self.seq_length):
            frame_data = np.zeros((len(input_data["objects"]), 4))
            index = 0
            for obj_id, obj in input_data["objects"].items():
                if frame_id < self.obs_length:
                    pos = obj["observe_trace"][frame_id, :]
                else:
                    pos = obj["future_trace"][frame_id-self.obs_length, :]

                if np.sum(pos) == 0:
                    continue

                frame_data[index, 0] = int(obj_id)
                frame_data[index, 3] = self.class_objtype(obj["type"])
                frame_data[index, 1] = (pos[0] - self.min_position_x) / (self.max_position_x - self.min_position_x) * 2 -1
                frame_data[index, 2] = (pos[1] - self.min_position_y) / (self.max_position_y - self.min_position_y) * 2 -1
                index += 1
            x.append(frame_data[:index,:])
        return x # default batch_size=1

    def postprocess(self, input_data, ret_nodes):
        for frame_id in range(self.obs_length, self.seq_length):
            frame_data = ret_nodes[frame_id]
            for index in range(frame_data.shape[0]):
                obj_id = str(frame_data[index,0])
                obj = input_data["objects"][obj_id]
                obj["predict_trace"][frame_id-self.obs_length,0] = (frame_data[index,1] + 1) / 2 * (self.max_position_x - self.min_position_x) + self.min_position_x
                obj["predict_trace"][frame_id-self.obs_length,1] = (frame_data[index,2] + 1) / 2 * (self.max_position_y - self.min_position_y) + self.min_position_y
        return input_data

    @staticmethod
    def class_objtype(object_type):
        if object_type == 1 or object_type == 2:
            return 3
        elif object_type == 3:
            return 1
        elif object_type == 4:
            return 2
        else:
            return -1
