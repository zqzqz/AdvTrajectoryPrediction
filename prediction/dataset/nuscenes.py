import os
import numpy as np
import random
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from .base import BaseDataset


class NuScenesDataset(BaseDataset):
    def __init__(self, obs_length, pred_length, time_step):
        super().__init__(obs_length, pred_length, time_step)

        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/nuScenes/")
        self.train_data_path = os.path.join(self.data_dir, "prediction_train")
        self.test_data_path = os.path.join(self.data_dir, "prediction_test")
        self.val_data_path = os.path.join(self.data_dir, "prediction_val")
        self.data_path = {
            "train": self.train_data_path,
            "val": self.val_data_path,
            "test": self.test_data_path
        }
        self.data = {
            "train": [],
            "val": [],
            "test": []
        }

        self.map_name_path = os.path.join(self.data_dir, "map_name.txt")
        self.scene_map = self.get_scene_map(self.map_name_path)
        self.maps = self.get_maps()

        self.default_time_step = 0.5
        self.skip_step = int(self.time_step / self.default_time_step)
        self.feature_dimension = 5

        self.xy_mean = np.array([132.04209346, 127.59561989])
        self.xy_std = np.array([47.65086468, 43.83605424])
        self.xy_min = np.array([49.502, 49.507])
        self.xy_max = np.array([405.27, 357.274])

        self.xy_distribution = {
            "mean": self.xy_mean,
            "std": self.xy_std,
            "min": self.xy_min,
            "max": self.xy_max,
        }

        self.bounds = {
            "scalar_v": 8.599,
            "linear_a": 1.275,
            "rotate_a": 0.468,
            "linear_aa": 1.957,
            "rotate_aa": 0.535
        }

    def get_scene_map(self, map_name_path):
        scene_map = {}
        with open(map_name_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tokens = line[:-1].split(' ')
                scene_name, map_name = tokens[1], tokens[2]
                scene_map[scene_name] = map_name
        return scene_map

    def get_map(self, map_name):
        nusc_map = NuScenesMap(dataroot=self.data_dir, map_name=map_name)
        return nusc_map

    def get_maps(self):
        maps = {}
        unique_maps = list(set(self.scene_map.values()))
        for map_name in unique_maps:
            maps[map_name] = self.get_map(map_name)
        return maps

    def format_data(self, data_dir, allow_incomplete_traces=True, allow_invisible_objects=True, require_one_complete=True, require_one_visible=True):
        files = os.listdir(data_dir)
        for filename in files:
            if filename.split('.')[-1] != "txt":
                continue
            file_path = os.path.join(data_dir, filename)
            scene_name = filename.split('.')[0]
            data = np.genfromtxt(file_path, delimiter=" ")
            data = data[~(data[:, 2] == 5)]
            start_frame_id = int(np.min(data[:,0]))

            numFrames = len(np.unique(data[:, 0]))
            numSlices = numFrames - (self.seq_length - 1) * self.skip_step + 1 + 1

            for slice_id in range(numSlices):
                input_data = {
                    "observe_length": self.obs_length,
                    "predict_length": self.pred_length,
                    "time_step": self.time_step,
                    "feature_dimension": self.feature_dimension,
                    "objects": {},
                    "map_name": self.scene_map[scene_name],
                    "scene_name": scene_name
                }

                # fill data
                for local_frame_id in range(self.seq_length):
                    frame_id = start_frame_id + slice_id + local_frame_id * self.skip_step
                    frame_data = data[data[:, 0] == frame_id, :]

                    for obj_index in range(frame_data.shape[0]):
                        obj_data = frame_data[obj_index, :]
                        obj_id = str(int(obj_data[1]))
                        obj_type = int(obj_data[2])
                        if obj_type > 3:
                            continue

                        if obj_id not in input_data["objects"]:
                            if local_frame_id < self.obs_length:
                                input_data["objects"][obj_id] = {
                                    "type": obj_type,
                                    "complete": True,
                                    "visible": True,
                                    "observe_trace": np.zeros((self.obs_length,2)),
                                    "observe_feature": np.zeros((self.obs_length,self.feature_dimension)),
                                    "observe_mask": np.zeros(self.obs_length),
                                    "future_trace": np.zeros((self.pred_length,2)),
                                    "future_feature": np.zeros((self.pred_length,self.feature_dimension)),
                                    "predict_trace": np.zeros((self.pred_length,2)),
                                    "future_mask": np.zeros(self.pred_length)
                                }
                            else:
                                continue

                        obj = input_data["objects"][obj_id]
                        if local_frame_id < self.obs_length:
                            obj["observe_trace"][local_frame_id, :] = obj_data[3:5]
                            obj["observe_feature"][local_frame_id, :] = obj_data[5:]
                            obj["observe_mask"][local_frame_id] = 1
                        else:
                            obj["future_trace"][local_frame_id-self.obs_length, :] = obj_data[3:5]
                            obj["future_feature"][local_frame_id-self.obs_length, :] = obj_data[5:]
                            obj["future_mask"][local_frame_id-self.obs_length] = 1

                # remove invalid data
                invalid_obj_ids = []
                complete_obj_ids = []
                visible_obj_ids = []
                for obj_id, obj in input_data["objects"].items():
                    if np.sum(obj["observe_mask"]) <= 0:
                        invalid_obj_ids.append(obj_id)
                    
                    if np.min(np.concatenate((obj["observe_mask"], obj["future_mask"]), axis=0)) <= 0:
                        if not allow_incomplete_traces:
                            invalid_obj_ids.append(obj_id)
                        else:
                            obj["complete"] = False
                    elif obj_id not in invalid_obj_ids:
                        complete_obj_ids.append(obj_id)

                    if np.min(obj["observe_mask"][-1]) <= 0:
                        if not allow_incomplete_traces:
                            invalid_obj_ids.append(obj_id)
                        else:
                            obj["visible"] = False
                    elif obj_id not in invalid_obj_ids:
                        visible_obj_ids.append(obj_id)

                for invalid_obj_id in invalid_obj_ids:
                    del input_data["objects"][invalid_obj_id]

                # may create empty data, especially after invalid data removal
                if len(input_data["objects"]) == 0:
                    continue

                if len(visible_obj_ids) == 0 and require_one_visible:
                    continue

                if len(complete_obj_ids) == 0 and require_one_complete:
                    continue

                yield input_data
