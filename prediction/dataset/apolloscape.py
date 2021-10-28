import os
import numpy as np
import random

from .base import BaseDataset


class ApolloscapeDataset(BaseDataset):
    def __init__(self, obs_length, pred_length, time_step=0.5, sample_step=1):
        super().__init__(obs_length, pred_length, time_step)

        self.data_dir =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/apolloscape/")
        self.test_data_dir = os.path.join(self.data_dir, "prediction_test")
        self.val_data_dir = os.path.join(self.data_dir, "prediction_val")
        self.train_data_dir = os.path.join(self.data_dir, "prediction_train")
        self.data_path = {
            "train": self.train_data_dir,
            "val": self.val_data_dir,
            "test": self.test_data_dir
        }
        self.data = {
            "train": [],
            "val": [],
            "test": []
        }

        self.default_time_step = 0.5
        self.skip_step = int(self.time_step / self.default_time_step)
        self.feature_dimension = 5
        self.sample_step = sample_step

        self.xy_mean = np.array([127.4431223, 102.740081])
        self.xy_std = np.array([124.44508522, 71.96368189])
        self.xy_min = np.array([0.389, 0.674])
        self.xy_max = np.array([708.908, 348.585])

        self.xy_distribution = {
            "mean": self.xy_mean,
            "std": self.xy_std,
            "min": self.xy_min,
            "max": self.xy_max,
        }

        self.bounds = {
            "scalar_v": 10.539,
            "linear_a": 4.957,
            "rotate_a": 0.956,
            "linear_aa": 8.418,
            "rotate_aa": 1.577
        }

        self.detect_opts = {
            "long": {
                "d": 0.36,
                "t": 0.15,
                "scale": self.bounds["linear_a"]
            },
            "lat": {
                "d": 0.36,
                "t": 0.15,
                "scale": self.bounds["rotate_a"]
            },
            "thres": 0.02
        }

    def format_data(self, data_dir, allow_incomplete_traces=True, allow_invisible_objects=True, require_one_complete=True, require_one_visible=True):
        files = os.listdir(data_dir)
        for filename in files:
            if filename.split('.')[-1] != "txt":
                continue
            file_path = os.path.join(data_dir, filename)
            data = np.genfromtxt(file_path, delimiter=" ")
            data = data[~(data[:, 2] == 5)]
            start_frame_id = int(np.min(data[:,0]))

            numFrames = len(np.unique(data[:, 0]))
            numSlices = (numFrames - self.seq_length) // (self.sample_step * self.skip_step) + 1

            for slice_id in range(numSlices):
                input_data = {
                    "observe_length": self.obs_length,
                    "predict_length": self.pred_length,
                    "time_step": self.time_step,
                    "feature_dimension": self.feature_dimension,
                    "objects": {}
                }

                # fill data
                for local_frame_id in range(self.seq_length):
                    frame_id = start_frame_id + slice_id * self.sample_step * self.skip_step + local_frame_id * self.skip_step
                    frame_data = data[data[:, 0] == frame_id, :]

                    for obj_index in range(frame_data.shape[0]):
                        obj_data = frame_data[obj_index, :]
                        obj_id = str(int(obj_data[1]))

                        if obj_id not in input_data["objects"]:
                            if local_frame_id < self.obs_length:
                                input_data["objects"][obj_id] = {
                                    "type": int(obj_data[2]),
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