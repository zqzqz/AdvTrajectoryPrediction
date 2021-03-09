import os
import numpy as np

class ApolloscapeDataset:
    def __init__(self, obs_length, pred_length):
        self.data_dir =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/apolloscape/")
        self.test_data_dir = os.path.join(self.data_dir, "prediction_test")
        self.val_data_dir = os.path.join(self.data_dir, "prediction_val")
        self.train_data_dir = os.path.join(self.data_dir, "prediction_train")
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = obs_length + pred_length

    def format_data(self, data):
        input_data_list = []
        data = data[~(data[:, 2] == 5)]

        numFrames = len(np.unique(data[:, 0]))
        numSlices = numFrames // self.seq_length

        for slice_id in range(numSlices):
            input_data = {
                "observe_length": self.obs_length,
                "predict_length": self.pred_length,
                "objects": {}
            }

            # fill data
            start_frame_id = int(data[0,0])
            for frame_id in range(start_frame_id+slice_id*self.seq_length, start_frame_id+(slice_id+1)*self.seq_length):
                frame_data = data[data[:, 0] == frame_id, :]
                for obj_index in range(frame_data.shape[0]):
                    obj_data = frame_data[obj_index, :]
                    obj_id = obj_data[1]
                    if obj_id not in input_data["objects"]:
                        input_data["objects"][int(obj_id)] = {
                            "type": int(obj_data[2]),
                            "observe_trace": np.zeros((self.obs_length,2)),
                            "future_trace": np.zeros((self.pred_length,2)),
                            "observe_full_trace": np.zeros((self.obs_length,7)),
                            "future_full_trace": np.zeros((self.pred_length,7)),
                            "predict_trace": np.zeros((self.pred_length,2)),
                            "frame": frame_id,
                            "length": 0
                        }
                    obj = input_data["objects"][obj_id]
                    if obj["length"] < self.seq_length and obj["frame"] == frame_id:
                        if obj["length"] < self.obs_length:
                            obj["observe_trace"][obj["length"], :] = obj_data[3:5]
                            obj["observe_full_trace"][obj["length"], :] = obj_data[3:]
                        else:
                            obj["future_trace"][obj["length"]-self.obs_length, :] = obj_data[3:5]
                            obj["future_full_trace"][obj["length"]-self.obs_length, :] = obj_data[3:]
                        obj["length"] += 1
                        obj["frame"] += 1

            # remove invalid data
            invalid_obj_ids = []
            for obj_id, obj in input_data["objects"].items():
                if obj["length"] != self.seq_length:
                    invalid_obj_ids.append(obj_id)
                else:
                    del obj["length"]
                    del obj["frame"]
            for invalid_obj_id in invalid_obj_ids:
                del input_data["objects"][invalid_obj_id]
            if len(input_data["objects"]) == 0:
                continue

            input_data_list.append(input_data)
        return input_data_list
            
    def raw_data_generator(self, data_dir, data_file_list=None):
        files = []
        if data_file_list is None:
            files = os.listdir(data_dir)
        elif isinstance(data_file_list, list):
            files = data_file_list
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            data = np.genfromtxt(file_path, delimiter=" ")
            yield data

    def format_data_generator(self, data_dir, data_file_list=None):
        for data in self.raw_data_generator(data_dir, data_file_list):
            input_data_list = self.format_data(data)
            for input_data in input_data_list:
                yield input_data
