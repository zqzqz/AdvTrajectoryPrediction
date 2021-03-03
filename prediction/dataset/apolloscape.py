import os
import numpy as np

class ApolloscapeDataset:
    def __init__(self, obs_length, pred_length):
        self.data_dir =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/apolloscape/")
        self.test_data_dir = os.path.join(self.data_dir, "prediction_test")
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = obs_length + pred_length

        self.min_position_x = 10000
        self.max_position_x = -10000
        self.min_position_y = 10000
        self.max_position_y = -10000

        for data in self.raw_test_data_generator():
            self.min_position_x = min(self.min_position_x, min(data[:, 3]))
            self.max_position_x = max(self.max_position_x, max(data[:, 3]))
            self.min_position_y = min(self.min_position_y, min(data[:, 4]))
            self.max_position_y = max(self.max_position_y, max(data[:, 4]))


    def raw_test_data_generator(self):
        for filename in os.listdir(self.test_data_dir):
            file_path = os.path.join(self.test_data_dir, filename)
            data = np.genfromtxt(file_path, delimiter=" ")
            yield data

    def format_test_data_generator(self):
        for data in self.raw_test_data_generator():
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
                for frame_id in range(slice_id*self.seq_length, (slice_id+1)*self.seq_length):
                    frame_data = data[data[:, 0] == frame_id, :]
                    for obj_index in range(frame_data.shape[0]):
                        obj_data = frame_data[obj_index, :]
                        obj_id = obj_data[1]
                        if obj_id not in input_data["objects"]:
                            input_data["objects"][int(obj_id)] = {
                                "type": int(obj_data[2]),
                                "observe_trace": np.zeros((self.obs_length,2)),
                                "future_trace": np.zeros((self.pred_length,2)),
                                "predict_trace": np.zeros((self.pred_length,2)),
                                "frame": slice_id*self.seq_length,
                                "length": 0
                            }
                        obj = input_data["objects"][obj_id]
                        if obj["length"] < self.seq_length and obj["frame"] == frame_id:
                            if obj["length"] < self.obs_length:
                                obj["observe_trace"][obj["length"], 0] = obj_data[3]
                                obj["observe_trace"][obj["length"], 1] = obj_data[4]
                            else:
                                obj["future_trace"][obj["length"]-self.obs_length, 0] = obj_data[3]
                                obj["future_trace"][obj["length"]-self.obs_length, 1] = obj_data[4]
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

                yield input_data
