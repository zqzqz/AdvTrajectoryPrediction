import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.nuscenes import NuScenesDataset
import numpy as np

def test_nuscenes():
    dataset = NuScenesDataset(9, 8, 0.5)
    dataset.generate_data("test")
    for input_data in dataset.data_generator("test"):
        print(input_data)
        break


def evaluate_nuscenes():
    dataset = NuScenesDataset(9, 8, 0.5)
    max_obj_cnt = 0
    max_x = 0
    max_y = 0
    for data_type in ["train", "val", "test"]:
        dataset.generate_data(data_type)
        for input_data in dataset.data_generator(data_type):
            obj_cnt = len(input_data["objects"])
            max_obj_cnt = max(max_obj_cnt, obj_cnt)
            
            for _, obj in input_data["objects"].items():
                max_x = max(max_x, np.max(obj["observe_trace"][:,0]), np.max(obj["future_trace"][:,0]))
                max_y = max(max_y, np.max(obj["observe_trace"][:,1]), np.max(obj["future_trace"][:,1]))

    print(max_obj_cnt, max_x, max_y)


if __name__ == "__main__":
    # test_nuscenes()
    evaluate_nuscenes()