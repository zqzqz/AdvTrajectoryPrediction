import os, sys

from test_utils import multiframe_data
from config import datasets

assert(len(sys.argv) >=2)
dataset_name = sys.argv[1]
dataset_config = datasets[dataset_name]
dataset_cls = dataset_config["api"]
output_dir = os.path.join(dataset_config["data_dir"], "raw")
multiframe_data(output_dir, dataset_cls, obs_length=dataset_config["obs_length"], pred_length=dataset_config["pred_length"], time_step=dataset_config["time_step"], attack_length=dataset_config["attack_length"])