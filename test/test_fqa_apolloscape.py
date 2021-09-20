import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import ApolloscapeDataset
from prediction.dataset.generate import data_offline_generator
from prediction.model.FQA.interface import FQAInterface
from prediction.visualize.visualize import *
from test_utils import *

import numpy as np
import copy

DATADIR = "data/fqa_apolloscape"
DATASET_DIR = "data/dataset/apolloscape"
obs_length = 6
pred_length = 6
attack_length = 6
time_step = 0.5

dataset = ApolloscapeDataset(obs_length, pred_length, time_step)
api = FQAInterface(obs_length, pred_length, pre_load_model=os.path.join(DATADIR, "model"), xy_distribution=dataset.xy_distribution)


normal_multiframe_test(api, os.path.join(DATASET_DIR, "multi_frame", "raw"), os.path.join(DATADIR, "multi_frame", "normal"), attack_length, figure_dir=os.path.join(DATADIR, "multi_frame", "normal_visualize"), overwrite=True)

# test_sample(api, DATASET_DIR, 28, attack_length, "test.json", "test.png")
