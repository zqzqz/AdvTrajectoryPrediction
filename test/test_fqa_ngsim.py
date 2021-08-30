import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import NGSIMDataset
from prediction.dataset.generate import data_offline_generator
from prediction.model.FQA.interface import FQAInterface
from prediction.visualize.visualize import *
from test_utils import *

import numpy as np
import copy

DATADIR = "data/fqa_ngsim"
DATASET_DIR = "data/dataset/ngsim"
obs_length = 15
pred_length = 25
attack_length = 6
time_step = 0.5

dataset = NGSIMDataset(obs_length, pred_length, time_step, sample_step=40)
# dataset.generate_data("test")

api = FQAInterface(obs_length, pred_length, pre_load_model=os.path.join(DATADIR, "model"), xy_distribution=dataset.xy_distribution)


def test(case_id):
    input_data = load_data(os.path.join(DATASET_DIR, "single_frame", "raw", "{}.json".format(case_id)))
    output_data = api.run(input_data)
    # print(output_data)


def attack_sample(case_id, obj_id):
    attacker = GradientAttacker(obs_length, pred_length, attack_length, api, seed_num=4, iter_num=50)
    
    input_data = load_data(os.path.join(DATASET_DIR, "multi_frame", "raw", "{}.json".format(case_id)))
    # result_path = os.path.join(DATADIR, "multi_frame", "normal", "{}.json".format(case_id))
    # figure_path = os.path.join(DATADIR, "multi_frame", "normal_visualize", "{}.png".format(case_id))
    # test_core(api, input_data, attack_length, result_path, figure_path)

    for attack_goal in ["fde", "left", "right", "front", "rear"]:
        result_path = os.path.join(DATADIR, "multi_frame", "attack", "{}-{}-{}.json".format(case_id, obj_id, attack_goal))
        figure_path = os.path.join(DATADIR, "multi_frame", "attack_visualize", "{}-{}-{}.png".format(case_id, obj_id, attack_goal))
        adv_attack_core(attacker, input_data, obj_id, attack_goal, result_path, figure_path)

test(100)