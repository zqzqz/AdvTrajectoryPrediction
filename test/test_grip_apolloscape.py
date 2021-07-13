import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import ApolloscapeDataset
from prediction.model.GRIP import GRIPInterface

import numpy as np
import copy


def test_train():
    obs_length = 6
    pred_length = 6
    time_step = 0.5
    dataset = ApolloscapeDataset(obs_length, pred_length, time_step)
    api = GRIPInterface(obs_length, pred_length, pre_load_model=None)
    api.set_dataset(dataset)
    api.train(save_dir="grip_model")


if __name__ == "__main__":
    test_train()