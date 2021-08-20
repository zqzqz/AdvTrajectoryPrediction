import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import ApolloscapeDataset
from prediction.model.TrafficPredict import TrafficPredictInterface
from prediction.evaluate.metrics import *
from prediction.visualize import *

import numpy as np
import copy


def train():
    obs_length = 6
    pred_length = 6
    time_step = 0.5
    dataset = ApolloscapeDataset(obs_length, pred_length, time_step)
    dataset.generate_data("train")
    dataset.generate_data("val")
    api = TrafficPredictInterface(obs_length, pred_length, pre_load_model=None)
    api.set_dataset(dataset)
    os.makedirs("trafficpredict_apolloscape_model", exist_ok=True)
    api.train(save_dir="trafficpredict_apolloscape_model", batch_size=8, total_epoch=300)


def test():
    obs_length = 6
    pred_length = 6
    time_step = 0.5
    dataset = ApolloscapeDataset(obs_length, pred_length, time_step)
    dataset.generate_data("test")
    api = TrafficPredictInterface(obs_length, pred_length, pre_load_model="trafficpredict_apolloscape_model/model_epoch_0099.pt")

    for input_data in dataset.data_generator("test", batch_size=0):
        output_data = api.run(input_data)
        # Do something
        draw_traces(output_data, filename="test.png")
        break


if __name__ == "__main__":
    train()
    # test()