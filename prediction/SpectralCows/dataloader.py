import logging
import os
import pickle
import random
import numpy as np

from prediction.base.dataloader import DataLoader

class SpectralCowsDataLoader(DataLoader):
    def __init__(self, dataset, obs_length=4, pred_length=6):
        super().__init__(dataset, obs_length, pred_length)

    def generate_data(self):
        pass

    def preprocess(self, input_data):
        # TODO: related parts are presented in `data_processing/data_stream.py`
        # should convert the original data into two streams
        pass

    def postprocess(self, model_output):
        pass
