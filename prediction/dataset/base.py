import numpy as np
import random

class BaseDataset:
    def __init__(self, obs_length, pred_length, time_step):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = obs_length + pred_length
        self.time_step = time_step
        self.data = []
        self.data_path = []

    def format_data(self, *args, **kwargs):
        pass

    def generate_data(self, tag, **kwargs):
        self.data[tag] = []
        for input_data in self.format_data(self.data_path[tag], **kwargs):
            self.data[tag].append(input_data)

    def data_size(self, tag):
        return len(self.data[tag])

    def data_generator(self, tag, batch_size=0, random_order=False):
        idx_list = [i for i in range(self.data_size(tag))]
        if random_order:
            random.shuffle(idx_list)

        if batch_size > 0:
            for i in range(int(len(idx_list) / batch_size)):
                yield [self.data[tag][idx_list[k]] for k in range(i * batch_size, (i+1) * batch_size)]
        else:
            for idx in idx_list:
                yield self.data[tag][idx]