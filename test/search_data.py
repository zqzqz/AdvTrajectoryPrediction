import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import ApolloscapeDataset
from prediction.evaluate import *
from prediction.visualize import *
from prediction.GRIP import *

import numpy as np
import copy
import json

api = GRIPInterface(None, 6, 6)
dataset = ApolloscapeDataset(18, 1)

# cnt = 0
# for input_data in dataset.format_data_generator(dataset.val_data_dir):
#     draw_traces(input_data, "trace18/{}.png".format(cnt), predict=False)
#     with open("trace18/{}.json".format(cnt), 'w') as f:
#         json.dump(data_to_json(input_data), f, indent=2)
#     cnt += 1