import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import ApolloscapeDataset
from prediction.evaluate import *
from prediction.visualize import *
from prediction.utils import *
from prediction.GRIP import *

import numpy as np
import copy


def loss(predict_trace, future_trace, perturbation):
    return torch.sum(torch.square(torch.absolute(perturbation) * 10 + 1)) - torch.sum(torch.square(predict_trace - future_trace))


if __name__ == "__main__":
    api = GRIPInterface("apolloscape", 6, 6)
    data = load_data("output_data_grip/325.json")
    best_out, best_perturb, best_iter, best_loss = api.adv(data, {"obj_id":40, "loss":loss}, iter_num=100, learn_rate=0.1)
    print(best_perturb)
    # print(best_out)

    tag = "grip_adv"
    with open("{}.npy".format(tag), 'wb') as f:
        np.save(f, best_perturb)
    draw_traces(best_out, "{}.png".format(tag))
