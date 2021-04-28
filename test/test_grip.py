import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import ApolloscapeDataset
from prediction.evaluate import *
from prediction.utils import *
from prediction.visualize import *
from prediction.GRIP import *

import numpy as np
import copy

def simple_test(api):
    # simple test
    for input_data in api.data():
        output_data = api.run(input_data)
        print(output_data)
        draw_traces(output_data, "trace.png")
        break


def draw_high_error_traces():
    # Draw traces with high error
    for index, output_data in output_data_offline_generator("output_data_grip"):
        warn = False
        for _, obj in output_data["objects"].items():
            error = fde(obj["future_trace"], obj["predict_trace"])
            if error > 5:
                warn = True
                break
        if warn:
            draw_traces(output_data, "tmp/{}_trace.png".format(index))


def get_error():
    ade, fde = evaluate_error("output_data_grip", online=False)
    draw_error_distribution(ade, fde, "error.png")
    print("ade:", sum(ade)/len(ade), "fde:", sum(fde)/len(fde))


def try_perturbation(api):
    output_data = output_data_offline_by_index("output_data_grip", 0)
    draw_traces(output_data, "tmp/base.png")
    for i in range(100):
        new_input_data = copy.deepcopy(output_data)
        perturb = np.random.rand(2) * 2 - 1
        new_input_data["objects"]["24"]["observe_full_trace"][5,:2] += perturb
        new_input_data["objects"]["24"]["observe_trace"][5,:] += perturb
        new_output_data = api.run(new_input_data)
        draw_traces(new_output_data, "tmp/{}.png".format(i))


if __name__ == "__main__":
    api = GRIPInterface("apolloscape", 6, 6)
    data_dir = "output_data_grip"
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    index = 0
    for input_data in api.data():
        output_data = api.run(input_data)
        with open(os.path.join(data_dir, "{}.json".format(index)), "w") as f:
            json.dump(data_to_json(output_data), f, indent=2)
        draw_traces(output_data, os.path.join(data_dir, "{}.png".format(index)))
        print("index {} done".format(index))
        index += 1