import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.utils import store_data, load_data
from prediction.dataset.generate import data_offline_generator, add_flags, input_data_by_attack_step
from prediction.model.utils import multi_frame_prediction
from prediction.evaluate.evaluate import SingleFrameEvaluator, MultiFrameEvaluator
from prediction.evaluate.utils import store_report, report_mean
from prediction.visualize.visualize import draw_single_frame
from prediction.attack.gradient import GradientAttacker
import matplotlib.pyplot as plt
from prediction.attack.loss import *
from prediction.attack.constraint import *
from prediction.visualize.visualize import *
from test_utils import *
from test import models, datasets, load_model
import traceback


model_name = "grip"
dataset_name = "apolloscape"
DATASET_DIR = "data/dataset/{}".format(dataset_name)
attack_length = 1
physical_bounds = datasets[dataset_name]["instance"].bounds
observe_length, predict_length = datasets[dataset_name]["obs_length"], datasets[dataset_name]["pred_length"]
api = load_model(model_name, dataset_name, smooth=True, augment=False)
attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=10, iter_num=100, physical_bounds=physical_bounds)
attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
samples = datasets[dataset_name]["samples"]

result_dir = "data/{}_{}/single_frame/attack/realistic/raw".format(model_name, dataset_name)
figure_dir = "data/{}_{}/single_frame/attack/realistic/visualize".format(model_name, dataset_name)
evaluate_dir = "data/{}_{}/single_frame/attack/realistic/evaluate".format(model_name, dataset_name)
result_dir2 = "data/{}_{}/single_frame/attack/realistic2/raw".format(model_name, dataset_name)
figure_dir2 = "data/{}_{}/single_frame/attack/realistic2/visualize".format(model_name, dataset_name)
evaluate_dir2 = "data/{}_{}/single_frame/attack/realistic2/evaluate".format(model_name, dataset_name)
for path in [result_dir, figure_dir, result_dir2, figure_dir2, evaluate_dir2]:
    create_dir(path)


def get_input_data1(case_id, obj_id):
    if int(case_id)-observe_length < 0:
        raise Exception("Invalid case")
    input_data = load_data(os.path.join(DATASET_DIR, "multi_frame", "raw", "{}.json".format(int(case_id)-observe_length)))
    input_data = input_data_by_attack_step(input_data, observe_length, predict_length, 0)
    add_flags(input_data)
    if not input_data["objects"][str(obj_id)]["complete"]:
        raise Exception("Invalid case")
    return input_data


def get_input_data2(case_id, obj_id, output_data):
    input_data = load_data(os.path.join(DATASET_DIR, "multi_frame", "raw", "{}.json".format(case_id)))
    for _obj_id, obj in output_data["objects"].items():
        if str(_obj_id) == str(obj_id):
            continue
        if str(_obj_id) not in input_data["objects"]:
            continue
        input_data["objects"][str(_obj_id)]["observe_trace"][:predict_length] = obj["predict_trace"]
    return input_data


def task(case_id, obj_id, predict=False):
    input_data = get_input_data1(case_id, obj_id)
    output_data = api.run(input_data)
    input_data = get_input_data2(case_id, obj_id, output_data)
    for attack_goal in attack_goals:
        adv_attack_core(attacker, input_data, obj_id, attack_goal, os.path.join(result_dir, "{}-{}-{}.json".format(case_id, obj_id, attack_goal)), os.path.join(figure_dir, "{}-{}-{}.png".format(case_id, obj_id, attack_goal)))


def run():
    for case_id, obj_id in samples:
        try:
            task(case_id, obj_id)
        except Exception as e:
            print(case_id, obj_id, e)
            print(traceback.format_exc())


def test():
    for case_id, obj_id in samples:
        for attack_goal in attack_goals:
            try:
                result_data = load_data(os.path.join(result_dir, "{}-{}-{}.json".format(case_id, obj_id, attack_goal)))
                input_data = load_data(os.path.join(DATASET_DIR, "multi_frame", "raw", "{}.json".format(case_id)))
                input_data["objects"][str(obj_id)]["observe_trace"][:observe_length] = result_data["output_data"]["0"]["objects"][str(obj_id)]["observe_trace"]
                test_core(api, input_data, obj_id, 1, os.path.join(result_dir2, "{}-{}-{}.json".format(case_id, obj_id, attack_goal)), os.path.join(figure_dir2, "{}-{}-{}.png".format(case_id, obj_id, attack_goal)))
                result_data = load_data(os.path.join(result_dir2, "{}-{}-{}.json".format(case_id, obj_id, attack_goal)))
                loss = result_data["loss"][attack_goal]
                result_data["loss"] = loss
                store_data(result_data, os.path.join(result_dir2, "{}-{}-{}.json".format(case_id, obj_id, attack_goal)))
            except Exception as e:
                print(case_id, obj_id, e)
                print(traceback.format_exc())
    evaluate_loss(result_dir, samples=samples, output_dir=evaluate_dir, normal_data=False, attack_length=1)
    evaluate_loss(result_dir2, samples=samples, output_dir=evaluate_dir2, normal_data=False, attack_length=1)

run()
test()