import os, sys
import traceback
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.utils import store_data, load_data
from prediction.dataset.generate import data_offline_generator
from prediction.model.utils import multi_frame_prediction
from prediction.evaluate.evaluate import SingleFrameEvaluator, MultiFrameEvaluator
from prediction.evaluate.utils import store_report, report_mean
from prediction.visualize.visualize import draw_single_frame
from prediction.attack.gradient import GradientAttacker
from prediction.attack.loss import *
from prediction.visualize.visualize import *


def create_dir(datadir, overwrite=False):
    if os.path.isdir(datadir) and not overwrite:
        return False
    
    if not os.path.isdir(datadir):
        os.makedirs(datadir)

    return True


def create_file(filepath, overwrite=False):
    if os.path.isfile(filepath) and not overwrite:
        return False
    else:
        return True


def singleframe_data(datadir, dataset_cls, obs_length, pred_length, time_step, overwrite=False):
    ret = create_dir(datadir, overwrite=overwrite)
    if not ret:
        return False

    dataset = dataset_cls(obs_length, pred_length, time_step)
    dataset.generate_data("test")
    idx = 0
    for input_data in dataset.data_generator("test", batch_size=0, random_order=False):
        store_data(input_data, os.path.join(datadir, "{}.json".format(idx)))
        idx += 1


def normal_singleframe_test(api, data_dir, result_dir, overwrite=False):
    ret = create_dir(result_dir, overwrite=overwrite)
    if not ret:
        return False

    for name, input_data in data_offline_generator(data_dir):
        output_data = api.run(input_data)
        store_data(output_data, os.path.join(result_dir, "{}.json".format(name)))


def singleframe_evaluate(result_dir, report_path, overwrite=False):
    ret = create_file(report_path, overwrite=overwrite)
    if not ret:
        return
    
    evaluator = SingleFrameEvaluator()
    report = evaluator.evaluate(data_offline_generator(result_dir))

    for metric in report:
        print(metric, report_mean(report[metric]))

    store_report(report, report_path)


def multiframe_data(datadir, dataset_cls, obs_length, pred_length, attack_length, time_step, overwrite=False):
    ret = create_dir(datadir, overwrite=overwrite)
    if not ret:
        return False

    dataset = dataset_cls(obs_length + pred_length + attack_length -1, 0, time_step)
    dataset.generate_data("test", require_one_complete=True)
    idx = 0
    for input_data in dataset.data_generator("test", batch_size=0, random_order=False):
        store_data(input_data, os.path.join(datadir, "{}.json".format(idx)))
        idx += 1


def normal_multiframe_test(api, data_dir, result_dir, attack_length, output_dir=None, overwrite=False):
    ret = create_dir(result_dir, overwrite=overwrite)
    if not ret:
        return False

    if output_dir is not None:
        ret = create_dir(output_dir, overwrite=overwrite)
        if not ret:
            return False

    for name, input_data in data_offline_generator(data_dir):
        output_data = multi_frame_prediction(input_data, api, attack_length)

        data_path = os.path.join(result_dir, "{}.json".format(name))
        store_data(output_data, data_path)

        if output_dir is not None:
            figure_path = os.path.join(output_dir, "{}.png".format(name))
            draw_multi_frame(output_data, filename=figure_path, future=True, predict=True)


def raw_multiframe_visualize(data_dir, output_dir, overwrite=False):
    ret = create_dir(output_dir, overwrite=overwrite)
    if not ret:
        return False

    for name, input_data in data_offline_generator(data_dir):
        data_path = os.path.join(data_dir, "{}.json".format(name))
        figure_path = os.path.join(output_dir, "{}.png".format(name))

        input_data = load_data(data_path)
        draw_single_frame(input_data, filename=figure_path, future=False, predict=False)


def normal_multiframe_visualize(data_dir, output_dir, overwrite=False):
    ret = create_dir(output_dir, overwrite=overwrite)
    if not ret:
        return False

    for name, input_data in data_offline_generator(data_dir):
        data_path = os.path.join(data_dir, "{}.json".format(name))
        figure_path = os.path.join(output_dir, "{}.png".format(name))

        input_data = load_data(data_path)
        draw_multi_frame(input_data, filename=figure_path, future=True, predict=True)


def test_core(api, input_data, obj_id, attack_length, result_path, figure_path):
    result = multi_frame_prediction(input_data, api, attack_length)
    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
    result["loss"] = {attack_goal:0 for attack_goal in attack_goals}
    for k in range(attack_length):
        args = []
        for trace_name in ["observe_trace", "future_trace", "predict_trace"]:
            args.append({str(obj_id): torch.from_numpy(result["output_data"][str(k)]["objects"][str(obj_id)][trace_name]).cuda()})
        for attack_goal in attack_goals:
            result["loss"][attack_goal] += float(attack_loss(*args, str(obj_id), None, type=attack_goal).item())
    store_data(result, result_path)
    draw_multi_frame(result, filename=figure_path, future=True, predict=True)
    return result


def adv_attack_core(attacker, input_data, obj_id, attack_goal, result_path, figure_path):
    obj_id = str(obj_id)
    result = attacker.run(input_data, obj_id, type=attack_goal)
    store_data(result, result_path)
    draw_multi_frame_attack(input_data, obj_id, result["perturbation"], result["output_data"], filename=figure_path)
    return result


def normal_test(api, data_dir, result_dir, figure_dir, samples=[], attack_length=1, overwrite=False):
    ret = create_dir(result_dir, overwrite=overwrite)
    ret = create_dir(figure_dir, overwrite=overwrite)

    for name, obj_id in samples:
        logging.warn("Log {} {}".format(name, obj_id))
        input_data = load_data(os.path.join(data_dir, "{}.json".format(name)))
        result_path = os.path.join(result_dir, "{}-{}.json".format(name, obj_id))
        figure_path = os.path.join(figure_dir, "{}-{}.png".format(name, obj_id))
        if not overwrite and os.path.isfile(result_path):
            logging.warn("skip")
            continue

        try:
            test_core(api, input_data, obj_id, attack_length, result_path, figure_path)
        except Exception as e:
            logging.error("Error!")
            logging.error(traceback.format_exc())


def adv_attack(attacker, data_dir, result_dir, figure_dir, samples=[], overwrite=False):
    ret = create_dir(result_dir, overwrite=overwrite)
    ret = create_dir(figure_dir, overwrite=overwrite)

    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]

    for name, obj_id in samples:
        input_data = load_data(os.path.join(data_dir, "{}.json".format(name)))

        for attack_goal in attack_goals:
            logging.warn("Log {} {} {}".format(name, obj_id, attack_goal))
            result_path = os.path.join(result_dir, "{}-{}-{}.json".format(name, obj_id, attack_goal))
            figure_path = os.path.join(figure_dir, "{}-{}-{}.png".format(name, obj_id, attack_goal))
            if not overwrite and os.path.isfile(result_path):
                logging.warn("skip")
                continue

            try:
                adv_attack_core(attacker, input_data, obj_id, attack_goal, result_path, figure_path)
            except Exception as e:
                logging.error("Error!")
                logging.error(traceback.format_exc())


def test_sample(api, DATASET_DIR, case_id, obj_id, attack_length, result_path, figure_path):
    input_data = load_data(os.path.join(DATASET_DIR, "multi_frame", "raw", "{}.json".format(case_id)))
    test_core(api, input_data, obj_id, attack_length, result_path, figure_path)


def attack_sample(attacker, DATASET_DIR, case_id, obj_id, attack_goal, result_path, figure_path):
    input_data = load_data(os.path.join(DATASET_DIR, "multi_frame", "raw", "{}.json".format(case_id)))
    adv_attack_core(attacker, input_data, obj_id, attack_goal, result_path, figure_path)


def evaluate_loss(result_dir, samples=[], output_dir=None, normal_data=False, attack_length=1):
    _ = create_dir(output_dir, overwrite=False)
    loss_data = {}
    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]
    for attack_goal in attack_goals:
        loss_data[attack_goal] = {}

    for name, obj_id in samples:
        if not normal_data:
            for attack_goal in attack_goals:
                result_file = os.path.join(result_dir, "{}-{}-{}.json".format(name, obj_id, attack_goal))
                try:
                    data = load_data(result_file)
                    loss_data[attack_goal][(name, obj_id)] = float(data["loss"]) / attack_length
                except Exception as e:
                    # logging.error(e)
                    pass
        else:
            result_file = os.path.join(result_dir, "{}-{}.json".format(name, obj_id))
            try:
                data = load_data(result_file)
                for attack_goal in attack_goals:
                    loss_data[attack_goal][(name, obj_id)] = float(data["loss"][attack_goal]) / attack_length
            except Exception as e:
                # logging.error(e)
                pass

    if len(loss_data["ade"]) == 0:
        print("Empty!")
        return

    print("Finished {}/{}".format(len(loss_data["ade"]), len(samples)))

    if output_dir is not None:
        for attack_goal in attack_goals:
            data = list(loss_data[attack_goal].values())
            data.sort()
            p, x = np.histogram(data, bins=100)
            x = x[:-1] + (x[1] - x[0])/2
            plt.plot(x, p)
            plt.savefig(os.path.join(output_dir, "loss_{}.png".format(attack_goal)))
            plt.clf()

            loss_data_np = np.array([[instance[0], instance[1], loss] for instance, loss in loss_data[attack_goal].items()])
            loss_data_np = loss_data_np[np.argsort(loss_data_np[:,2])]
            np.savetxt(os.path.join(output_dir, "loss_{}.txt".format(attack_goal)), loss_data_np, fmt="%d %d %.4f")

    return loss_data
