import os, sys
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


def normal_multiframe_test(api, data_dir, result_dir, attack_length, overwrite=False):
    ret = create_dir(result_dir, overwrite=overwrite)
    if not ret:
        return False

    for name, input_data in data_offline_generator(data_dir):
        output_data = multi_frame_prediction(input_data, api, attack_length)
        store_data(output_data, os.path.join(result_dir, "{}.json".format(name)))


def raw_multiframe_visualize(data_dir, figure_dir, overwrite=False):
    ret = create_dir(figure_dir, overwrite=overwrite)
    if not ret:
        return False

    for name, input_data in data_offline_generator(data_dir):
        data_path = os.path.join(data_dir, "{}.json".format(name))
        figure_path = os.path.join(figure_dir, "{}.png".format(name))

        input_data = load_data(data_path)
        draw_single_frame(input_data, filename=figure_path, future=False, predict=False)


def normal_multiframe_visualize(data_dir, figure_dir, overwrite=False):
    ret = create_dir(figure_dir, overwrite=overwrite)
    if not ret:
        return False

    for name, input_data in data_offline_generator(data_dir):
        data_path = os.path.join(data_dir, "{}.json".format(name))
        figure_path = os.path.join(figure_dir, "{}.png".format(name))

        input_data = load_data(data_path)
        draw_multi_frame(input_data, filename=figure_path, future=True, predict=True)


def test_core(api, input_data, attack_length, result_path, figure_path):
    output_data = multi_frame_prediction(input_data, api, attack_length)
    store_data(output_data, result_path)
    draw_multi_frame(output_data, filename=figure_path, future=True, predict=True)


def adv_attack_core(attacker, input_data, obj_id, attack_goal, result_path, figure_path):
    obj_id = str(obj_id)
    result = attacker.run(input_data, obj_id, type=attack_goal)
    store_data(result, result_path)
    draw_multi_frame_attack(input_data, obj_id, result["perturbation"], result["output_data"], filename=figure_path)


def adv_attack(attacker, data_dir, result_dir, figure_dir, overwrite=False):
    ret = create_dir(result_dir, overwrite=overwrite)
    if not ret:
        return False
    ret = create_dir(figure_dir, overwrite=overwrite)
    if not ret:
        return False

    attack_goals = ["ade", "fde", "left", "right", "front", "rear"]

    for name, input_data in data_offline_generator(data_dir):
        for obj_id, obj in input_data["objects"].items():
            if obj["type"] not in [1, 2]:
                continue
            if not obj["complete"]:
                continue

            for attack_goal in attack_goals:
                result_path = os.path.join(result_dir, "{}-{}-{}.json".format(name, obj_id, attack_goal))
                figure_path = os.path.join(figure_dir, "{}-{}-{}.png".format(name, obj_id, attack_goal))
                if os.path.isfile(result_path):
                    continue

                try:
                    adv_attack_core(attacker, input_data, obj_id, attack_goal, result_path, figure_path)
                except Exception as e:
                    print(e)