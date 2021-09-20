import os, sys
import random
import logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.apolloscape import ApolloscapeDataset
from prediction.dataset.ngsim import NGSIMDataset
from prediction.dataset.nuscenes import NuScenesDataset
from prediction.dataset.generate import data_offline_generator
from test_utils import *


############################################################################################################################################################

datasets = {
    "apolloscape": {
        "api": ApolloscapeDataset,
        "obs_length": 6,
        "pred_length": 6,
        "time_step": 0.5,
        "sample_step": 5,
        "attack_length": 6,
        "data_dir": "data/dataset/apolloscape/multi_frame"
    },
    "ngsim": {
        "api": NGSIMDataset,
        "obs_length": 15,
        "pred_length": 25,
        "time_step": 0.2,
        "skip": 5,
        "attack_length": 15,
        "data_dir": "data/dataset/ngsim/multi_frame"
    },
    "nuscenes": {
        "api": NuScenesDataset,
        "obs_length": 4,
        "pred_length": 12,
        "time_step": 0.5,
        "attack_length": 6,
        "data_dir": "data/dataset/nuscenes/multi_frame"
    }
}

for dataset_name in datasets:
    datasets[dataset_name]["instance"] = datasets[dataset_name]["api"](datasets[dataset_name]["obs_length"], datasets[dataset_name]["pred_length"], datasets[dataset_name]["time_step"])
    samples_file = os.path.join(datasets[dataset_name]["data_dir"], "samples.txt")
    with open(samples_file, 'r') as f:
        lines = f.readlines()
    datasets[dataset_name]["samples"] = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in lines]

models = {
    "grip": {
        "apolloscape": {
            "pre_load_model": "data/grip_apolloscape/model/best_model.pt",
            "num_node": 120, 
            "in_channels": 4,
            "rescale": [1,1]
        },
        "ngsim": {
            "pre_load_model": "data/grip_ngsim/model/best_model.pt",
            "num_node": 260, 
            "in_channels": 4,
            "rescale": [669.691/2, 669.691/2]
        },
        "nuscenes": {
            "pre_load_model": "data/grip_nuscenes/model/best_model.pt", 
            "num_node": 160, 
            "in_channels": 4,
            "rescale": [1,1]
        }
    },
    "fqa": {
        "apolloscape": {
            "pre_load_model": "data/fqa_apolloscape/model",
            "xy_distribution": datasets["apolloscape"]["instance"].xy_distribution
        },
        "ngsim": {
            "pre_load_model": "data/fqa_ngsim/model",
            "xy_distribution": datasets["ngsim"]["instance"].xy_distribution
        },
        "nuscenes": {
            "pre_load_model": "data/fqa_nuscenes/model",
            "xy_distribution": datasets["nuscenes"]["instance"].xy_distribution
        }
    },
    "trajectron": {
        "apolloscape": {
            "pre_load_model": "data/trajectron_apolloscape/model", 
            "maps": None
        },
        "ngsim": {
            "pre_load_model": "data/trajectron_ngsim/model", 
            "maps": None
        },
        "nuscenes": {
            "pre_load_model": "data/trajectron_nuscenes/model", 
            "maps": None
        }
    },
    "trajectron_map": {
        "nuscenes": {
            "pre_load_model": "data/trajectron_map_nuscenes/model", 
            "maps": datasets["nuscenes"]["instance"].maps
        }
    }
}


def load_model(model_name, dataset_name):
    if model_name == "grip":
        from prediction.model.GRIP.interface import GRIPInterface
        api_class = GRIPInterface
    elif model_name == "fqa":
        from prediction.model.FQA.interface import FQAInterface
        api_class = FQAInterface
    elif model_name == "trajectron" or model_name == "trajectron_map":
        from prediction.model.Trajectron.interface import TrajectronInterface
        api_class = TrajectronInterface

    return api_class(
        datasets[dataset_name]["obs_length"],
        datasets[dataset_name]["pred_length"],
        **models[model_name][dataset_name]
    )

#####################################################################################################################################################################


def sample():
    dataset_name = sys.argv[1]
    DATASET_DIR = "data/dataset/{}/multi_frame/raw".format(dataset_name)
    SAMPLE_PATH = "data/dataset/{}/multi_frame/samples.txt".format(dataset_name)
    interval_bound = 0.02 if dataset_name == "ngsim" else 0.05

    scene_candidates = []
    for name, input_data in data_offline_generator(DATASET_DIR):
        obj_candidates = []
        for obj_id, obj in input_data["objects"].items():
            if obj["type"] not in [1, 2]:
                continue
            if not obj["complete"]:
                continue
            if np.sum(np.sum((obj["observe_trace"][1:,:] - obj["observe_trace"][:-1,:]) ** 2, axis=1) < interval_bound) > 0:
                continue
            obj_candidates.append(obj_id)
        if len(obj_candidates) == 0:
            continue
        obj_id = random.choice(obj_candidates)
        scene_candidates.append((name, obj_id))
    cases = scene_candidates[::(len(scene_candidates)//100)][:100]

    with open(SAMPLE_PATH, 'w') as f:
        f.write('\n'.join(["{} {}".format(scene[0], scene[1]) for scene in cases]))


def sample_fix():
    dataset_name = sys.argv[1]
    DATASET_DIR = "data/dataset/{}/multi_frame/raw".format(dataset_name)
    SAMPLE_PATH = "data/dataset/{}/multi_frame/samples.txt".format(dataset_name)
    interval_bound = 0.02 if dataset_name == "ngsim" else 0.05

    with open(SAMPLE_PATH, 'r') as f:
        lines = f.readlines()
    scenes = [line[:-1].split(' ') for line in lines]
    scenes = [(int(scene[0]), int(scene[1])) for scene in scenes]
    new_scenes = []

    for case_id, obj_id in scenes:
        input_data = load_data(os.path.join(DATASET_DIR, "{}.json".format(case_id)))
        obj = input_data["objects"][str(obj_id)]
        if np.sum(np.sum((obj["observe_trace"][1:,:] - obj["observe_trace"][:-1,:]) ** 2, axis=1) < interval_bound) > 0:
            logging.error("wrong case {}".format((case_id, obj_id)))
            replace_candidates = []
            new_case_id = int(case_id)
            while True:
                _input_data = load_data(os.path.join(DATASET_DIR, "{}.json".format(new_case_id)))
                for _obj_id, _obj in _input_data["objects"].items():
                    if _obj["type"] not in [1, 2]:
                        continue
                    if not _obj["complete"]:
                        continue
                    if np.sum(np.sum((_obj["observe_trace"][1:,:] - _obj["observe_trace"][:-1,:]) ** 2, axis=1) < interval_bound) > 0:
                        continue
                    replace_candidates.append((new_case_id, int(_obj_id)))
                if len(replace_candidates) > 0:
                    break
                new_case_id += 1
            new_scene = random.choice(replace_candidates)
            logging.error("replace case {}".format(new_scene))
        else:
            new_scene = (case_id, obj_id)
        new_scenes.append(new_scene)

    print(new_scenes)
    with open(SAMPLE_PATH, 'w') as f:
        f.write('\n'.join(["{} {}".format(scene[0], scene[1]) for scene in new_scenes]))


def attack_multi_frame():
    assert(len(sys.argv) == 4)
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    overwrite = int(sys.argv[3])
    api = load_model(model_name, dataset_name)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"]
    physical_bounds = datasets[dataset_name]["instance"].bounds

    attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=1, iter_num=100, physical_bounds=physical_bounds)

    adv_attack(attacker, "data/dataset/{}/multi_frame/raw".format(dataset_name), 
                        "data/{}_{}/multi_frame/attack".format(model_name, dataset_name),
                        "data/{}_{}/multi_frame/attack_visualize".format(model_name, dataset_name), overwrite=overwrite, samples=samples)


def attack_single_frame():
    assert(len(sys.argv) == 4)
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    overwrite = int(sys.argv[3])
    api = load_model(model_name, dataset_name)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = 1
    physical_bounds = datasets[dataset_name]["instance"].bounds

    attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=1, iter_num=100, physical_bounds=physical_bounds)

    adv_attack(attacker, "data/dataset/{}/multi_frame/raw".format(dataset_name), 
                        "data/{}_{}/single_frame/attack".format(model_name, dataset_name),
                        "data/{}_{}/single_frame/attack_visualize".format(model_name, dataset_name), overwrite=overwrite, samples=samples)


def normal_multi_frame():
    assert(len(sys.argv) == 4)
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    overwrite = int(sys.argv[3])
    api = load_model(model_name, dataset_name)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"]

    normal_test(api, "data/dataset/{}/multi_frame/raw".format(dataset_name), 
                        "data/{}_{}/multi_frame/normal".format(model_name, dataset_name),
                        "data/{}_{}/multi_frame/normal_visualize".format(model_name, dataset_name), 
                        overwrite=overwrite, samples=samples, attack_length=attack_length)


def normal_single_frame():
    assert(len(sys.argv) == 4)
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    overwrite = int(sys.argv[3])
    api = load_model(model_name, dataset_name)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = 1

    normal_test(api, "data/dataset/{}/multi_frame/raw".format(dataset_name), 
                        "data/{}_{}/single_frame/normal".format(model_name, dataset_name),
                        "data/{}_{}/single_frame/normal_visualize".format(model_name, dataset_name), 
                        overwrite=overwrite, samples=samples, attack_length=attack_length)


def attack_one():
    model_name = "grip"
    dataset_name = "apolloscape"
    case_id = 8
    obj_id = 24
    attack_goal = "ade"

    api = load_model(model_name, dataset_name)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    attack_length = datasets[dataset_name]["attack_length"]
    physical_bounds = datasets[dataset_name]["instance"].bounds
    attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=4, iter_num=100, physical_bounds=physical_bounds)
    result_path = "{}-{}-{}-{}-{}.json".format(model_name, dataset_name, case_id, obj_id, attack_goal)
    figure_path = "{}-{}-{}-{}-{}.png".format(model_name, dataset_name, case_id, obj_id, attack_goal)
    attack_sample(attacker, DATASET_DIR, case_id, obj_id, attack_goal, result_path, figure_path)


def evaluate(mode="single_frame"):
    for model_name in models:
        for dataset_name in datasets:
            if model_name == "trajectron_map" and dataset_name in ["apolloscape", "ngsim"]:
                continue
            print(model_name, dataset_name)
            samples = datasets[dataset_name]["samples"]
            if mode.startswith("normal"):
                evaluate_loss("data/{}_{}/{}/normal".format(model_name, dataset_name, mode[7:]), samples=samples, output_dir="data/{}_{}/{}/normal_evaluate".format(model_name, dataset_name, mode[7:]), normal_data=True)
            else:
                evaluate_loss("data/{}_{}/{}/attack".format(model_name, dataset_name, mode), samples=samples, output_dir="data/{}_{}/{}/attack_evaluate".format(model_name, dataset_name, mode))


if __name__ == "__main__":
    attack_single_frame()
