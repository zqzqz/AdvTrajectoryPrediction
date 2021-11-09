import setGPU
import os, sys
import random
import logging
import copy
import torch
torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.apolloscape import ApolloscapeDataset
from prediction.dataset.ngsim import NGSIMDataset
from prediction.dataset.nuscenes import NuScenesDataset
from prediction.dataset.generate import data_offline_generator
from prediction.attack.gradient import GradientAttacker
from prediction.attack.pso import PSOAttacker
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
            "pre_load_model": "data/grip_apolloscape/model/original/best_model.pt",
            "num_node": 120, 
            "in_channels": 4,
            "rescale": [1,1]
        },
        "ngsim": {
            "pre_load_model": "data/grip_ngsim/model/original/best_model.pt",
            "num_node": 260, 
            "in_channels": 4,
            "rescale": [669.691/2, 669.691/2]
        },
        "nuscenes": {
            "pre_load_model": "data/grip_nuscenes/model/original/best_model.pt", 
            "num_node": 160, 
            "in_channels": 4,
            "rescale": [1,1]
        }
    },
    "fqa": {
        "apolloscape": {
            "pre_load_model": "data/fqa_apolloscape/model/original"
        },
        "ngsim": {
            "pre_load_model": "data/fqa_ngsim/model/original"
        },
        "nuscenes": {
            "pre_load_model": "data/fqa_nuscenes/model/original"
        }
    },
    "trajectron": {
        "apolloscape": {
            "pre_load_model": "data/trajectron_apolloscape/model/original", 
            "maps": None
        },
        "ngsim": {
            "pre_load_model": "data/trajectron_ngsim/model/original", 
            "maps": None
        },
        "nuscenes": {
            "pre_load_model": "data/trajectron_nuscenes/model/original", 
            "maps": None
        }
    },
    "trajectron_map": {
        "nuscenes": {
            "pre_load_model": "data/trajectron_map_nuscenes/model/original", 
            "maps": datasets["nuscenes"]["instance"].maps
        }
    }
}


def load_model(model_name, dataset_name, augment=False, smooth=0, models=models):
    if model_name == "grip":
        from prediction.model.GRIP.interface import GRIPInterface
        api_class = GRIPInterface
    elif model_name == "fqa":
        from prediction.model.FQA.interface import FQAInterface
        api_class = FQAInterface
    elif model_name == "trajectron" or model_name == "trajectron_map":
        from prediction.model.Trajectron.interface import TrajectronInterface
        api_class = TrajectronInterface

    model_config = copy.deepcopy(models)
    model_config[model_name][dataset_name]["dataset"] = datasets[dataset_name]["instance"]
    if augment and not smooth:
        model_config[model_name][dataset_name]["pre_load_model"] = model_config[model_name][dataset_name]["pre_load_model"].replace("/original", "/augment")
    if smooth and not augment:
        if smooth == 1:
            model_config[model_name][dataset_name]["pre_load_model"] = model_config[model_name][dataset_name]["pre_load_model"].replace("/original", "/smooth")
        model_config[model_name][dataset_name]["smooth"] = smooth
    if smooth and augment:
        model_config[model_name][dataset_name]["pre_load_model"] = model_config[model_name][dataset_name]["pre_load_model"].replace("/original", "/augment_smooth")
        model_config[model_name][dataset_name]["smooth"] = True

    return api_class(
        datasets[dataset_name]["obs_length"],
        datasets[dataset_name]["pred_length"],
        **model_config[model_name][dataset_name]
    )

#####################################################################################################################################################################

def get_tag(augment=False, smooth=0, blackbox=False):
    if augment and smooth:
        return "augment_smooth"
    elif augment:
        return "augment"
    elif smooth > 0:
        return "smooth" if smooth == 1 else "smooth"+str(smooth)
    elif blackbox:
        return "blackbox"
    else:
        return "original"


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


def attack(mode="single_frame", augment=False, smooth=0, blackbox=False):
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    overwrite = int(sys.argv[3])
    api = load_model(model_name, dataset_name, augment=augment, smooth=smooth)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode.endswith("multi_frame") else 1
    physical_bounds = datasets[dataset_name]["instance"].bounds
    tag = get_tag(augment=augment, smooth=smooth, blackbox=blackbox)

    if not blackbox:
        attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=10, iter_num=200, physical_bounds=physical_bounds, bound=1, learn_rate=0.001)
    else:
        attacker = PSOAttacker(api.obs_length, api.pred_length, attack_length, api, physical_bounds=physical_bounds)

    datadir = "data/{}_{}/{}/attack/{}".format(model_name, dataset_name, mode, tag)
    adv_attack(attacker, "data/dataset/{}/multi_frame/raw".format(dataset_name, mode), 
                        "{}/raw".format(datadir),
                        "{}/visualize".format(datadir), 
                        overwrite=overwrite, samples=samples)


def normal(mode="single_frame", augment=False, smooth=0):
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    overwrite = int(sys.argv[3])
    api = load_model(model_name, dataset_name, augment=augment, smooth=smooth)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode.endswith("multi_frame") else 1
    tag = get_tag(augment=augment, smooth=smooth, blackbox=blackbox)

    datadir = "data/{}_{}/{}/normal/{}".format(model_name, dataset_name, mode, tag)
    print(datadir)
    normal_test(api, "data/dataset/{}/multi_frame/raw".format(dataset_name, mode), 
                        "{}/raw".format(datadir),
                        "{}/visualize".format(datadir), 
                        overwrite=overwrite, samples=samples, attack_length=attack_length)


def evaluate(mode="single_frame", augment=False, smooth=0, blackbox=False):
    if len(sys.argv) >= 3:
        model_list = [sys.argv[1]]
        dataset_list = [sys.argv[2]]
    else:
        model_list = list(models.keys())
        dataset_list = list(datasets.keys())

    tag = get_tag(augment=augment, smooth=smooth, blackbox=blackbox)
    
    for model_name in model_list:
        for dataset_name in dataset_list:
            if model_name == "trajectron_map" and dataset_name in ["apolloscape", "ngsim"]:
                continue
            print(model_name, dataset_name)
            attack_length = datasets[dataset_name]["attack_length"] if mode.endswith("multi_frame") else 1
            samples = datasets[dataset_name]["samples"]
            if mode.startswith("normal"):
                datadir = "data/{}_{}/{}/normal/{}".format(model_name, dataset_name, mode[7:], tag)
                evaluate_loss("{}/raw".format(datadir), samples=samples, output_dir="{}/evaluate".format(datadir), normal_data=True, attack_length=attack_length)
            elif mode.startswith("transfer"):
                for other_model_name in models:
                    if other_model_name == model_name:
                        continue
                    datadir = "data/{}_{}/{}/transfer/{}".format(model_name, dataset_name, mode[9:], other_model_name)
                    evaluate_loss("{}/raw".format(datadir), samples=samples, output_dir="{}/evaluate".format(datadir), normal_data=False, attack_length=attack_length)
            else:
                datadir = "data/{}_{}/{}/attack/{}".format(model_name, dataset_name, mode, tag)
                evaluate_loss("{}/raw".format(datadir), samples=samples, output_dir="{}/evaluate".format(datadir), normal_data=False, attack_length=attack_length)


def attack_one():
    model_name = "grip"
    dataset_name = "apolloscape"
    case_id = 312
    obj_id = 319
    attack_goal = "ade"

    api = load_model(model_name, dataset_name, smooth=True, augment=False)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    # attack_length = datasets[dataset_name]["attack_length"]
    attack_length = 1
    physical_bounds = datasets[dataset_name]["instance"].bounds

    print("normal")
    result_path = "{}-{}-{}-{}-smooth.json".format(model_name, dataset_name, case_id, obj_id)
    figure_path = "{}-{}-{}-{}-smooth.png".format(model_name, dataset_name, case_id, obj_id)
    test_sample(api, DATASET_DIR, case_id, obj_id, attack_length, result_path, figure_path)

    print("attack")
    attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=1, iter_num=100, physical_bounds=physical_bounds)
    # attacker = PSOAttacker(api.obs_length, api.pred_length, attack_length, api, physical_bounds=physical_bounds)
    
    result_path = "{}-{}-{}-{}-{}-smooth.json".format(model_name, dataset_name, case_id, obj_id, attack_goal)
    figure_path = "{}-{}-{}-{}-{}-smooth.png".format(model_name, dataset_name, case_id, obj_id, attack_goal)
    attack_sample(attacker, DATASET_DIR, case_id, obj_id, attack_goal, result_path, figure_path)


if __name__ == "__main__":
    mode = "single_frame"
    augment = False
    smooth = 0
    blackbox = False
    normal(mode=mode, augment=augment, smooth=smooth)
    attack(mode=mode, augment=augment, smooth=smooth, blackbox=blackbox)
    evaluate(mode="normal_"+mode, augment=augment, smooth=smooth)
    evaluate(mode=mode, augment=augment, smooth=smooth, blackbox=blackbox)