import os, sys
import random
import logging
import copy
import torch
import argparse
torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from prediction.dataset.apolloscape import ApolloscapeDataset
from prediction.dataset.ngsim import NGSIMDataset
from prediction.dataset.nuscenes import NuScenesDataset
from prediction.dataset.generate import data_offline_generator
from prediction.attack.gradient import GradientAttacker
from prediction.attack.pso import PSOAttacker
from test_utils import *
from config import datasets, models


for dataset_name in datasets:
    samples_file = os.path.join(datasets[dataset_name]["data_dir"], "samples.txt")
    with open(samples_file, 'r') as f:
        lines = f.readlines()
    datasets[dataset_name]["samples"] = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in lines]


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


def attack(model_name, dataset_name, overwrite=0, mode="single_frame", augment=False, smooth=0, blackbox=False):
    api = load_model(model_name, dataset_name, augment=augment, smooth=smooth)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode.endswith("multi_frame") else 1
    physical_bounds = datasets[dataset_name]["instance"].bounds
    tag = get_tag(augment=augment, smooth=smooth, blackbox=blackbox)

    if not blackbox:
        attacker = GradientAttacker(api.obs_length, api.pred_length, attack_length, api, seed_num=10, iter_num=100, physical_bounds=physical_bounds, bound=1, learn_rate=0.1)
    else:
        attacker = PSOAttacker(api.obs_length, api.pred_length, attack_length, api, physical_bounds=physical_bounds)

    datadir = "data/{}_{}/{}/attack/{}".format(model_name, dataset_name, mode, tag)
    adv_attack(attacker, "data/dataset/{}/multi_frame/raw".format(dataset_name, mode), 
                        "{}/raw".format(datadir),
                        "{}/visualize".format(datadir), 
                        overwrite=overwrite, samples=samples)


def normal(model_name, dataset_name, overwrite=0, mode="single_frame", augment=False, smooth=0):
    api = load_model(model_name, dataset_name, augment=augment, smooth=smooth)
    DATASET_DIR = "data/dataset/{}".format(dataset_name)
    samples = datasets[dataset_name]["samples"]
    attack_length = datasets[dataset_name]["attack_length"] if mode.endswith("multi_frame") else 1
    tag = get_tag(augment=augment, smooth=smooth, blackbox=False)

    datadir = "data/{}_{}/{}/normal/{}".format(model_name, dataset_name, mode, tag)
    print(datadir)
    normal_test(api, "data/dataset/{}/multi_frame/raw".format(dataset_name, mode), 
                        "{}/raw".format(datadir),
                        "{}/visualize".format(datadir), 
                        overwrite=overwrite, samples=samples, attack_length=attack_length)


def evaluate(model_name=None, dataset_name=None, overwrite=0, mode="single_frame", augment=False, smooth=0, blackbox=False):
    if model_name is None:
        model_list = list(models.keys())
    else:
        model_list = [model_name]

    if dataset_name is None:
        dataset_list = list(datasets.keys())
    else:
        dataset_list = [dataset_name]

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


def main():
    parser = argparse.ArgumentParser(description='Testing script for prediction attacks.')
    parser.add_argument("--dataset", type=str, default="apolloscape", help="Name of dataset [apolloscape, ngsim, nuscenes]")
    parser.add_argument("--model", type=str, default="grip", help="Name of model [grip, fqa, trajectron, trajectron_map]")
    parser.add_argument("--mode", type=str, default="single_frame", help="Prediction mode [single_frame, multi_frame]")
    parser.add_argument("--augment", action="store_true", default=False, help="Enable data augmentation")
    parser.add_argument("--smooth", type=int, default=0, help="Enable trajectory smoothing -- 0: no smoothing; 1: train-time smoothing; 2: test-time smoothing; 3: test-time smoothing with anomaly detection")
    parser.add_argument("--blackbox", action="store_true", default=False, help="Use blackbox attack instead of whitebox")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing data")
    args = parser.parse_args()

    normal(dataset_name=args.dataset, model_name=args.model, mode=args.mode, augment=args.augment, smooth=args.smooth, overwrite=args.overwrite)
    attack(dataset_name=args.dataset, model_name=args.model, mode=args.mode, augment=args.augment, smooth=args.smooth, blackbox=args.blackbox, overwrite=args.overwrite)
    evaluate(dataset_name=args.dataset, model_name=args.model, mode="normal_"+args.mode, augment=args.augment, smooth=args.smooth, overwrite=args.overwrite)
    evaluate(dataset_name=args.dataset, model_name=args.model, mode=args.mode, augment=args.augment, smooth=args.smooth, blackbox=args.blackbox, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
