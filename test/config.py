import os

from prediction.dataset.apolloscape import ApolloscapeDataset
from prediction.dataset.ngsim import NGSIMDataset
from prediction.dataset.nuscenes import NuScenesDataset


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