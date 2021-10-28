import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FQA/src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FQA'))
import torch
import torch.optim as optim
import pickle
import time
import warnings
import itertools
import numpy as np
import logging
import copy
from datetime import datetime
import importlib.util

from .dataloader import FQADataLoader
from prediction.model.base.interface import Interface
from prediction.model.utils import detect_tensor, smooth_tensor

logger = logging.getLogger(__name__)


class FQAInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=None, seed=1, smooth=0, dataset=None):
        super().__init__(obs_length, pred_length)

        self.dataloader = FQADataLoader(
            self.obs_length, self.pred_length
        )

        self.device = 'cuda:0'
        self.seed = 1

        if pre_load_model is not None:
            self.model = self.load_model(pre_load_model)
        else:
            self.model = None

        self.smooth = smooth
        self.dataset = dataset

    def load_model(self, model_path):
        # load config
        spec = importlib.util.spec_from_file_location('cfg', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FQA/config/cfg_FQA.py'))
        conf_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf_mod)
        config = conf_mod.config
        config["net"]["saved_params_path"] = os.path.join(model_path, "best_valid_params.ptp")

        # set seed
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # load model
        spec = importlib.util.spec_from_file_location('model', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FQA/models/FQA/FQA.py'))
        model_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_mod)
        global_records = {'info': {}, 'result': {}}
        model = model_mod.Model(self.device, global_records, config)
        # the best checkpoint will be loeded when initalizing the model

        return model.net

    def run(self, input_data, perturbation=None, backward=False):
        if backward:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        if perturbation is not None:
            target_obj_id = str(perturbation["obj_id"])
        else:
            target_obj_id = None
        sources, s_masks, sizes, obj_index_map = self.dataloader.preprocess(input_data, self.dataset.xy_distribution, target_obj_id=target_obj_id)
        if perturbation is not None:
            sources[obj_index_map[perturbation["obj_id"]]][:self.obs_length,:2] += (perturbation["ready_value"][perturbation["obj_id"]] / np.max(self.dataset.xy_distribution["std"]))
        if self.smooth > 0:
            for obj_id, index in obj_index_map.items():
                # if torch.sum(sources[index][:self.obs_length,0] != 0) < self.obs_length:
                #     continue
                if self.smooth == 3 and not detect_tensor(sources[index][:self.obs_length,:2], self.dataset.detect_opts):
                    continue
                sources[index][:self.obs_length] = smooth_tensor(sources[index][:self.obs_length])
        
        burn_in_steps = self.obs_length
        preds, _ = self.model(sources, masks=s_masks, sizes=sizes, burn_in_steps=burn_in_steps)
        preds = torch.split(preds, sizes, dim=0)[0]
        output_data = self.dataloader.postprocess(input_data, preds, self.dataset.xy_distribution, obj_index_map)

        if perturbation is not None:
            observe_traces = {}
            future_traces = {}
            predict_traces = {}
            for obj_id, obj_index in obj_index_map.items():
                observe_traces[obj_id] = sources[obj_index][:self.obs_length,:2] * float(np.max(self.dataset.xy_distribution["std"])) + torch.from_numpy(self.dataset.xy_distribution["mean"]).cuda()
                future_traces[obj_id] = torch.from_numpy(input_data["objects"][obj_id]["future_trace"]).cuda()
                predict_traces[obj_id] = preds[obj_index][self.obs_length-1:self.obs_length+self.pred_length-1,:] * float(np.max(self.dataset.xy_distribution["std"])) + torch.from_numpy(self.dataset.xy_distribution["mean"]).cuda()
            loss = perturbation["loss"](observe_traces, future_traces, predict_traces, 
                                        perturbation["obj_id"], perturbation["ready_value"][perturbation["obj_id"]], **perturbation["attack_opts"])
        else:
            loss = None

        if loss is None:
            return output_data
        else:
            return output_data, loss
