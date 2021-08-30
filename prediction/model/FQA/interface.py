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

logger = logging.getLogger(__name__)


class FQAInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=None, seed=1, xy_distribution={}):
        super().__init__(obs_length, pred_length)

        self.dataloader = FQADataLoader(
            self.obs_length, self.pred_length
        )

        self.device = 'cuda:0'
        self.seed = 1
        self.xy_distribution = xy_distribution

        if pre_load_model is not None:
            self.model = self.load_model(pre_load_model)
        else:
            self.model = None

    def load_model(self, model_path):
        # load config
        spec = importlib.util.spec_from_file_location('cfg', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FQA/config/cfg_FQA.py'))
        conf_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf_mod)
        config = conf_mod.config

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

        sources, s_masks, sizes = self.dataloader.preprocess(input_data, self.xy_distribution)
        burn_in_steps = self.obs_length
        preds, _ = self.model(sources, masks=s_masks, sizes=sizes, burn_in_steps=burn_in_steps)
        preds = torch.split(preds, sizes, dim=0)[0][:,self.obs_length-1:self.obs_length+self.pred_length-1]

        print(sources[0], preds[0])

        output_data = self.dataloader.postprocess(input_data, preds, self.xy_distribution)
        return output_data
