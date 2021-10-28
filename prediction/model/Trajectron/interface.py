import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectron-plus-plus/trajectron'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectron-plus-plus/experiments/nuScenes'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectron-plus-plus/experiments/nuScenes/devkit/python-sdk'))
import dill
import json
import argparse
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils

from .dataloader import TrajectronDataLoader
from prediction.model.base.interface import Interface
from environment import Environment, Scene, Node, GeometricMap, derivative_of
from model.dataset import get_timesteps_data, restore
from prediction.model.utils import detect_tensor, smooth_tensor

logger = logging.getLogger(__name__)


class TrajectronInterface(Interface):
    def __init__(self, obs_length, pred_length, time_step=0.5, pre_load_model=None, maps=None, smooth=0, dataset=None):
        super().__init__(obs_length, pred_length)
        self.time_step = time_step

        self.dataloader = TrajectronDataLoader(
            self.obs_length, self.pred_length, maps=maps
        )

        self.standardization = {
            'PEDESTRIAN': {
                'position': {
                    'x': {'mean': 0, 'std': 1},
                    'y': {'mean': 0, 'std': 1}
                },
                'velocity': {
                    'x': {'mean': 0, 'std': 2},
                    'y': {'mean': 0, 'std': 2}
                },
                'acceleration': {
                    'x': {'mean': 0, 'std': 1},
                    'y': {'mean': 0, 'std': 1}
                }
            },
            'VEHICLE': {
                'position': {
                    'x': {'mean': 0, 'std': 80},
                    'y': {'mean': 0, 'std': 80}
                },
                'velocity': {
                    'x': {'mean': 0, 'std': 15},
                    'y': {'mean': 0, 'std': 15},
                    'norm': {'mean': 0, 'std': 15}
                },
                'acceleration': {
                    'x': {'mean': 0, 'std': 4},
                    'y': {'mean': 0, 'std': 4},
                    'norm': {'mean': 0, 'std': 4}
                },
                'heading': {
                    'x': {'mean': 0, 'std': 1},
                    'y': {'mean': 0, 'std': 1},
                    '°': {'mean': 0, 'std': np.pi},
                    'd°': {'mean': 0, 'std': 1}
                }
            }
        }

        env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=self.standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        env.scenes = [Scene(timesteps=self.seq_length, dt=self.time_step, name="", aug_func=None)]
        self.env = env

        self.dev = 'cuda:0'

        if pre_load_model is not None:
            self.model, self.hyperparams = self.load_model(pre_load_model)
        else:
            self.model, self.hyperparams = None, {}

        self.test_vars = []

        self.smooth = smooth
        self.dataset = dataset

    def load_model(self, model_dir):
        filenames = os.listdir(model_dir)
        ts = -1
        for filename in filenames:
            try:
                if filename.split('.')[1] == "pt":
                    new_ts = int(filename.split('.')[0].split('-')[1])
                    if new_ts > ts:
                        ts = new_ts
            except:
                pass
        if ts < 0:
            raise Exception("Model not found.")

        model_registrar = ModelRegistrar(model_dir, self.dev)
        model_registrar.load_models(ts)
        with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
            hyperparams = json.load(config_json)

        trajectron = Trajectron(model_registrar, hyperparams, None, self.dev)

        trajectron.set_environment(self.env)
        trajectron.set_annealing_params()
        return trajectron, hyperparams

    def run(self, input_data, perturbation=None, backward=False):
        scene = self.dataloader.preprocess(input_data, env=self.env)
        scene.calculate_scene_graph(self.env.attention_radius,
                                self.hyperparams['edge_addition_filter'],
                                self.hyperparams['edge_removal_filter'])

        ph = self.pred_length
        ht = self.obs_length - 1
        timesteps = np.array([ht])

        observe_traces = {}
        future_traces = {}
        predict_traces = {}

        for node_type in self.env.NodeType:
            if node_type not in self.model.pred_state:
                continue

            model = self.model.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.model.state,
                                       pred_state=self.model.pred_state, edge_types=model.edge_types,
                                       min_ht=1, max_ht=ht, min_ft=ph,
                                       max_ft=ph, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.model.device)
            x_st_t = x_st_t.to(self.model.device)

            if perturbation is not None:
                target_index = -1
                for i, n in enumerate(nodes):
                    if n.id == perturbation["obj_id"]:
                        target_index = i
                        break
                if target_index >= 0:
                    p = perturbation["ready_value"][perturbation["obj_id"]]
                    dx = p
                    dv = torch.cat((torch.reshape(p[1,:] - p[0,:], (1,2)), p[1:,:] - p[:-1,:]), 0) / self.time_step
                    x[target_index][:,:2] += dx
                    x[target_index][:,2:4] += dv
                    x_st_t[target_index][:,:2] += dx / 80
                    x_st_t[target_index][:,2:4] += dv / 15
            
            if self.smooth > 0:
                for i, n in enumerate(nodes):
                    if torch.isnan(x[i]).sum() > 0:
                        continue
                    if torch.sum(x[i,0] != 0) < self.obs_length:
                        continue
                    if self.smooth == 3 and not detect_tensor(x[i,:2].T, self.dataset.detect_opts):
                        continue
                    x[i] = smooth_tensor(x[i])
                    x_st_t[i] = smooth_tensor(x_st_t[i])

            y = y_t.to(self.model.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.model.device)
            if type(map) == torch.Tensor:
                map = map.to(self.model.device)

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=1,
                                        z_mode=True,
                                        gmm_mode=True,
                                        full_dist=False,
                                        all_z_sep=False)

            if perturbation is not None:
                for i, n in enumerate(nodes):
                    if n.id != str(perturbation["obj_id"]):
                        continue
                    observe_traces[n.id] = x[i][:,:2]
                    future_traces[n.id] = y[i][:,:2]
                    predict_traces[n.id] = predictions[0][i][:,:2]

            predictions_np = predictions.cpu().detach().numpy()
            input_data = self.dataloader.postprocess(input_data, predictions_np, nodes, timesteps_o)

        if perturbation is not None:
            loss = perturbation["loss"](observe_traces, future_traces, predict_traces, 
                                        perturbation["obj_id"], perturbation["ready_value"][perturbation["obj_id"]], **perturbation["attack_opts"])
        else:
            loss = None

        if loss is None:
            return input_data
        else:
            return input_data, loss
