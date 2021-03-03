import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrafficPredict/srnn'))
import pickle
import time
import warnings

import numpy as np
import torch
from IPython import embed
from torch.autograd import Variable
from sample import sample

from criterion import Gaussian2DLikelihood
from helper import (
    compute_edges,
    get_final_error_separately,
    get_mean_error_separately,
    getCoef,
    sample_gaussian_2d,
)
from model import SRNN
from st_graph import ST_GRAPH
from .dataloader import TrafficPredictDataLoader

class TrafficPredictInterface():
    def __init__(self, configs={}):
        # TODO: make the trace legnth configurable
        self.obs_length = 4
        self.pred_length = 6
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        self.config_path = os.path.join(self.model_dir, "config.pkl")
        self.checkpoint_path = os.path.join(self.model_dir, "srnn_model.tar")

        # load model
        with open(self.config_path, "rb") as f:
            self.saved_args = pickle.load(f)

        self.use_cuda = self.saved_args.use_cuda

        self.net = SRNN(self.saved_args, True)
        if self.use_cuda:
            self.net = self.net.cuda()

        self.checkpoint = torch.load(self.checkpoint_path)
        self.net.load_state_dict(self.checkpoint["state_dict"])
        self.dataloader = dataloader = TrafficPredictDataLoader(
            1, self.obs_length, self.pred_length, infer=True
        )

    def data(self, dataset="apolloscape"):
        return self.dataloader.generate_data()

    def run(self, input_data):
        x = self.dataloader.preprocess(input_data)
        # Construct the ST-graph object
        stgraph = ST_GRAPH(1, self.pred_length + self.obs_length)
        # Construct ST graph
        stgraph.readGraph(x)

        nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

        # Convert to cuda variables
        nodes = Variable(torch.from_numpy(nodes).float(), volatile=True)
        edges = Variable(torch.from_numpy(edges).float(), volatile=True)
        if self.use_cuda:
            nodes = nodes.cuda()
            edges = edges.cuda()

        # Separate out the observed part of the trajectory
        obs_nodes, obs_edges, obs_nodesPresent, obs_edgesPresent = (
            nodes[: self.obs_length],
            edges[: self.obs_length],
            nodesPresent[: self.obs_length],
            edgesPresent[: self.obs_length],
        )

        # Sample function
        ret_nodes, ret_attn = sample(
            obs_nodes,
            obs_edges,
            obs_nodesPresent,
            obs_edgesPresent,
            self,
            self.net,
            nodes,
            edges,
            nodesPresent,
        )

        output_data = self.dataloader.postprocess(input_data, ret_nodes.cpu().numpy())

        return output_data
