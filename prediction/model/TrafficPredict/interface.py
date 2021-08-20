import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrafficPredict/srnn'))
import pickle
import time
import warnings
import logging

import numpy as np
import torch
from IPython import embed
from torch.autograd import Variable
from sample import sample
from argparse import Namespace

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
from prediction.model.base.interface import Interface

logger = logging.getLogger(__name__)


class TrafficPredictInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")):
        super().__init__(obs_length, pred_length)

        if pre_load_model is not None:
            self.load_model(pre_load_model)
        else:
            self.default_model()

        self.dataloader = TrafficPredictDataLoader(
            self.obs_length, self.pred_length
        )

        self.net = None

    def load_model(self, model_path):
        model_dir = model_path
        config_path = os.path.join(model_dir, "config.pkl")
        checkpoint_path = os.path.join(model_dir, "srnn_model.tar")

        # load model
        with open(config_path, "rb") as f:
            self.args = pickle.load(f)

        self.net = SRNN(self.args, True)
        if self.args.use_cuda:
            self.net = self.net.cuda()

        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint["state_dict"])

    def default_model(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "config.pkl")
        with open(config_path, "rb") as f:
            self.args = pickle.load(f)

        self.args.seq_length = self.obs_length + self.pred_length
        self.args.pred_length = self.pred_length

        self.net = SRNN(self.args, False)
        if self.args.use_cuda:
            self.net = self.net.cuda()
        
        return self.net

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
        if self.args.use_cuda:
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

    def train(self, total_epoch=200, batch_size=64, save_dir=""):
        assert(self.dataset is not None)

        stgraph = ST_GRAPH(1, self.args.seq_length)
        net = self.default_model()

        optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)

        logger.info("Training begin")
        num_batches = int(self.dataset.data_size("train") / batch_size)
        valid_num_batches = int(self.dataset.data_size("val") / batch_size)
        best_val_loss = 100
        best_epoch = 0
        # Training
        for epoch in range(total_epoch):
            loss_epoch = 0
            batch_id = 0
            for batch in self.dataset.data_generator("train", batch_size=batch_size, random_order=True):
                batch_id += 1
                start = time.time()

                # Loss for this batch
                loss_batch = 0

                # For each sequence in the batch
                for input_data in batch:
                    x = self.dataloader.preprocess(input_data)
                    # Construct the graph for the current sequence
                    stgraph.readGraph([x])
                    nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

                    # Convert to cuda variables
                    nodes = Variable(torch.from_numpy(nodes).float())
                    # nodes[0] represent all the person's corrdinate show up in  frame 0.
                    if self.args.use_cuda:
                        nodes = nodes.cuda()
                    edges = Variable(torch.from_numpy(edges).float())
                    if self.args.use_cuda:
                        edges = edges.cuda()

                    # Define hidden states
                    numNodes = nodes.size()[1]

                    hidden_states_node_RNNs = Variable(
                        torch.zeros(numNodes, self.args.node_rnn_size)
                    )
                    if self.args.use_cuda:
                        hidden_states_node_RNNs = hidden_states_node_RNNs.cuda()

                    hidden_states_edge_RNNs = Variable(
                        torch.zeros(numNodes * numNodes, self.args.edge_rnn_size)
                    )
                    if self.args.use_cuda:
                        hidden_states_edge_RNNs = hidden_states_edge_RNNs.cuda()

                    cell_states_node_RNNs = Variable(
                        torch.zeros(numNodes, self.args.node_rnn_size)
                    )
                    if self.args.use_cuda:
                        cell_states_node_RNNs = cell_states_node_RNNs.cuda()

                    cell_states_edge_RNNs = Variable(
                        torch.zeros(numNodes * numNodes, self.args.edge_rnn_size)
                    )
                    if self.args.use_cuda:
                        cell_states_edge_RNNs = cell_states_edge_RNNs.cuda()

                    hidden_states_super_node_RNNs = Variable(
                        torch.zeros(3, self.args.node_rnn_size)
                    )
                    if self.args.use_cuda:
                        hidden_states_super_node_RNNs = hidden_states_super_node_RNNs.cuda()

                    cell_states_super_node_RNNs = Variable(
                        torch.zeros(3, self.args.node_rnn_size)
                    )
                    if self.args.use_cuda:
                        cell_states_super_node_RNNs = cell_states_super_node_RNNs.cuda()

                    hidden_states_super_node_Edge_RNNs = Variable(
                        torch.zeros(3, self.args.edge_rnn_size)
                    )
                    if self.args.use_cuda:
                        hidden_states_super_node_Edge_RNNs = (
                            hidden_states_super_node_Edge_RNNs.cuda()
                        )

                    cell_states_super_node_Edge_RNNs = Variable(
                        torch.zeros(3, self.args.edge_rnn_size)
                    )
                    if self.args.use_cuda:
                        cell_states_super_node_Edge_RNNs = (
                            cell_states_super_node_Edge_RNNs.cuda()
                        )

                    # Zero out the gradients
                    net.zero_grad()
                    optimizer.zero_grad()

                    # Forward prop
                    outputs, _, _, _, _, _, _, _, _, _ = net(
                        nodes[: self.args.seq_length],
                        edges[: self.args.seq_length],
                        nodesPresent[: self.args.seq_length],
                        edgesPresent[: self.args.seq_length],
                        hidden_states_node_RNNs,
                        hidden_states_edge_RNNs,
                        cell_states_node_RNNs,
                        cell_states_edge_RNNs,
                        hidden_states_super_node_RNNs,
                        hidden_states_super_node_Edge_RNNs,
                        cell_states_super_node_RNNs,
                        cell_states_super_node_Edge_RNNs,
                    )

                    # Compute loss
                    loss = Gaussian2DLikelihood(
                        outputs, nodes, nodesPresent, self.args.pred_length
                    )

                    if not isinstance(loss, int):
                        loss_batch += loss.item()
                        # embed()
                        # Compute gradients
                        loss.backward()

                        # Clip gradients
                        torch.nn.utils.clip_grad_norm(net.parameters(), self.args.grad_clip)
                        # Update parameters
                        optimizer.step()

                    # Reset the stgraph
                    stgraph.reset()

                end = time.time()
                loss_batch = loss_batch / batch_size
                loss_epoch += loss_batch

                logger.warn(
                    "{}/{} (epoch {}), train_loss = {:.12f}, time/batch = {:.12f}".format(
                        epoch * num_batches + batch_id,
                        total_epoch * num_batches,
                        epoch,
                        loss_batch,
                        end - start,
                    )
                )
            # Compute loss for the entire epoch
            loss_epoch /= num_batches
            # Log it
            logger.warn("Epoch {} loss {}".format(epoch, loss_epoch))

            # Validation
            loss_epoch = 0
            batch_id = 0
            for batch in self.dataset.data_generator("val", batch_size=batch_size, random_order=False):
                batch_id += 1
                # Loss for this batch
                loss_batch = 0

                for input_data in batch:
                    x = self.dataloader.preprocess(input_data)
                    stgraph.readGraph([x])

                    nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

                    # Convert to cuda variables
                    nodes = Variable(torch.from_numpy(nodes).float())
                    if self.args.use_cuda:
                        nodes = nodes.cuda()
                    edges = Variable(torch.from_numpy(edges).float())
                    if self.args.use_cuda:
                        edges = edges.cuda()

                    # Define hidden states
                    numNodes = nodes.size()[1]

                    hidden_states_node_RNNs = Variable(
                        torch.zeros(numNodes, self.args.node_rnn_size)
                    )
                    if self.args.use_cuda:
                        hidden_states_node_RNNs = hidden_states_node_RNNs.cuda()

                    hidden_states_edge_RNNs = Variable(
                        torch.zeros(numNodes * numNodes, self.args.edge_rnn_size)
                    )
                    if self.args.use_cuda:
                        hidden_states_edge_RNNs = hidden_states_edge_RNNs.cuda()
                    cell_states_node_RNNs = Variable(
                        torch.zeros(numNodes, self.args.node_rnn_size)
                    )
                    if self.args.use_cuda:
                        cell_states_node_RNNs = cell_states_node_RNNs.cuda()
                    cell_states_edge_RNNs = Variable(
                        torch.zeros(numNodes * numNodes, self.args.edge_rnn_size)
                    )
                    if self.args.use_cuda:
                        cell_states_edge_RNNs = cell_states_edge_RNNs.cuda()

                    hidden_states_super_node_RNNs = Variable(
                        torch.zeros(3, self.args.node_rnn_size)
                    )
                    if self.args.use_cuda:
                        hidden_states_super_node_RNNs = hidden_states_super_node_RNNs.cuda()

                    cell_states_super_node_RNNs = Variable(
                        torch.zeros(3, self.args.node_rnn_size)
                    )
                    if self.args.use_cuda:
                        cell_states_super_node_RNNs = cell_states_super_node_RNNs.cuda()

                    hidden_states_super_node_Edge_RNNs = Variable(
                        torch.zeros(3, self.args.edge_rnn_size)
                    )
                    if self.args.use_cuda:
                        hidden_states_super_node_Edge_RNNs = (
                            hidden_states_super_node_Edge_RNNs.cuda()
                        )

                    cell_states_super_node_Edge_RNNs = Variable(
                        torch.zeros(3, self.args.edge_rnn_size)
                    )
                    if self.args.use_cuda:
                        cell_states_super_node_Edge_RNNs = (
                            cell_states_super_node_Edge_RNNs.cuda()
                        )

                    outputs, _, _, _, _, _, _, _, _, _ = net(
                        nodes[: self.args.seq_length],
                        edges[: self.args.seq_length],
                        nodesPresent[: self.args.seq_length],
                        edgesPresent[: self.args.seq_length],
                        hidden_states_node_RNNs,
                        hidden_states_edge_RNNs,
                        cell_states_node_RNNs,
                        cell_states_edge_RNNs,
                        hidden_states_super_node_RNNs,
                        hidden_states_super_node_Edge_RNNs,
                        cell_states_super_node_RNNs,
                        cell_states_super_node_Edge_RNNs,
                    )

                    # Compute loss
                    loss = Gaussian2DLikelihood(
                        outputs, nodes, nodesPresent, self.args.pred_length
                    )

                    if not isinstance(loss, int):
                        loss_batch += loss.item()

                    # Reset the stgraph
                    stgraph.reset()

                loss_batch = loss_batch / batch_size
                loss_epoch += loss_batch

            loss_epoch = loss_epoch / valid_num_batches

            # Update best validation loss until now
            if loss_epoch < best_val_loss:
                best_val_loss = loss_epoch
                best_epoch = epoch

            # Record best epoch and best validation loss
            logger.info("(epoch {}), valid_loss = {:.3f}".format(epoch, loss_epoch))
            logger.info(
                "Best epoch {}, Best validation loss {}".format(best_epoch, best_val_loss)
            )
            # Log it
            logger.warn("Epoch {} validation loss {}".format(epoch, loss_epoch))

            # Save the model after each epoch
            logger.info("Saving model")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_dir, "model_epoch_{:04}.pt".format(epoch))
            )

        # Record the best epoch and best validation loss overall
        logger.info(
            "Best epoch {}, Best validation loss {}".format(best_epoch, best_val_loss)
        )