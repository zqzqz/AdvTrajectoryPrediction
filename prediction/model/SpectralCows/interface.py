import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Spectral-Trajectory-and-Behavior-Prediction/data_processing'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Spectral-Trajectory-and-Behavior-Prediction/ours'))
import time

import pickle
import numpy as np
import torch
from torch import optim

from .dataloader import SpectralCowsDataLoader
from prediction.base.interface import Interface

# import the following modules from the original repo
from def_train_eval import load_batch
from def_train_eval import compute_accuracy_stream1
from models import *

class SpectralCowsInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=None):
        super().__init__(obs_length, pred_length)

        self.device = torch.device("cuda")

        input_dim = hidden_dim = output_dim = 2

        self.encoder_stream1 = Encoder(input_dim, hidden_dim, output_dim).to(self.device)
        self.decoder_stream1 = Decoder('s1', input_dim , hidden_dim , output_dim, 1, obs_length).to(self.device)
        self.encoder_stream1.load_state_dict(torch.load(self.pre_load_model[0]))
        self.encoder_stream1.eval()       
        self.decoder_stream1.load_state_dict(torch.load(self.pre_load_model[1]))        
        self.decoder_stream1.eval()

    def run(self, input_data):
        # in `main.py`
        # eval(10, tr_seq_1, pred_seq_1, "APOL", "1stS1new")

        # the original arguments passed into `eval` function
        epochs = 10
        dataset_name = "LYFT"
        dataset_sufix = "1stS1new"
        learning_rate=1e-3  # the default learning rate in `eval` function

        encoder_stream1 = None
        decoder_stream1 = None
        encoder_stream2 = None
        decoder_stream2 = None

        encoder1loc = os.path.join(self.trained_model_dir, 'encoder_stream1_{}{}.pt'.format(dataset_name, dataset_sufix))
        decoder1loc = os.path.join(self.trained_model_dir, 'decoder_stream1_{}{}.pt'.format(dataset_name, dataset_sufix))
        encoder2loc = os.path.join(self.trained_model_dir, 'encoder_stream2_{}{}.pt'.format(dataset_name, dataset_sufix))
        decoder2loc = os.path.join(self.trained_model_dir, 'decoder_stream2_{}{}.pt'.format(dataset_name, dataset_sufix))

        train_raw = self.tr_seq_1
        pred_raw = self.pred_seq_1
        #    train2_raw = tr_seq_2
        #    pred2_raw = pred_seq_2
        # Initialize encoder, decoders for both streams
        batch = load_batch ( 0 , self.batch_size , 'pred' , train_raw , pred_raw , [], [], [], [] )
        batch , _, _ = batch
        batch_in_form = np.asarray ( [ batch[ i ][ 'sequence' ] for i in range ( self.batch_size ) ] )
        batch_in_form = torch.Tensor ( batch_in_form )
        [ batch_size , step_size , fea_size ] = np.shape(batch_in_form)
        input_dim = fea_size
        hidden_dim = fea_size
        output_dim = fea_size

        encoder_stream1 = Encoder ( input_dim , hidden_dim , output_dim ).to ( device )
        decoder_stream1 = Decoder ( 's1' , input_dim , hidden_dim , output_dim, batch_size, step_size ).to ( device )
        encoder_stream1_optimizer = optim.RMSprop(encoder_stream1.parameters(), lr=learning_rate)
        decoder_stream1_optimizer = optim.RMSprop(decoder_stream1.parameters(), lr=learning_rate)
        encoder_stream1.load_state_dict(torch.load(encoder1loc))
        encoder_stream1.eval()
        decoder_stream1.load_state_dict(torch.load(decoder1loc))
        decoder_stream1.eval()

        compute_accuracy_stream1(self.tr_seq_1, self.pred_seq_1, encoder_stream1, decoder_stream1, epochs)
