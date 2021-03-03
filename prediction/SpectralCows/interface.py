import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Spectral-Trajectory-and-Behavior-Prediction/data_processing'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Spectral-Trajectory-and-Behavior-Prediction/ours'))
import time

import pickle
import numpy as np

class SpectralCowsInterface():
    def __init__(self):
        self.repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spectral-Trajectory-and-Behavior-Prediction")
        self.data_dir = os.path.join(self.repo_dir, "/resources/data/APOL")
        with open(os.path.join(self.data_dir, "stream1_obs_data_test.pkl"), 'rb') as f1:
            self.tr_seq_1 = pickle.load(f1)
        with open(os.path.join(self.data_dir, "stream1_pred_data_test.pkl"), 'rb') as g1:
            self.pred_seq_1 = pickle.load(g1)

