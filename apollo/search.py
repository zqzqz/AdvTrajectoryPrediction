from modules.prediction.proto import feature_pb2
from google.protobuf import text_format
import numpy as np
import pyswarms as ps
import os, sys

def objective(swarms, cfg={}):
    print("Iteration...")
    history_length = cfg["history_length"]
    point_num = swarms.shape[1]//2
    ret_score = np.zeros((swarms.shape[0]))
    for i in range(swarms.shape[0]):
        perturbation = np.zeros((history_length,2))
        perturbation[history_length-point_num:,0] = swarms[i,:point_num].T
        perturbation[history_length-point_num:,1] = swarms[i,point_num:].T
        np.savetxt(cfg["perturbation_path"], perturbation, delimiter=",", newline='\n', fmt='%.3f')

        os.system(cfg["cmd"])

        cost = np.sum(perturbation[:,0])
        obj = 0
        with open(cfg["predict_trajectory_path"], 'r') as f:
            traj = feature_pb2.Trajectory()
            text_format.Parse(f.read(), traj)
            for k in range(cfg["prediction_length"]):
                x, y = traj.trajectory_point[k].path_point.x, traj.trajectory_point[k].path_point.y
                if y - 4140714.42 > (4140699.51 - 4140714.42) / (587483.32 - 587473.07) * (x - 587473.07):
                    obj = 5
                    break
        ret_score[i] = cost - obj
    return ret_score


class Searcher():
    def __init__(self, cfg={}):
        self.cfg = cfg
        if "point_num" in cfg:
            self.point_num = cfg["point_num"]
        else:
            self.point_num = 1
        self.options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        min_bound = np.zeros(self.point_num*2)
        max_bound = np.zeros(self.point_num*2)
        max_bound[:self.point_num] += 1.0
        max_bound[self.point_num:] += 180
        self.center = np.zeros(self.point_num*2)
        self.center[:self.point_num] = 0.99
        self.center[self.point_num:] = 90
        self.bounds = (min_bound, max_bound)
        self.optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=self.point_num*2,
                                                 options=self.options, bounds=self.bounds, center=self.center)

    def run(self):
        return self.optimizer.optimize(objective, iters=300, cfg=self.cfg)

