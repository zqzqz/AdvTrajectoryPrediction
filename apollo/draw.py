import sys, os
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from cyber_py3 import cyber
from cyber_py3 import record

from modules.perception.proto import perception_obstacle_pb2
from modules.prediction.proto import feature_pb2
from modules.common.proto import pnc_point_pb2
from google.protobuf import text_format

class Visualizer():
    def __init__(self, cfg={}):
        self.root = "/apollo/eval_data"

        self.record_path = "/apollo/modules/prediction/eval_data/test2.record"
        freader = record.RecordReader(self.record_path)
        obstacle_msg_cnt = 0
        record_traj_list = []
        tmp_list = []
        for topic, msg, _, timestamp in freader.read_messages():
            if topic == "/apollo/perception/obstacles":
                parsed = perception_obstacle_pb2.PerceptionObstacles()
                parsed.ParseFromString(msg)
                msg = parsed

                x = msg.perception_obstacle[0].position.x
                y = msg.perception_obstacle[0].position.y
                record_traj_list.append([x, y])
                tmp_list.append([msg.perception_obstacle[0].height, msg.perception_obstacle[0].width, msg.perception_obstacle[0].length, x, y, msg.perception_obstacle[0].position.z, msg.perception_obstacle[0].theta])
        self.record_traj = np.array(record_traj_list)

        np.save("record_traj.npy", np.array(tmp_list))
        
        # not loaded yet
        self.history_length = 0
        self.prediction_length = 0
        self.gt_history_traj = None
        self.gt_prediction_traj = None
        self.history_traj = None
        self.prediction_traj = None

    def load(self, history_traj_path, prediction_traj_path):
        with open(history_traj_path, 'r') as f:
            traj = feature_pb2.Trajectory()
            text_format.Parse(f.read(), traj)

            # set history length
            self.history_length = len(traj.trajectory_point)
            print(self.history_length)
            # load history trajectory
            self.gt_history_traj = self.record_traj[:self.history_length,:]
            self.history_traj = np.zeros((self.history_length, 2))

            for i in range(self.history_length):
                self.history_traj[i,0] = traj.trajectory_point[i].path_point.x
                self.history_traj[i,1] = traj.trajectory_point[i].path_point.y
        
        with open(prediction_traj_path, 'r') as f:
            traj = feature_pb2.Trajectory()
            text_format.Parse(f.read(), traj)

            # set prediction length
            self.prediction_length = min(self.record_traj.shape[0]-self.history_length, 
                                         len(traj.trajectory_point))
            # load prediction trajectory
            self.gt_prediction_traj = self.record_traj[self.history_length:self.history_length+self.prediction_length,:]
            self.prediction_traj = np.zeros((self.prediction_length, 2))
            
            for i in range(self.prediction_length):
                self.prediction_traj[i,0] = traj.trajectory_point[i].path_point.x
                self.prediction_traj[i,1] = traj.trajectory_point[i].path_point.y


    def set_figure(self):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(587473,587493)
        ax.set_ylim(4140692,4140712)
        ax.plot([587470.62, 587470.62 + (587480.36 - 587470.62) * 3], [4140711.88, 4140711.88 + (4140697.39 - 4140711.88) * 3], 'k')
        ax.plot([587473.07, 587473.07 + (587483.32 - 587473.07) * 3], [4140714.42, 4140714.42 + (4140699.51 - 4140714.42) * 3], 'k')
        ax.plot([587475.53, 587475.53 + (587486.20 - 587475.53) * 3], [4140716.28, 4140716.28 + (4140701.63 - 4140716.28) * 3], 'k')
        return fig, ax


    def radial_heatmap(self, rad, a, data, label):
        fig = plt.figure()
        ax = Axes3D(fig)
        r, th = np.meshgrid(rad, a)

        plt.subplot(projection="polar")
        plt.pcolormesh(th, r, data, cmap = 'inferno')
        plt.plot(a, r, ls='none', color = 'k') 
        plt.grid()
        plt.colorbar()
        plt.savefig(os.path.join(self.root, '{}.png'.format(label)))


    def mlp(self):
        my_data = np.genfromtxt(os.path.join(self.root, "output/output1"), delimiter=',')
        rad = np.linspace(0.1, 1.0, 10)
        a = np.linspace(0, 2 * np.pi, 12)
        
        self.radial_heatmap(rad, a, my_data[:,3].reshape((10,12)).T, "mlp_heatmap")
        # self.radial_heatmap(rad, a, (my_data[:,4] == 0).reshape((10,12)).T, "mlp_bin_heatmap")
        self.radial_heatmap(rad, a, my_data[:,4].reshape((10,12)).T, "mlp_heatmap_2")


    def lstm(self):
        predicted_traj = np.zeros((10,12,6,2))
        for d in range(10):
            for t in range(12):
                try:
                    with open(os.path.join(self.root, "trajectories/{}_{}_0_1.pb.txt".format(d+1, t+1)), 'r') as f:
                        traj = feature_pb2.Trajectory()
                        text_format.Parse(f.read(), traj)
                        for i in range(self.prediction_length):
                            predicted_traj[d,t,i,0] = traj.trajectory_point[i].path_point.x
                            predicted_traj[d,t,i,1] = traj.trajectory_point[i].path_point.y
                except Exception as e:
                    print(e)

        # ADE & FDE
        extended_gt_traj = np.tile(self.gt_traj, (10,12, 1, 1))
        FDE = np.sum(np.power(extended_gt_traj[:,:,-1,:] - predicted_traj[:,:,-1,:], 2), axis=2)
        # print(FDE)
        ADE = np.mean(np.sum(np.power(extended_gt_traj - predicted_traj, 2), axis=3), axis=2)
        # print(ADE)

        rad = np.linspace(0.1, 1.0, 10)
        a = np.linspace(0, 2 * np.pi, 12)
        self.radial_heatmap(rad, a, FDE.T, "lstm_fde")
        self.radial_heatmap(rad, a, ADE.T, "lstm_ade")

        fig, ax = self.set_figure()
        ax.plot(self.gt_traj[:,0], self.gt_traj[:,1], 'bo-')
        ax.plot(self.history_traj[:,0], self.history_traj[:,1], 'bo-')
        for d in range(10):
            for t in range(12):
                ax.plot(predicted_traj[d,t,:,0], predicted_traj[d,t,:,1], 'o:')
        fig.savefig(os.path.join(self.root, "traj.png"))


    def draw_single_traj(self, label="traj"):
        fig, ax = self.set_figure()
        ax.plot(self.prediction_traj[:,0], self.prediction_traj[:,1], 'ro:', label="Perturbed prediction")
        ax.plot(self.history_traj[:,0], self.history_traj[:,1], 'co-', label="Perturbed history")
        ax.plot(self.gt_prediction_traj[:,0], self.gt_prediction_traj[:,1], 'bo:', label="GT prediction")
        ax.plot(self.gt_history_traj[:,0], self.gt_history_traj[:,1], 'ko-', label="GT history")
        ax.legend()
        fig.savefig(os.path.join(self.root, "{}.png".format(label)))

if __name__ == "__main__":
    V = Visualizer()
    for i in range(20, 31):
        V.load("/apollo/eval_data/trajectories/history{}.pb.txt".format(i), 
               "/apollo/eval_data/trajectories/predict{}.pb.txt".format(i))
        V.draw_single_traj("{}".format(i))
