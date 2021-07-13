import os, sys
sys.path.append("/apollo/eval_data/AB3DMOT")
from AB3DMOT_libs.model import AB3DMOT
import numpy as np
import math
import copy


if __name__ == "__main__":
    record_traj = np.load("record_traj.npy")
    # print(record_traj)
    perturbation = np.genfromtxt('perturbation_ori.txt', delimiter=',')
    perturbation_new = np.zeros((30,3))
    perturbation_new[:,0] = np.ones(30) * 2
    # print(perturbation)
    model = AB3DMOT(max_age=10, min_hits=3)
    for i in range(record_traj.shape[0]):
        way_point = copy.deepcopy(record_traj[i,:])

        if i == 0:
            original_theta = way_point[6]

        if i > 0 and i < perturbation.shape[0]:
            d = perturbation[i,1]
            t = perturbation[i,2]
            x_offset = math.cos(original_theta + t / 360 * 2 * math.pi) * d
            y_offset = math.sin(original_theta + t / 360 * 2 * math.pi) * d

            way_point[3] += x_offset
            way_point[4] += y_offset
            
            v_x, v_y = (way_point[3] - last_x) / 0.1, (way_point[4] - last_y) / 0.1
            way_point[6] = math.atan(v_y / v_x)

        last_x, last_y = way_point[3], way_point[4]

        trackers = model.update({
            "dets": way_point.reshape((1, 7)),
            "info": np.zeros((1, 1))
        })
        if len(trackers) > 0 and i < perturbation_new.shape[0]:
            for t in trackers:
                perturbation_new[i,1:3] = trackers[0][3:5] - record_traj[i,3:5]
                print(i, perturbation_new[i,1:3])
        else:
            break

    np.savetxt("perturbation.txt", perturbation_new, delimiter=",", fmt='%1.3f')

