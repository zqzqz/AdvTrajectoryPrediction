import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import ApolloscapeDataset
from prediction.evaluate import *
from prediction.visualize import *
from prediction.GRIP import *

import pyswarms as ps

import numpy as np
import copy
import math
import json

obs_length = 6
pred_length = 6
api = GRIPInterface(None, obs_length, pred_length)
perturb_budget = 0.5

# trace_length = 26
# start_point = np.array([100, 100])
# step = np.array([5, 5])
# lane_distance = np.array([-2.5, 2.5])
# av_trace = np.zeros((trace_length, 7))
# pov_trace = np.zeros((trace_length, 7))
# for i in range(trace_length):
#     pov_trace[i,:2] = start_point + i * step
#     av_trace[i,:2] = pov_trace[i,:2] + lane_distance
# for trace in [av_trace, pov_trace]:
#     trace[:,2] = 38
#     trace[:,3] = 3
#     trace[:,4] = 1.5
#     trace[:,5] = 1
#     trace[:,6] = -0.75 * math.pi

trace_length = 18
with open("trace18/120.json", 'r') as f:
    input_data = json.load(f)
av_trace = np.array(input_data["objects"]["3"]["observe_full_trace"])
pov_trace = np.array(input_data["objects"]["1"]["observe_full_trace"])

attack_trace_length = 11
options = {'c1': 0.5, 'c2': 0.3, 'w': 1.0}
min_bound = -np.ones(attack_trace_length * 2) * perturb_budget
max_bound = np.ones(attack_trace_length * 2) * perturb_budget
center = np.zeros(attack_trace_length * 2)
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=attack_trace_length * 2, options=options, bounds=(min_bound, max_bound), center=center)

def to_input_data(av_trace, pov_trace, perturb, obs_length, pred_length, step):
    perturbed_trace = copy.deepcopy(av_trace[step:step+obs_length,:])
    perturbed_trace[:,:2] += perturb[step:step+obs_length,:]
    input_data = {
        "obs_length": obs_length,
        "pred_length": pred_length,
        "objects": {
            "0": {
                "type": 1,
                "observe_full_trace": perturbed_trace,
                "observe_trace": perturbed_trace[:,:2],
                "future_full_trace": av_trace[step+obs_length:step+obs_length+pred_length,:],
                "future_trace": av_trace[step+obs_length:step+obs_length+pred_length,:2],
                "predict_trace": np.zeros((pred_length, 2))
            },
            "1": {
                "type": 1,
                "observe_full_trace": pov_trace[step:step+obs_length,:],
                "observe_trace": pov_trace[step:step+obs_length,:2],
                "future_full_trace": pov_trace[step+obs_length:step+obs_length+pred_length,:],
                "future_trace": pov_trace[step+obs_length:step+obs_length+pred_length,:2],
                "predict_trace": np.zeros((pred_length, 2))
            }
        }
    }
    return input_data

def objective(x, av_trace, pov_trace, api):
    attack_trace_length = x.shape[1]//2
    obs_length, pred_length = api.obs_length, api.pred_length
    p1, p2 = pov_trace[0,:2], pov_trace[trace_length-1,:2]
    k, b = (p1[0]-p2[0])/(p2[1]-p1[1]), (p2[0]*p1[1]-p1[0]*p2[1])/(p2[1]-p1[1])
    loss = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        perturb = x[i,:].reshape((attack_trace_length, 2))
        for step in range(attack_trace_length-obs_length+1):
            input_data = to_input_data(av_trace, pov_trace, perturb, obs_length, pred_length, step)
            output_data = api.run(input_data)
            prediction = output_data["objects"]["0"]["predict_trace"]
            # distance = 0.707 * np.amin(prediction[:,1] - prediction[:,0])
            distance = np.amin(prediction[:,0] + prediction[:,1] * k + b) / abs(k)
            loss[i] += 1 / (1 + math.exp(-1*distance))
        loss[i] /= (attack_trace_length-obs_length+1)
        # print(loss[i])
    return loss

loss, pos = optimizer.optimize(objective, iters=1000, av_trace=av_trace, pov_trace=pov_trace, api=api)
perturb = pos.reshape((attack_trace_length, 2))

tag = "dataset_cutoff_case"

fig, ax = plt.subplots(figsize=(14,8))
ax.plot(av_trace[:attack_trace_length+pred_length,0], av_trace[:attack_trace_length+pred_length,1], "ko-")
ax.plot(pov_trace[:attack_trace_length+pred_length,0], pov_trace[:attack_trace_length+pred_length,1], "ko-")
perturbed_trace = copy.deepcopy(av_trace[:attack_trace_length,:2])
perturbed_trace += perturb
ax.plot(perturbed_trace[:,0], perturbed_trace[:,1], "ro-")
for step in range(attack_trace_length-obs_length+1):
    input_data = to_input_data(av_trace, pov_trace, perturb, obs_length, pred_length, step)
    output_data = api.run(input_data)
    predict_trace = np.concatenate((output_data["objects"]["0"]["observe_trace"][-1,:].reshape((1,2)), output_data["objects"]["0"]["predict_trace"]), axis=0)
    ax.plot(predict_trace[:,0], predict_trace[:,1], 'ro:')
ax.set_xlim([420, 560])
ax.set_ylim([80, 160])
fig.savefig("{}.png".format(tag))
plt.close(fig)

with open("{}.npy".format(tag), 'wb') as f:
    np.save(f, pos)