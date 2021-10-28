import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.utils import store_data, load_data
from prediction.dataset.generate import data_offline_generator
from prediction.model.utils import multi_frame_prediction
from prediction.evaluate.evaluate import SingleFrameEvaluator, MultiFrameEvaluator
from prediction.evaluate.utils import store_report, report_mean
from prediction.visualize.visualize import draw_single_frame
from prediction.attack.gradient import GradientAttacker
import matplotlib.pyplot as plt
from prediction.attack.loss import *
from prediction.attack.constraint import *
from prediction.visualize.visualize import *
from test_utils import *
from test import models, datasets, load_model


def hard_scenarios():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    ax_id = 0
    for case_id, obj_id in zip([28, 130], [24, 4]):
        predict_traces = []
        for model_name in ["grip", "fqa", "trajectron"]:
            normal_data = load_data("case_study/{}-apolloscape-{}-{}-ade/normal.json".format(model_name, case_id, obj_id))
            frame_data = normal_data["output_data"]["2"]["objects"][str(obj_id)]
            observe_trace, future_trace = frame_data["observe_trace"], frame_data["future_trace"]
            predict_traces.append(frame_data["predict_trace"])
        last_point = observe_trace[-1,:]
        full_trace = np.concatenate((observe_trace, future_trace, *predict_traces), axis=0)
        min_x, max_x, min_y, max_y = np.min(full_trace[:,0]), np.max(full_trace[:,0]), np.min(full_trace[:,1]), np.max(full_trace[:,1])
        scale = max(max_x - min_x, max_y - min_y) * 1.1 / 2
        
        for model_name, predict_trace, color in zip(["GRIP++", "FQA", "Trajectron++"], predict_traces, ["red", "darkorange", "green"]):
            predict_trace = np.concatenate((last_point.reshape(1,2), predict_trace), axis=0)
            ax[ax_id].plot(predict_trace[:,0], predict_trace[:,1], "o:", color=color, label="Prediction of {}".format(model_name))
        
        future_trace = np.concatenate((last_point.reshape(1,2), future_trace), axis=0)
        ax[ax_id].plot(future_trace[:,0], future_trace[:,1], "bo:", label="Future")
        ax[ax_id].plot(observe_trace[:,0], observe_trace[:,1], "bo-", label="History")
        ax[ax_id].set_xlim([(min_x + max_x)/ 2 - scale, (min_x + max_x)/ 2 + scale])
        ax[ax_id].set_ylim([(min_y + max_y)/ 2 - scale, (min_y + max_y)/ 2 + scale])
        ax[ax_id].legend()
        ax_id += 1
    fig.savefig("figures/hard_scenarios.pdf")


hard_scenarios()