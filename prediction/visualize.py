import matplotlib.pyplot as plt
import numpy as np

def draw_traces(output_data, filename=None):
    fig, ax = plt.subplots(figsize=(10,10))
    observe_length = output_data["observe_length"]
    xlim, ylim = [0x7fffffff, -0x7fffffff], [0x7fffffff, -0x7fffffff]
    for obj_id, obj in output_data["objects"].items():
        # update boundaries
        trace_all = np.concatenate((obj["observe_trace"], obj["predict_trace"], obj["future_trace"]), axis=0)
        xlim[0] = min(xlim[0], trace_all[:,0].min())
        xlim[1] = max(xlim[0], trace_all[:,0].max())
        ylim[0] = min(ylim[0], trace_all[:,1].min())
        ylim[1] = max(ylim[1], trace_all[:,1].max())
        # draw lines
        pred = np.concatenate((obj["observe_trace"], obj["predict_trace"]), axis=0)
        ax.plot(pred[:,0], pred[:,1], "ro:")
        gt = np.concatenate((obj["observe_trace"], obj["future_trace"]), axis=0)
        ax.plot(gt[:,0], gt[:,1], "bo-")
        # print object id
        last_point = obj["observe_trace"][observe_length-1,:]
        ax.text(last_point[0], last_point[1], str(obj_id))
    ax.set_xlabel("x")
    lim = max(xlim[1]-xlim[0], ylim[1]-ylim[0]) * 1.1
    ax.set_xlim([sum(xlim)/2 - lim/2, sum(xlim)/2 + lim/2])
    ax.set_ylim([sum(ylim)/2 - lim/2, sum(ylim)/2 + lim/2])
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)