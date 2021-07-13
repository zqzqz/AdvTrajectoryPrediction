import matplotlib.pyplot as plt
import numpy as np

def draw_traces(output_data, filename=None, predict=True):
    fig, ax = plt.subplots(figsize=(10,10))
    observe_length = output_data["observe_length"]
    xlim, ylim = [0x7fffffff, -0x7fffffff], [0x7fffffff, -0x7fffffff]
    for obj_id, obj in output_data["objects"].items():
        # update boundaries
        if predict:
            trace_all = np.concatenate((obj["observe_trace"], obj["predict_trace"], obj["future_trace"]), axis=0)
        else:
            trace_all = np.concatenate((obj["observe_trace"], obj["future_trace"]), axis=0)
        xlim[0] = min(xlim[0], trace_all[:,0].min())
        xlim[1] = max(xlim[1], trace_all[:,0].max())
        ylim[0] = min(ylim[0], trace_all[:,1].min())
        ylim[1] = max(ylim[1], trace_all[:,1].max())
        # draw lines
        if predict:
            pred = np.concatenate((obj["observe_trace"], obj["predict_trace"]), axis=0)
            ax.plot(pred[:,0], pred[:,1], "ro:")
        gt = np.concatenate((obj["observe_trace"], obj["future_trace"]), axis=0)
        ax.plot(gt[:,0], gt[:,1], "bo-")
        # print object id
        last_point = obj["observe_trace"][observe_length-1,:]
        ax.text(last_point[0], last_point[1], "{}:{}".format(obj_id, obj["type"]))
    ax.set_xlabel("x")
    lim = max(xlim[1]-xlim[0], ylim[1]-ylim[0]) * 1.1
    ax.set_xlim([sum(xlim)/2 - lim/2, sum(xlim)/2 + lim/2])
    ax.set_ylim([sum(ylim)/2 - lim/2, sum(ylim)/2 + lim/2])
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)
    plt.close(fig)


def draw_multi_frame_attack(input_data, obj_id, perturbation, output_data_list, filename=None):
    fig, ax = plt.subplots(figsize=(10,10))
    xlim, ylim = [0x7fffffff, -0x7fffffff], [0x7fffffff, -0x7fffffff]
    
    for _obj_id, obj in input_data["objects"].items():
        # update boundaries
        trace_all = np.concatenate((obj["observe_trace"], obj["future_trace"]), axis=0)
        xlim[0] = min(xlim[0], trace_all[:,0].min())
        xlim[1] = max(xlim[1], trace_all[:,0].max())
        ylim[0] = min(ylim[0], trace_all[:,1].min())
        ylim[1] = max(ylim[1], trace_all[:,1].max())
        # draw lines
        gt = trace_all
        ax.plot(gt[:,0], gt[:,1], "bo-")
        # print object id
        last_point = trace_all[0,:]
        ax.text(last_point[0], last_point[1], "{}:{}".format(_obj_id, obj["type"]))

    if perturbation is not None:
        perturbed_length = perturbation[obj_id].shape[0]
        for _obj_id, _perturb_value in perturbation.items():
            _perturbed_trace = input_data["objects"][str(_obj_id)]["observe_trace"][:perturbed_length,:] + _perturb_value
            ax.plot(_perturbed_trace[:,0], _perturbed_trace[:,1], "ro-")
            if _obj_id == obj_id:
                perturbed_trace = _perturbed_trace
    else:
        perturbed_trace = input_data["objects"][str(obj_id)]["observe_trace"][:,:]

    for k, output_data in enumerate(output_data_list):
        last_point = perturbed_trace[k+output_data["observe_length"]-1,:]
        predict_trace = np.concatenate((last_point.reshape(1,2), output_data["objects"][str(obj_id)]["predict_trace"]), axis=0)
        ax.plot(predict_trace[:,0], predict_trace[:,1], "ro:")

    ax.set_xlabel("x")
    lim = max(xlim[1]-xlim[0], ylim[1]-ylim[0]) * 1.1
    ax.set_xlim([sum(xlim)/2 - lim/2, sum(xlim)/2 + lim/2])
    ax.set_ylim([sum(ylim)/2 - lim/2, sum(ylim)/2 + lim/2])
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)
    plt.close(fig)


def draw_error_distribution(ade_list, fde_list, filename=None):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.hist(ade_list, bins=20, label="ade")
    ax1.axvline(x=sum(ade_list)/len(ade_list))
    ax1.legend()
    ax2.hist(fde_list, bins=20, label="fde")
    ax2.axvline(x=sum(fde_list)/len(fde_list))
    ax2.legend()

    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)
    plt.close(fig) 