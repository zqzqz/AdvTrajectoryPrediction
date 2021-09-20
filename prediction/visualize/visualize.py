import matplotlib.pyplot as plt
import numpy as np


def get_trace(obj, name):
    trace = obj["{}_trace".format(name)]
    if name == "predict":
        return trace
    mask = obj["{}_mask".format(name)]
    indexes = np.argwhere(mask > 0)
    if indexes.shape[0] == 0:
        return None
    else:
        return trace[np.concatenate(indexes), :]


def draw_single_frame(output_data, filename=None, future=True, predict=True):
    fig, ax = plt.subplots(figsize=(10,10))
    xlim, ylim = [0x7fffffff, -0x7fffffff], [0x7fffffff, -0x7fffffff]

    for obj_id, obj in output_data["objects"].items():
        use_future, use_predict = future, predict

        observe_trace = get_trace(obj, "observe")
        if future:
            future_trace = get_trace(obj, "future")
            if future_trace is None:
                use_future = False
        if predict:
            predict_trace = get_trace(obj, "predict")
            if predict_trace is None:
                use_predict = False

        # update boundaries
        trace_all = observe_trace
        if use_future:
            trace_all = np.vstack((trace_all, future_trace))
        if use_predict:
            trace_all = np.vstack((trace_all, predict_trace))
        
        if np.min(trace_all) < 0.1:
            continue

        xlim[0] = min(xlim[0], trace_all[:,0].min())
        xlim[1] = max(xlim[1], trace_all[:,0].max())
        ylim[0] = min(ylim[0], trace_all[:,1].min())
        ylim[1] = max(ylim[1], trace_all[:,1].max())

        # draw lines
        if use_predict:
            pred = np.concatenate((observe_trace, predict_trace), axis=0)
            ax.plot(pred[:,0], pred[:,1], "ro:")

        if use_future:
            gt = np.concatenate((observe_trace, future_trace), axis=0)
            ax.plot(gt[:,0], gt[:,1], "bo-")
        else:
            ax.plot(observe_trace[:,0], observe_trace[:,1], "bo-")

        # print object id
        last_point = observe_trace[-1,:]
        ax.text(last_point[0], last_point[1], "{}:{}".format(obj_id, obj["type"]))

    ax.set_xlabel("x")
    ax.set_xlabel("y")
    lim = max(xlim[1]-xlim[0], ylim[1]-ylim[0]) * 1.1
    ax.set_xlim([sum(xlim)/2 - lim/2, sum(xlim)/2 + lim/2])
    ax.set_ylim([sum(ylim)/2 - lim/2, sum(ylim)/2 + lim/2])

    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)
    plt.close(fig)


def draw_multi_frame(output_data_dict, filename=None, future=True, predict=True):
    fig, ax = plt.subplots(figsize=(10,10))
    xlim, ylim = [0x7fffffff, -0x7fffffff], [0x7fffffff, -0x7fffffff]

    labeled_objs = []

    for index in range(output_data_dict["attack_length"]):
        index = str(index)
        output_data = output_data_dict["output_data"][index]

        for obj_id, obj in output_data["objects"].items():
            use_future, use_predict = future, predict

            observe_trace = get_trace(obj, "observe")
            if observe_trace is None:
                continue

            if future:
                future_trace = get_trace(obj, "future")
                if future_trace is None:
                    use_future = False
            if predict:
                predict_trace = get_trace(obj, "predict")
                if predict_trace is None:
                    use_predict = False
                if obj["type"] > 2:
                    use_predict = False

            # update boundaries
            trace_all = observe_trace
            if use_future: 
                trace_all = np.vstack((trace_all, future_trace))
            if use_predict:
                trace_all = np.vstack((trace_all, predict_trace))
            
            if np.min(trace_all) < 0.1:
                continue

            xlim[0] = min(xlim[0], trace_all[:,0].min())
            xlim[1] = max(xlim[1], trace_all[:,0].max())
            ylim[0] = min(ylim[0], trace_all[:,1].min())
            ylim[1] = max(ylim[1], trace_all[:,1].max())

            # draw lines
            if use_predict:
                pred = np.concatenate((observe_trace, predict_trace), axis=0)
                ax.plot(pred[:,0], pred[:,1], "ro:")

            if use_future:
                gt = np.concatenate((observe_trace, future_trace), axis=0)
                ax.plot(gt[:,0], gt[:,1], "bo-")
            else:
                ax.plot(observe_trace[:,0], observe_trace[:,1], "bo-")

            if obj_id not in labeled_objs:
                # print object id
                last_point = trace_all[-1,:]
                ax.text(last_point[0], last_point[1], "{}:{}".format(obj_id, obj["type"]))
                labeled_objs.append(obj_id)

    ax.set_xlabel("x")
    ax.set_xlabel("y")
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
        trace_all = get_trace(obj, "observe")
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

    for k, output_data in output_data_list.items():
        last_point = perturbed_trace[int(k)+output_data["observe_length"]-1,:]
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