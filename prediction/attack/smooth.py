import numpy as np
import torch


def get_trace(obj, name):
    if "{}_trace".format(name) not in obj or "{}_mask".format(name) not in obj:
        return None
    trace = obj["{}_trace".format(name)]
    mask = obj["{}_mask".format(name)]
    indexes = np.argwhere(mask > 0)
    if indexes.shape[0] == 0:
        return None
    else:
        return trace[np.concatenate(indexes), :]


def get_unit_vector(vectors):
    scale = np.sum(vectors ** 2, axis=1) ** 0.5
    result = np.zeros(vectors.shape)
    result[:,0] = vectors[:,0] / scale
    result[:,1] = vectors[:,1] / scale
    return result


# def smooth_core(trace, p):
#     scale = np.ones((p.shape[0],1))
#     return scale


# def smooth_tensor(input_data, perturbation_value, smooth_args):
#     perturbation_ready_value = {}
#     for obj_id, p_torch in perturbation_value.items():
#         obj = input_data["objects"][obj_id]
        
#         # assume the trace is complete (no zeros)
#         if not obj["complete"]:
#             continue

#         trace = obj["observe_trace"]
#         p = p_torch.cpu().detach().numpy()

#         scale = smooth_core(trace, p, smooth_args)

#         new_p_torch = p_torch.clone()
#         new_p_torch = new_p_torch * torch.from_numpy(scale).cuda()
#         perturbation_ready_value[obj_id] = new_p_torch
#     return perturbation_ready_value


def get_physical_constraints(data_generator):
    min_rotate_a, max_rotate_a = 0xffff, -0xffff
    min_linear_a, max_linear_a = 0xffff, -0xffff
    min_rotate_aa, max_rotate_aa = 0xffff, -0xffff
    min_linear_aa, max_linear_aa = 0xffff, -0xffff
    
    for _, input_data in data_generator:
        for _, obj in input_data["objects"].items():
            observe_trace = get_trace(obj, "observe")
            future_trace = get_trace(obj, "future")
            predict_trace = get_trace(obj, "predict")

            # update boundaries
            trace_all = observe_trace
            if future_trace:
                trace_all = np.vstack((trace_all, future_trace))

            if trace_all.shape[0] < 4:
                continue

            velocity = trace_all[1:,:] - trace_all[:-1,:]
            a = velocity[1:,:] - velocity[:-1,:]
            aa = a[1:,:] - a[:-1,:]
            direction = get_unit_vector(velocity)
            direction_r = np.concatenate((direction[:,1].reshape(direction.shape[0],1), 
                                       -direction[:,0].reshape(direction.shape[0],1)), axis=1)
            
            linear_a = np.sum(direction[:-1,:] * a, axis=1)
            min_linear_a = min(min_linear_a, np.min(linear_a))
            max_linear_a = max(max_linear_a, np.max(linear_a))

            rotate_a = np.sum(direction_r[:-1,:] * a, axis=1)
            min_rotate_a = min(min_rotate_a, np.min(rotate_a))
            max_rotate_a = max(max_rotate_a, np.max(rotate_a))

            linear_aa = np.sum(direction[:-2,:] * aa, axis=1)
            min_linear_aa = min(min_linear_aa, np.min(linear_aa))
            max_linear_aa = max(max_linear_aa, np.max(linear_aa))

            rotate_aa = np.sum(direction_r[:-2,:] * aa, axis=1)
            min_rotate_aa = min(min_rotate_aa, np.min(rotate_aa))
            max_rotate_aa = max(max_rotate_aa, np.max(rotate_aa))

    return min_rotate_a, max_rotate_a, min_linear_a, max_linear_a, min_rotate_aa, max_rotate_aa, min_linear_aa, max_linear_aa
            