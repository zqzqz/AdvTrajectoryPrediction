import numpy as np
import torch
import copy


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
    scale = np.sum(vectors ** 2, axis=1) ** 0.5 + 0.001
    result = np.zeros(vectors.shape)
    result[:,0] = vectors[:,0] / scale
    result[:,1] = vectors[:,1] / scale
    return result


def get_metrics(trace_array):
    v = trace_array[1:,:] - trace_array[:-1,:]
    a = v[1:,:] - v[:-1,:]
    aa = a[1:,:] - a[:-1,:]

    direction = get_unit_vector(v)
    direction_r = np.concatenate((direction[:,1].reshape(direction.shape[0],1), 
                                -direction[:,0].reshape(direction.shape[0],1)), axis=1)

    scalar_v = np.sum(v ** 2, axis=1) ** 0.5

    linear_a = np.absolute(np.sum(direction[:-1,:] * a, axis=1))
    rotate_a = np.absolute(np.sum(direction_r[:-1,:] * a, axis=1))

    linear_aa = np.absolute(np.sum(direction[:-2,:] * aa, axis=1))
    rotate_aa = np.absolute(np.sum(direction_r[:-2,:] * aa, axis=1))

    return scalar_v, linear_a, rotate_a, linear_aa, rotate_aa


def get_deviation(perturbation_array):
    return np.sum(perturbation_array ** 2, axis=1) ** 0.5


def hard_constraint(observe_trace_array, perturbation_tensor, hard_bound, physical_bounds):
    if not isinstance(perturbation_tensor, np.ndarray):
        perturbation_array = perturbation_tensor.cpu().detach().numpy()
    else:
        perturbation_array = perturbation_tensor
    
    step = 0.01
    theta = 1 + step
    check_pass = False
    while not check_pass:
        theta -= 0.01
        if theta <= 0.01:
            break
        merged_trace_array = copy.deepcopy(observe_trace_array)
        merged_trace_array[:perturbation_array.shape[0],:] += theta * perturbation_array
        scalar_v, linear_a, rotate_a, linear_aa, rotate_aa = get_metrics(merged_trace_array)
        deviation = get_deviation(theta * perturbation_array)
        check_pass = (np.sum(scalar_v > physical_bounds["scalar_v"]) == 0 and np.sum(linear_a > physical_bounds["linear_a"]) == 0 and np.sum(rotate_a > physical_bounds["rotate_a"]) == 0 and np.sum(linear_aa > physical_bounds["linear_aa"]) == 0 and np.sum(rotate_aa > physical_bounds["rotate_aa"]) == 0 and np.sum(deviation > hard_bound) == 0) 
    return perturbation_tensor * theta


def get_physical_constraints(data_generator):
    max_scalar_v = 0
    max_rotate_a = 0
    max_linear_a = 0
    max_rotate_aa = 0
    max_linear_aa = 0
    
    for input_data in data_generator:
        for _, obj in input_data["objects"].items():
            if obj["type"] not in [1,2]:
                continue
            
            observe_trace = get_trace(obj, "observe")
            future_trace = get_trace(obj, "future")
            predict_trace = get_trace(obj, "predict")

            # update boundaries
            trace_all = observe_trace
            if future_trace is not None:
                trace_all = np.vstack((trace_all, future_trace))

            if trace_all.shape[0] < 4:
                continue

            scalar_v, linear_a, rotate_a, linear_aa, rotate_aa = get_metrics(trace_all)

            max_scalar_v = max(max_scalar_v, np.max(scalar_v))
            max_linear_a = max(max_linear_a, np.max(linear_a))
            max_rotate_a = max(max_rotate_a, np.max(rotate_a))
            max_linear_aa = max(max_linear_aa, np.max(linear_aa))
            max_rotate_aa = max(max_rotate_aa, np.max(rotate_aa))

    return max_scalar_v, max_linear_a, max_rotate_a, max_linear_aa, max_rotate_aa
            