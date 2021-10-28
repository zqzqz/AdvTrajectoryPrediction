from prediction.dataset.generate import input_data_by_attack_step
import torch
import numpy as np


def multi_frame_prediction(data, api, duration):
    outputs = {}
    for k in range(duration):
        input_data = input_data_by_attack_step(data, api.obs_length, api.pred_length, k)
        output_data = api.run(input_data, perturbation=None, backward=False)
        outputs[str(k)] = output_data
    return {
        "attack_length": duration,
        "output_data": outputs
    }

def smooth_tensor(feature_tensor):
    new_tensor = torch.zeros_like(feature_tensor).cuda()
    new_tensor += feature_tensor
    new_tensor[1:feature_tensor.shape[0]-1] += (feature_tensor[:-2] + feature_tensor[2:])
    new_tensor[1:feature_tensor.shape[0]-1] /= 3
    return new_tensor.float()


def smooth_array(feature_array):
    new_array = np.copy(feature_array)
    new_array[1:feature_array.shape[0]-1] += (feature_array[:-2] + feature_array[2:])
    new_array[1:feature_array.shape[0]-1] /= 3
    return new_array


def get_unit_vector(vectors):
    scale = np.sum(vectors ** 2, axis=1) ** 0.5 + 0.001
    result = np.zeros(vectors.shape)
    result[:,0] = vectors[:,0] / scale
    result[:,1] = vectors[:,1] / scale
    return result


def get_acceleration(trace_array):
    v = trace_array[1:,:] - trace_array[:-1,:]
    a = v[1:,:] - v[:-1,:]

    direction = get_unit_vector(v)
    direction_r = np.concatenate((direction[:,1].reshape(direction.shape[0],1), 
                                -direction[:,0].reshape(direction.shape[0],1)), axis=1)

    long_a = np.sum(direction[:-1,:] * a, axis=1)
    lat_a = np.sum(direction_r[:-1,:] * a, axis=1)

    return long_a, lat_a


def CUSUM(trace_array, opts):
    long_a, lat_a = get_acceleration(trace_array)
    long_opts, lat_opts = opts["long"], opts["lat"]
    
    result = False
    for opts, a in [(long_opts, long_a), (lat_opts, lat_a)]:
        s = 0
        last_m = 0
        for m in a.tolist():
            if m * last_m < 0:
                s = max(0, s + abs(m-last_m)/opts["scale"] - opts["d"])
            last_m = m
        # print(s)
        if s > opts["t"]:
            result = True
    
    return result


def variance_based_detect(trace_array, thres):
    v = trace_array[1:,:] - trace_array[:-1,:]
    a = v[1:,:] - v[:-1,:]
    mean_a = np.mean(a, axis=0)
    dist_a = a - np.tile(mean_a, (a.shape[0],1))
    var_a = np.sum(np.sum(dist_a ** 2, axis=1)) / a.shape[0]
    mean_scalar_v = np.sum(v ** 2) / a.shape[0]
    var_a_rescale = var_a / mean_scalar_v
    return var_a_rescale > thres


def detect_array(trace_array, opts):
    return variance_based_detect(trace_array, opts["thres"])


def detect_tensor(trace_tensor, opts):
    return variance_based_detect(trace_tensor.cpu().detach().numpy(), opts["thres"])