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