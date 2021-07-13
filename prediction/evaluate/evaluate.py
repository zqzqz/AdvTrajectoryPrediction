from prediction.dataset.generate import *
from .metrics import ade, fde

def multi_frame_prediction(data, api, duration, perturbation=None):
    outputs = {}
    for k in range(duration):
        input_data = input_data_by_attack_step(data, api.obs_length, api.pred_length, k)
        if perturbation is None:
            output_data = api.run(input_data, perturbation=None, mode="eval")
        else:
            raise NotImplementedError()
        outputs[str(k)] = output_data
    return {
        "attack_length": duration,
        "output_data": outputs
    }


def evaluate_error(IN, online=True):
    ade_list = []
    fde_list = []

    if online:
        generator = output_data_online_generator(IN)
    else:
        generator = output_data_offline_generator(IN)
    
    for _, output_data in generator:
        for _, obj in output_data["objects"].items():
            ade_list.append(ade(obj["future_trace"], obj["predict_trace"]))
            fde_list.append(fde(obj["future_trace"], obj["predict_trace"]))

    return ade_list, fde_list