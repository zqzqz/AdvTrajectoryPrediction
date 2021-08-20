from prediction.dataset.generate import input_data_by_attack_step


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

