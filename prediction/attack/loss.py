import torch

def ade(predict_trace, future_trace):
    return (torch.sum(torch.square(predict_trace - future_trace)) / predict_trace.shape[0])


def fde(predict_trace, future_trace):
    return torch.sum(torch.square(predict_trace[-1,:] - future_trace[-1,:]))


def perturbation_cost(perturbation):
    return torch.sum(torch.square(torch.absolute(perturbation)+1))


def physical_constraint(observe_trace):
    v = observe_trace[1:,:] - observe_trace[:-1,:]
    dif = v[1:,:] - v[:-1,:]
    return torch.sum(torch.square(dif))


def perturbation_physical_constraint(observe_trace, perturbed_trace):
    return physical_constraint(perturbed_trace) - physical_constraint(observe_trace)


def interpolation(trace, inject_num=3):
    extended_trace = torch.zeros((trace.shape[0] -1) * inject_num + trace.shape[0], 2).cuda()
    for i in range(extended_trace.shape[0]):
        if i % (inject_num + 1) == 0:
            index = i // (inject_num + 1)
            extended_trace[i,:] = trace[index,:]
        else:
            start_index = i // (inject_num + 1)
            end_index = start_index + 1
            extended_trace[i,:] = (trace[end_index,:] - trace[start_index,:]) / (inject_num + 1) * (i - start_index * (inject_num + 1)) + trace[start_index,:]
    return extended_trace


def square_distance(point1, point2):
    return torch.sum(torch.square(point1 - point2))


def change_lane_attack_goal(predict_traces, future_traces, obj_id, **attack_opts):
    attacker_predict_trace = predict_traces[obj_id]
    # attacker_future_trace = future_traces[obj_id]
    victim_predict_trace = predict_traces[attack_opts["target_obj_id"]]
    extended_attacker_predict_trace = interpolation(attacker_predict_trace)
    extended_victim_predict_trace = interpolation(victim_predict_trace)
    # extended_attacker_future_trace = interpolation(attacker_future_trace)

    distance1 = torch.min(torch.sum(torch.square(extended_attacker_predict_trace - extended_victim_predict_trace), 1))
    distance2 = torch.min(torch.cdist(extended_attacker_predict_trace, extended_victim_predict_trace, p=2))
    return distance1 + distance2


def horizonal_distance(observe_trace, predict_trace, future_trace):
    offset = predict_trace - future_trace
    direction = (future_trace - 
                 torch.cat(
                   (torch.reshape(observe_trace[-1,:], (1,2)), 
                    future_trace[:-1,:]), 0)).float()
    scale = torch.sqrt(torch.sum(torch.square(direction), 1)).float()
    right_direction = torch.matmul(
                        torch.tensor([[0., 1.], [-1., 0.]]).float().to("cuda"),
                        direction.t().float() / scale).t()
    average_distance = torch.sum(offset * right_direction) / predict_trace.shape[0]
    return average_distance


def vertical_distance(observe_trace, predict_trace, future_trace):
    offset = predict_trace - future_trace
    direction = (future_trace - 
                 torch.cat(
                   (torch.reshape(observe_trace[-1,:], (1,2)), 
                    future_trace[:-1,:]), 0)).float()
    scale = torch.sqrt(torch.sum(torch.square(direction), 1)).float()
    average_distance = torch.sum(offset * (direction.t().float() / scale).t()) / predict_trace.shape[0]
    return average_distance


def attack_loss(observe_traces, future_traces, predict_traces, obj_id, perturbation, **attack_opts):
    if "perturbation_cost_c" not in attack_opts:
        attack_opts["perturbation_cost_c"] = 0.1
    if "physical_constraint_c" not in attack_opts:
        attack_opts["physical_constraint_c"] = 0.1
    if "attack_goal_c" not in attack_opts:
        attack_opts["attack_goal_c"] = 1


    # attacker_observe_trace = observe_traces[obj_id]
    # attacker_perturbed_trace = attacker_observe_trace + perturbation
    # loss = attack_opts["perturbation_cost_c"] * perturbation_cost(perturbation) + attack_opts["physical_constraint_c"] * perturbation_physical_constraint(attacker_observe_trace, attacker_perturbed_trace)
    loss = 0

    if "type" in attack_opts:
        attack_goal = attack_opts["type"]
        if attack_goal == "ade":
            loss -= attack_opts["attack_goal_c"] * ade(predict_traces[obj_id], future_traces[obj_id])
        elif attack_goal == "fde":
            loss -= attack_opts["attack_goal_c"] * fde(predict_traces[obj_id], future_traces[obj_id])
        elif attack_goal == "left":
            loss += attack_opts["attack_goal_c"] * horizonal_distance(observe_traces[obj_id], predict_traces[obj_id], future_traces[obj_id])
        elif attack_goal == "right":
            loss -= attack_opts["attack_goal_c"] * horizonal_distance(observe_traces[obj_id], predict_traces[obj_id], future_traces[obj_id])
        elif attack_goal == "front":
            loss -= attack_opts["attack_goal_c"] * vertical_distance(observe_traces[obj_id], predict_traces[obj_id], future_traces[obj_id])
        elif attack_goal == "rear":
            loss += attack_opts["attack_goal_c"] * vertical_distance(observe_traces[obj_id], predict_traces[obj_id], future_traces[obj_id])
        else:
            raise NotImplementedError()

    return loss
