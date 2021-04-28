import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP'))
import logging
import pickle
import random
import torch
from scipy import spatial 
from prediction.base.dataloader import DataLoader
from layers.graph import Graph
import numpy as np

class GRIPDataLoader(DataLoader):
    def __init__(self, dataset, obs_length=6, pred_length=6):
        super().__init__(dataset, obs_length, pred_length)
        self.format_data = self.generate_data
        self.max_num_object = 120
        self.neighbor_distance = 10
        self.total_feature_dimension = 11
        self.graph = Graph(num_node = 120, max_hop = 2)
        self.dev = 'cuda:0' 

    def generate_data(self):
        return self.dataset.format_data_generator(self.dataset.val_data_dir, None)

    def preprocess(self, input_data, perturbation, *args):
        rescale_xy = torch.ones((1,2,1,1)).to(self.dev)
        rescale_xy[:,0] = 1.
        rescale_xy[:,1] = 1.

        # get object index of perturbation target if any
        if perturbation is not None:
            perturbation["obj_index"] = list(input_data["objects"].keys()).index(str(perturbation["obj_id"]))

        # TODO: GRIP maintains invisible objects
        visible_object_id_list = list(input_data["objects"].keys())
        num_visible_object = len(visible_object_id_list)

        # compute the mean values of x and y for zero-centralization. 
        visible_object_value = np.array([input_data["objects"][obj_id]["observe_full_trace"][self.obs_length-1,:] for obj_id in input_data["objects"]])
        xy = visible_object_value[:,:2].astype(float)
        mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
        m_xy = np.mean(xy, axis=0)
        mean_xy[:2] = m_xy

        # compute distance between any pair of two objects
        dist_xy = spatial.distance.cdist(xy, xy)
        # if their distance is less than $neighbor_distance, we regard them are neighbors.
        neighbor_matrix = np.zeros((self.max_num_object, self.max_num_object))
        neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy<self.neighbor_distance).astype(int)

        non_visible_object_id_list = []
        num_non_visible_object = 0
        
        # for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
        object_feature_list = []
        # non_visible_object_feature_list = []
        for frame_ind in range(self.seq_length):
            now_frame_feature_dict = {}
            # we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1] 
            # -mean_xy is used to zero_centralize data
            # now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
            for obj_id in input_data["objects"]:
                if frame_ind < self.obs_length:
                    feature_data = input_data["objects"][obj_id]["observe_full_trace"][frame_ind,:]
                else:
                    feature_data = input_data["objects"][obj_id]["future_full_trace"][frame_ind-self.obs_length,:]
                if obj_id in visible_object_id_list:
                    now_frame_feature_dict[obj_id] = [frame_ind, obj_id, input_data["objects"][obj_id]["type"]]+list(feature_data-mean_xy)+[1]
                else:
                    now_frame_feature_dict[obj_id] = [frame_ind, obj_id, input_data["objects"][obj_id]["type"]]+list(feature_data-mean_xy)+[0]
                # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
            now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(self.total_feature_dimension)) for vis_id in visible_object_id_list+non_visible_object_id_list])
            object_feature_list.append(now_frame_feature)

        # object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
        object_feature_list = np.array(object_feature_list)
        
        # object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
        object_frame_feature = np.zeros((self.max_num_object, self.seq_length, self.total_feature_dimension))
        
        # np.transpose(object_feature_list, (1,0,2))
        object_frame_feature[:num_visible_object+num_non_visible_object] = np.transpose(object_feature_list, (1,0,2))
        
        # result: object_frame_feature, neighbor_matrix, m_xy (function process_data)
        all_feature_list = np.transpose(np.array([object_frame_feature]), (0, 3, 2, 1))
        all_adjacency_list = neighbor_matrix
        all_mean_list = np.array([m_xy])
        # return all_feature_list, all_adjacency_list, all_mean_list

        now_adjacency = self.graph.get_adjacency(all_adjacency_list)
        now_A = self.graph.normalize_adjacency(now_adjacency)

        _ori_data, A, mean_xy = all_feature_list, np.array([now_A]), all_mean_list
        ori_data = torch.from_numpy(_ori_data).cuda()
        A = torch.from_numpy(A).cuda()

        # inject perturbation if any
        if perturbation is not None:
            ori_data[0,3:5,:self.obs_length,perturbation["obj_index"]] += torch.transpose(perturbation["clamp_value"], 0, 1)

        feature_id = [3, 4, 9, 10]
        no_norm_loc_data = ori_data[:,feature_id]
        data = no_norm_loc_data.clone()

        new_mask = (data[:, :2, 1:]!=0) * (data[:, :2, :-1]!=0) 
        data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
        data[:, :2, 0] = 0    
        # # small vehicle: 1, big vehicles: 2, pedestrian 3, bicycle: 4, others: 5
        object_type = ori_data[:,2:3]
        data = data.float().to(self.dev)
        no_norm_loc_data = no_norm_loc_data.float().to(self.dev)
        object_type = object_type.to(self.dev) #type
        data[:,:2] = data[:,:2] / rescale_xy
        # result: data, no_norm_loc_data, object_type (function main.py:preprocess_data)
        
        _input_data = data[:,:,:self.obs_length,:] # (N, C, T, V)=(N, 4, 6, 120)
        output_loc_GT = data[:,:2,self.obs_length:,:] # (N, C, T, V)=(N, 2, 6, 120)
        output_mask = data[:,-1:,self.obs_length:,:] # (N, C, T, V)=(N, 1, 6, 120)
        A = A.float().to(self.dev)

        return _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data

    def postprocess(self, input_data, perturbation, *args):
        predicted, ori_data, mean_xy, rescale_xy, no_norm_loc_data = args

        predicted = predicted * rescale_xy
        ori_output_last_loc = no_norm_loc_data[:,:2,self.obs_length-1,:]
        predicted[:,:2,0,:] = ori_output_last_loc + predicted[:,:2,0,:]
        for ind in range(1, predicted.shape[-2]):
            predicted[:,:,ind,:] = predicted[:,:,ind-1,:] + predicted[:,:,ind,:]

        for n in range(predicted.shape[0]):
            mean_x, mean_y = mean_xy[n,0], mean_xy[n,1]
            predicted[n,0,:,:] += mean_x
            predicted[n,1,:,:] += mean_y

        if perturbation is not None:
            input_data["objects"][str(perturbation["obj_id"])]["observe_trace"] += perturbation["clamp_value"].detach().cpu().numpy()
            predict_trace = torch.transpose(predicted[0,:,:,perturbation["obj_index"]], 0, 1)
            future_trace = torch.from_numpy(input_data["objects"][str(perturbation["obj_id"])]["future_trace"]).cuda()
            # print(future_trace - predict_trace)
            loss = perturbation["loss"](predict_trace, future_trace, perturbation["clamp_value"])
        else:
            loss = None

        predicted = predicted.detach().cpu().numpy()

        now_pred = predicted # (N, C, T, V)=(N, 2, 6, 120)
        now_mean_xy = mean_xy # (N, 2)
        now_ori_data = ori_data # (N, C, T, V)=(N, 11, 6, 120)
        now_mask = now_ori_data[:, -1, -1, :] # (N, V)

        now_pred = np.transpose(now_pred, (0, 2, 3, 1)) # (N, T, V, 2)
        now_ori_data = np.transpose(now_ori_data, (0, 2, 3, 1)) # (N, T, V, 11)

        for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, now_ori_data, now_mask):
            # (6, 120, 2), (2,), (6, 120, 11), (120, )
            num_object = np.sum(n_mask).astype(int)
            # only use the last time of original data for ids (frame_id, object_id, object_type)
            # (6, 120, 11) -> (num_object, 3)
            n_dat = n_data[-1, :num_object, :3].astype(int)
            for time_ind, n_pre in enumerate(n_pred[:, :num_object]):
                # (120, 2) -> (n, 2)
                for info, pred in zip(n_dat, n_pre):
                    information = info.copy()
                    information[0] = information[0] + time_ind
                    input_data["objects"][str(information[1])]["predict_trace"][time_ind,:] = pred
        return input_data, loss