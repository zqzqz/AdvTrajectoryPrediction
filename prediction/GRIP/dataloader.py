import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP'))
import logging
import pickle
import random
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

    def generate_data(self):
        return self.dataset.format_data_generator(self.dataset.val_data_dir, None)

    def preprocess(self, input_data):
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

        return all_feature_list, np.array([now_A]), all_mean_list

    def postprocess(self, input_data, ori_data, mean_xy, predicted):
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
                for info, pred in zip(n_dat, n_pre+np.repeat(n_mean_xy.reshape((1,2)), repeats=self.pred_length, axis=0)):
                    information = info.copy()
                    information[0] = information[0] + time_ind
                    input_data["objects"][information[1]]["predict_trace"][time_ind,:] = pred
                    print(information, pred)
        return input_data