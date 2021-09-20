import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP'))
import torch
import torch.optim as optim
import pickle
import time
import warnings
import itertools
import numpy as np
import logging
import copy
from datetime import datetime

from .dataloader import GRIPDataLoader
from prediction.model.base.interface import Interface
from model import Model
from main import my_load_model, compute_RMSE, display_result

logger = logging.getLogger(__name__)


class GRIPInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=None, max_hop=2, num_node=120, in_channels=4, rescale=[1,1]):
        super().__init__(obs_length, pred_length)

        self.graph_args = {'max_hop':max_hop, 'num_node':num_node}
        self.dataloader = GRIPDataLoader(
            self.obs_length, self.pred_length, graph_args=self.graph_args
        )

        self.dev = 'cuda:0'
        if pre_load_model is not None:
            self.model = self.load_model(self.default_model(in_channels=in_channels), pre_load_model)
        else:
            self.model = None

        self.rescale = rescale

    def default_model(self, in_channels=4):
        model = Model(in_channels=in_channels, graph_args=self.graph_args, edge_importance_weighting=True)
        model.to(self.dev)
        return model

    def load_model(self, model, model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
        logger.warn('Successfull loaded from {}'.format(model_path))
        return model

    def save_model(self, model, model_path):
        torch.save(
            {
                'xin_graph_seq2seq_model': model.state_dict(),
            }, 
            model_path)
        logger.warn("Model saved to {}".format(model_path))

    def run(self, input_data, perturbation=None, backward=False):
        assert(self.model is not None)

        if not backward:
            self.model.eval()
        else:
            self.model.train()
        _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask = self.dataloader.preprocess(input_data, perturbation, self.rescale[0], self.rescale[1])
        predicted = self.model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
        output_data, loss = self.dataloader.postprocess(input_data, perturbation, predicted, _ori_data, mean_xy, rescale_xy, no_norm_loc_data)

        if loss is None:
            return output_data
        else:
            return output_data, loss

    def train(self, total_epoch=200, batch_size=64, save_dir=""):
        assert(self.dataset is not None)

        model = self.default_model()
        optimizer = optim.Adam(
            [{'params':model.parameters()},],) # lr = 0.0001)
            
        best_epoch = -1
        best_result = 100000
        for now_epoch in range(total_epoch):
            iteration = 0
            for batch in self.dataset.data_generator("train", batch_size=batch_size, random_order=True):
                iteration += 1

                batch_processed_data = []
                for input_data in batch:
                    _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask = self.dataloader.preprocess(input_data, None)
                    batch_processed_data.append([_input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask])
                
                _input_data = torch.cat([i[0] for i in batch_processed_data], axis=0)
                A = torch.cat([i[1] for i in batch_processed_data], axis=0)
                _ori_data = np.concatenate([i[2] for i in batch_processed_data], axis=0)
                mean_xy = np.concatenate([i[3] for i in batch_processed_data], axis=0)
                rescale_xy = torch.cat([i[4] for i in batch_processed_data], axis=0)
                no_norm_loc_data = torch.cat([i[5] for i in batch_processed_data], axis=0)
                output_loc_GT = torch.cat([i[6] for i in batch_processed_data], axis=0)
                output_mask = torch.cat([i[7] for i in batch_processed_data], axis=0)
                
                predicted = model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)

                overall_sum_time, overall_num, _ = compute_RMSE(predicted, output_loc_GT, output_mask, pra_error_order=1)
                total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num), torch.ones(1,).to(self.dev)) #(1,)

                now_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                logger.warn('|{}|Epoch:{:>4}/{:>4}|\tIteration:{:>5}|\tLoss:{:.8f}|lr: {}|'.format(datetime.now(), now_epoch, total_epoch, iteration, total_loss.data.item(), now_lr))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            all_overall_sum_list = []
            all_overall_num_list = []
            all_car_sum_list = []
            all_car_num_list = []
            all_human_sum_list = []
            all_human_num_list = []
            all_bike_sum_list = []
            all_bike_num_list = []
            for batch in self.dataset.data_generator("val", batch_size=32, random_order=False):
                batch_processed_data = []
                for input_data in batch:
                    _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask = self.dataloader.preprocess(input_data, None)
                    batch_processed_data.append([_input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask])
                
                _input_data = torch.cat([i[0] for i in batch_processed_data], axis=0)
                A = torch.cat([i[1] for i in batch_processed_data], axis=0)
                _ori_data = torch.from_numpy(np.concatenate([i[2] for i in batch_processed_data], axis=0))
                mean_xy = np.concatenate([i[3] for i in batch_processed_data], axis=0)
                rescale_xy = torch.cat([i[4] for i in batch_processed_data], axis=0)
                no_norm_loc_data = torch.cat([i[5] for i in batch_processed_data], axis=0)
                output_loc_GT = torch.cat([i[6] for i in batch_processed_data], axis=0)
                output_mask = torch.cat([i[7] for i in batch_processed_data], axis=0)

                ori_output_loc_GT = no_norm_loc_data[:,:2,self.obs_length:,:]
                ori_output_last_loc = no_norm_loc_data[:,:2,self.obs_length-1:self.obs_length,:]
                
                # for categories
                cat_mask = _ori_data[:,2:3, self.obs_length:, :] # (N, C, T, V)=(N, 1, 6, 120)
                
                predicted = model(pra_x=_input_data, pra_A=A, pra_pred_length=self.pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
                predicted = predicted * rescale_xy

                overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask)
                all_overall_num_list.extend(overall_num.detach().cpu().numpy())
                # x2y2 (N, 6, 39)
                now_x2y2 = x2y2.detach().cpu().numpy()
                now_x2y2 = now_x2y2.sum(axis=-1)
                all_overall_sum_list.extend(now_x2y2)

                ### car dist
                car_mask = (((cat_mask==1)+(cat_mask==2))>0).float().to(self.dev)
                car_mask = output_mask * car_mask
                car_sum_time, car_num, car_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, car_mask)        
                all_car_num_list.extend(car_num.detach().cpu().numpy())
                # x2y2 (N, 6, 39)
                car_x2y2 = car_x2y2.detach().cpu().numpy()
                car_x2y2 = car_x2y2.sum(axis=-1)
                all_car_sum_list.extend(car_x2y2)

                ### human dist
                human_mask = (cat_mask==3).float().to(self.dev)
                human_mask = output_mask * human_mask
                human_sum_time, human_num, human_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, human_mask)        
                all_human_num_list.extend(human_num.detach().cpu().numpy())
                # x2y2 (N, 6, 39)
                human_x2y2 = human_x2y2.detach().cpu().numpy()
                human_x2y2 = human_x2y2.sum(axis=-1)
                all_human_sum_list.extend(human_x2y2)

                ### bike dist
                bike_mask = (cat_mask==4).float().to(self.dev)
                bike_mask = output_mask * bike_mask
                bike_sum_time, bike_num, bike_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, bike_mask)        
                all_bike_num_list.extend(bike_num.detach().cpu().numpy())
                # x2y2 (N, 6, 39)
                bike_x2y2 = bike_x2y2.detach().cpu().numpy()
                bike_x2y2 = bike_x2y2.sum(axis=-1)
                all_bike_sum_list.extend(bike_x2y2)

            result_car = display_result([np.array(all_car_sum_list), np.array(all_car_num_list)], pra_pref='car')
            result_human = display_result([np.array(all_human_sum_list), np.array(all_human_num_list)], pra_pref='human')
            result_bike = display_result([np.array(all_bike_sum_list), np.array(all_bike_num_list)], pra_pref='bike')

            all_overall_sum_list = np.array(all_overall_sum_list)
            all_overall_num_list = np.array(all_overall_num_list)
            display_result(
                [all_overall_sum_list, all_overall_num_list],
                pra_pref='{}_Epoch{}'.format('Test', now_epoch)
            )

            result = 0.20*result_car + 0.58*result_human + 0.22*result_bike
            overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), 'WS', ' '.join(['{:.3f}'.format(x) for x in list(result) + [np.sum(result)]]))
            logger.warn(overall_log)
            result = np.sum(result)

            if result < best_result:
                best_result = result
                best_epoch = now_epoch
            logger.warn("Best epoch {}".format(best_epoch))

            save_path = os.path.join(save_dir, "model_epoch_{:04}.pt".format(now_epoch))
            self.save_model(model, save_path)
