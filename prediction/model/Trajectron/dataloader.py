import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectron-plus-plus/trajectron'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectron-plus-plus/experiments/nuScenes'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectron-plus-plus/experiments/nuScenes/devkit/python-sdk'))
import logging
import pickle
import random
import torch
from scipy import spatial 
from prediction.model.base.dataloader import DataLoader
import numpy as np
import pandas as pd
from environment import Environment, Scene, Node, GeometricMap, derivative_of
from pyquaternion import Quaternion
from kalman_filter import NonlinearKinematicBicycle


class TrajectronDataLoader(DataLoader):
    def __init__(self, obs_length=6, pred_length=6, maps=None):
        super().__init__(obs_length, pred_length)
        self.maps = maps

    @staticmethod
    def input_data_to_ndarray(input_data):
        obs_length, pred_length = input_data["observe_length"], input_data["predict_length"]
        data = []

        for frame_id in range(obs_length):
            for obj_id, obj in input_data["objects"].items():
                if obj["observe_mask"][frame_id] < 1:
                    continue
                # if obj["static"]:
                #     continue
                data.append([frame_id, int(obj_id), obj["type"], obj["observe_trace"][frame_id,:], obj["observe_feature"][frame_id,:]])

        for frame_id in range(pred_length):
            for obj_id, obj in input_data["objects"].items():
                if obj["future_mask"][frame_id] < 1:
                    continue
                if obj["static"]:
                    continue
                data.append([frame_id+obs_length, int(obj_id), obj["type"], obj["future_trace"][frame_id,:], obj["future_feature"][frame_id,:]])

        np_data = np.zeros((len(data), 3+2+input_data["feature_dimension"]))
        for i, d in enumerate(data):
            np_data[i,0] = d[0]
            np_data[i,1] = d[1]
            np_data[i,2] = d[2]
            np_data[i,3:5] = d[3]
            np_data[i,5:] = d[4]
        return np_data

    @staticmethod
    def input_data_to_dataframe(input_data, env):
        obs_length, pred_length = input_data["observe_length"], input_data["predict_length"]
        data = []

        for frame_id in range(obs_length):
            for obj_id, obj in input_data["objects"].items():
                if obj["observe_mask"][frame_id] < 1:
                    continue
                # if obj["static"]:
                #     continue
                data.append([frame_id, int(obj_id), obj["type"], obj["observe_trace"][frame_id,:], obj["observe_feature"][frame_id,:]])

        for frame_id in range(pred_length):
            for obj_id, obj in input_data["objects"].items():
                if obj["future_mask"][frame_id] < 1:
                    continue
                # if obj["static"]:
                #     continue
                data.append([frame_id+obs_length, int(obj_id), obj["type"], obj["future_trace"][frame_id,:], obj["future_feature"][frame_id,:]])

        df_data = pd.DataFrame(columns=['frame_id',
                                    'type',
                                    'node_id',
                                    'robot',
                                    'x', 'y', 'z',
                                    'length',
                                    'width',
                                    'height',
                                    'heading'])

        for d in data:
            data_point = pd.Series({'frame_id': int(d[0]),
                                    'type': env.NodeType.PEDESTRIAN if d[2] == 3 else env.NodeType.VEHICLE,
                                    'node_id': str(d[1]),
                                    'robot': True if d[1] == 0 else False,
                                    'x': d[3][0],
                                    'y': d[3][1],
                                    'z': d[4][0],
                                    'length': d[4][1],
                                    'width': d[4][2],
                                    'height': d[4][3],
                                    'heading': d[4][4]})

            df_data = df_data.append(data_point, ignore_index=True)

        return df_data

    @staticmethod
    def trajectory_curvature(t):
        path_distance = np.linalg.norm(t[-1] - t[0])

        lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
        path_length = np.sum(lengths)
        if np.isclose(path_distance, 0.):
            return 0, 0, 0
        return (path_length / path_distance) - 1, path_length, path_distance

    def preprocess(self, input_data, env=None):
        scene_name = input_data["scene_name"] if "scene_name" in input_data else "none"
        scene = Scene(timesteps=self.seq_length, dt=float(input_data["time_step"]), name=scene_name, aug_func=None)
        curv_0_2 = 0
        curv_0_1 = 0
        total = 0

        data = self.input_data_to_dataframe(input_data, env)
        x_min = np.round(data['x'].min() - 50)
        x_max = np.round(data['x'].max() + 50)
        y_min = np.round(data['y'].min() - 50)
        y_max = np.round(data['y'].max() + 50)

        map_name = input_data["map_name"] if "map_name" in input_data else None
        use_map = self.maps is not None and map_name is not None and map_name in self.maps
        if use_map:
            nusc_map = self.maps[map_name]
            type_map = dict()
            x_size = x_max - x_min
            y_size = y_max - y_min
            patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
            patch_angle = 0  # Default orientation where North is up
            canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
            homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
            layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line',
                        'ped_crossing', 'stop_line', 'ped_crossing', 'walkway']
            map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(
                np.uint8)
            map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
            # PEDESTRIANS
            map_mask_pedestrian = np.stack((map_mask[9], map_mask[8], np.max(map_mask[:3], axis=0)), axis=0)
            type_map['PEDESTRIAN'] = GeometricMap(data=map_mask_pedestrian, homography=homography, description=', '.join(layer_names))
            # VEHICLES
            map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)
            type_map['VEHICLE'] = GeometricMap(data=map_mask_vehicle, homography=homography, description=', '.join(layer_names))

            map_mask_plot = np.stack(((np.max(map_mask[:3], axis=0) - (map_mask[3] + 0.5 * map_mask[4]).clip(
                max=255)).clip(min=0).astype(np.uint8), map_mask[8], map_mask[9]), axis=0)
            type_map['VISUALIZATION'] = GeometricMap(data=map_mask_plot, homography=homography, description=', '.join(layer_names))

            scene.map = type_map
            del map_mask
            del map_mask_pedestrian
            del map_mask_vehicle
            del map_mask_plot

        for node_id in pd.unique(data['node_id']):
            node_frequency_multiplier = 1
            node_df = data[data['node_id'] == node_id]

            if node_df['x'].shape[0] < 2:
                continue

            if not np.all(np.diff(node_df['frame_id']) == 1):
                # print('Occlusion')
                continue  # TODO Make better

            node_values = node_df[['x', 'y']].values
            x = node_values[:, 0]
            y = node_values[:, 1]
            heading = node_df['heading'].values
            if node_df.iloc[0]['type'] == env.NodeType.VEHICLE and not node_id == 0:
                # Kalman filter Agent
                vx = derivative_of(x, scene.dt)
                vy = derivative_of(y, scene.dt)
                velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

                filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
                P_matrix = None
                for i in range(len(x)):
                    if i == 0:  # initalize KF
                        # initial P_matrix
                        P_matrix = np.identity(4)
                    elif i < len(x):
                        # assign new est values
                        x[i] = x_vec_est_new[0][0]
                        y[i] = x_vec_est_new[1][0]
                        heading[i] = x_vec_est_new[2][0]
                        velocity[i] = x_vec_est_new[3][0]

                    if i < len(x) - 1:  # no action on last data
                        # filtering
                        x_vec_est = np.array([[x[i]],
                                            [y[i]],
                                            [heading[i]],
                                            [velocity[i]]])
                        z_new = np.array([[x[i + 1]],
                                        [y[i + 1]],
                                        [heading[i + 1]],
                                        [velocity[i + 1]]])
                        x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                            x_vec_est=x_vec_est,
                            u_vec=np.array([[0.], [0.]]),
                            P_matrix=P_matrix,
                            z_new=z_new
                        )
                        P_matrix = P_matrix_new

                curvature, pl, _ = self.trajectory_curvature(np.stack((x, y), axis=-1))
                if pl < 1.0:  # vehicle is "not" moving
                    x = x[0].repeat(self.seq_length)
                    y = y[0].repeat(self.seq_length)
                    heading = heading[0].repeat(self.seq_length)

                total += 1
                if pl > 1.0:
                    if curvature > .2:
                        curv_0_2 += 1
                        node_frequency_multiplier = 3*int(np.floor(total/curv_0_2))
                    elif curvature > .1:
                        curv_0_1 += 1
                        node_frequency_multiplier = 3*int(np.floor(total/curv_0_1))

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
            data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
            data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

            data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

            if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
                v = np.stack((vx, vy), axis=-1)
                v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
                heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
                heading_x = heading_v[:, 0]
                heading_y = heading_v[:, 1]

                data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay,
                            ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                            ('heading', 'x'): heading_x,
                            ('heading', 'y'): heading_y,
                            ('heading', '°'): heading,
                            ('heading', 'd°'): derivative_of(heading, scene.dt, radian=True)}
                node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
            else:
                data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay}
                node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
            node.first_timestep = node_df['frame_id'].iloc[0]
            if node_df.iloc[0]['robot'] == True:
                node.is_robot = True
                scene.robot = node

            scene.nodes.append(node)
        
        return scene

    def postprocess(self, input_data, predictions_np, nodes, timesteps_o):
        predictions_dict = {}
        # Assign predictions to node
        for i, ts in enumerate(timesteps_o):
            if ts not in predictions_dict.keys():
                predictions_dict[ts] = dict()
            predictions_dict[ts][nodes[i].id] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        for obj_id in predictions_dict[self.obs_length-1]:
            if obj_id in input_data["objects"]:
                input_data["objects"][obj_id]["predict_trace"] = predictions_dict[self.obs_length-1][obj_id][0,0]

        return input_data
