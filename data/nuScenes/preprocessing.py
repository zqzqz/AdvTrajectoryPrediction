import sys
import os
trajectron_path = "../../prediction/model/Trajectron/Trajectron-plus-plus"
sys.path.append(os.path.join(trajectron_path, 'experiments/nuScenes'))
sys.path.append(os.path.join(trajectron_path, 'experiments/nuScenes/devkit/python-sdk/'))
sys.path.append(os.path.join(trajectron_path, "trajectron"))

import numpy as np
import pandas as pd
import dill
import argparse
import pickle
from tqdm import tqdm
from pyquaternion import Quaternion
from kalman_filter import NonlinearKinematicBicycle
from sklearn.model_selection import train_test_split

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes
from environment import Environment, Scene, Node, GeometricMap, derivative_of

scene_blacklist = [499, 515, 517]

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

curv_0_2 = 0
curv_0_1 = 0
total = 0

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

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
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}

            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep,
                        non_aug_node=node)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def process_scene(ns_scene, env, nusc, data_path):
    scene_id = int(ns_scene['name'].replace('scene-', ''))
    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])

    sample_token = ns_scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    frame_id = 0
    while sample['next']:
        annotation_tokens = sample['anns']
        for annotation_token in annotation_tokens:
            annotation = nusc.get('sample_annotation', annotation_token)
            category = annotation['category_name']
            if len(annotation['attribute_tokens']):
                attribute = nusc.get('attribute', annotation['attribute_tokens'][0])['name']
            else:
                continue

            if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category:
                our_category = 3
            elif 'parked' in attribute:
                continue
            elif 'bicycle' in category or 'motorcycle' in category:
                our_category = 4
            elif "bus" in category or "construction" in category or "truck" in category or "tailer" in category:
                our_category = 2
            elif 'vehicle' in category:
                our_category = 1
            else:
                continue
                

            data_point = pd.Series({'frame_id': frame_id,
                                    'type': our_category,
                                    'node_id': annotation['instance_token'],
                                    'robot': False,
                                    'x': annotation['translation'][0],
                                    'y': annotation['translation'][1],
                                    'z': annotation['translation'][2],
                                    'length': annotation['size'][0],
                                    'width': annotation['size'][1],
                                    'height': annotation['size'][2],
                                    'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0]})
            data = data.append(data_point, ignore_index=True)

        # Ego Vehicle
        our_category = 1
        sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        annotation = nusc.get('ego_pose', sample_data['ego_pose_token'])
        data_point = pd.Series({'frame_id': frame_id,
                                'type': our_category,
                                'node_id': 'ego',
                                'robot': True,
                                'x': annotation['translation'][0],
                                'y': annotation['translation'][1],
                                'z': annotation['translation'][2],
                                'length': 4,
                                'width': 1.7,
                                'height': 1.5,
                                'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
                                'orientation': None})
        data = data.append(data_point, ignore_index=True)

        sample = nusc.get('sample', sample['next'])
        frame_id += 1

    if len(data.index) == 0:
        return None

    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()

    x_min = np.round(data['x'].min() - 50)
    x_max = np.round(data['x'].max() + 50)
    y_min = np.round(data['y'].min() - 50)
    y_max = np.round(data['y'].max() + 50)

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    map_name = nusc.get('log', ns_scene['log_token'])['location']
    map_coordinates = [x_min, x_max, y_min, y_max]

    return data, map_name, map_coordinates
    

def process_data(data_path, version, output_path, val_split):
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    splits = create_splits_scenes()
    train_scenes, val_scenes = train_test_split(splits['train' if 'mini' not in version else 'mini_train'], test_size=val_split)
    train_scene_names = splits['train' if 'mini' not in version else 'mini_train']
    val_scene_names = val_scenes
    test_scene_names = splits['val' if 'mini' not in version else 'mini_val']

    ns_scene_names = dict()
    ns_scene_names['train'] = train_scene_names
    ns_scene_names['val'] = val_scene_names
    ns_scene_names['test'] = test_scene_names

    os.makedirs(os.path.join(output_path, "scene_maps"), exist_ok=True)

    for data_class in ['train', 'val', 'test']:
        map_name_path = os.path.join(output_path, "map_name.txt")

        env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        scenes = []

        for ns_scene_name in tqdm(ns_scene_names[data_class]):
            ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', ns_scene_name)[0])
            scene_id = int(ns_scene['name'].replace('scene-', ''))
            if scene_id in scene_blacklist:  # Some scenes have bad localization
                continue

            scene_data, map_name, map_coordinates = process_scene(ns_scene, env, nusc, data_path)

            scene_data_path = os.path.join(output_path, "prediction_" + data_class, ns_scene_name + ".txt")
            with open(scene_data_path, 'w') as f:
                obj_ids = ["ego"]
                for index, row in scene_data.iterrows():
                    if row["node_id"] not in obj_ids:
                        obj_ids.append(row["node_id"])
                    obj_id = obj_ids.index(row["node_id"])

                    f.write("{:d} {:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(row["frame_id"], obj_id, row["type"], 
                                                                                                     row['x'], row['y'], row['z'],
                                                                                                     row["length"], row["width"], row["height"],
                                                                                                     row["heading"]))
                with open(map_name_path, 'a') as f:
                    f.write("{} {} {} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(data_class, ns_scene_name, map_name, *map_coordinates))

        print(f'Processed {len(scenes):.2f} scenes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--val_split', type=int, default=0.15)
    args = parser.parse_args()
    process_data(args.data, args.version, args.output_path, args.val_split)
