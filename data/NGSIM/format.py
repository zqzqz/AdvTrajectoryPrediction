import pickle
from scipy.io import loadmat
import numpy as np
import os

def ngsim_to_apolloscape(input_filename, output_dir):
    full_data = loadmat(input_filename)
    traj_data = full_data["traj"]
    track_data = full_data["tracks"]
    datasets = np.unique(traj_data[:, 0])
    feet_to_meter = 0.3048

    for dataset_id in datasets:
        dataset_id = int(dataset_id)
        data = traj_data[traj_data[:, 0] == dataset_id]

        start_frame = int(np.min(data[:,2]))
        end_frame = int(np.max(data[:,2]))

        data_output = []
        slice_id = 0
        type_mapping = {1:4, 2:1, 3:2}
        for frame_id in range(start_frame, end_frame+1):
            frame_data = data[data[:,2] == frame_id]

            if (frame_data.shape[0] == 0 and len(data_output) > 0):
                output_file = os.path.join(output_dir, "{:d}-{:d}.txt".format(int(dataset_id), slice_id))
                print("write {}".format(output_file))
                np.savetxt(output_file, np.vstack(data_output), delimiter=' ', fmt="%i %i %i %.4f %.4f %.4f %.4f %.4f %.4f %.4f")
                slice_id += 1
                data_output = []
            
            if frame_data.shape[0] == 0:
                continue

            frame_data_output = np.zeros((frame_data.shape[0],10))
            for i in range(frame_data.shape[0]):
                obj_id = int(frame_data[i,1])
                frame_data_output[i,0] = frame_id
                frame_data_output[i,1] = obj_id
                frame_data_output[i,2] = type_mapping[int(frame_data[i,11])]
                frame_data_output[i,3:5] = frame_data[i,5:7] # * feet_to_meter
                frame_data_output[i,5] = 0
                frame_data_output[i,6:8] = frame_data[i,9:11] # * feet_to_meter
                frame_data_output[i,8] = 1.5

                track = track_data[dataset_id-1][obj_id-1]
                current = track[1:,track[0,:] == frame_id]
                next = track[1:,track[0,:] == frame_id+1]
                prev = track[1:,track[0,:] == frame_id-1]

                delta = None
                if current.shape[1] > 0:
                    if next.shape[1] > 0:
                        delta = (next - current).reshape(2)
                    elif prev.shape[1] > 0:
                        delta = (current - prev).reshape(2)
                if delta is not None:
                    if delta[0] == 0:
                        delta[0] = 0.001
                    heading = np.arctan(delta[1]/delta[0])
                else:
                    heading = 0
                frame_data_output[i,9] = heading

            data_output.append(frame_data_output)

            if frame_id == end_frame:
                output_file = os.path.join(output_dir, "{:d}-{:d}.txt".format(int(dataset_id), slice_id))
                print("write {}".format(output_file))
                np.savetxt(output_file, np.vstack(data_output), delimiter=' ', fmt="%i %i %i %.4f %.4f %.4f %.4f %.4f %.4f %.4f")


def preprocess(input_dir, output_dir, enable_downsample, enable_rescale):
    os.makedirs(output_dir, exist_ok=True)
    for data_file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, data_file)
        print(file_path)
        data = np.genfromtxt(file_path, delimiter=" ")
        dim = data.shape[1]

        if enable_downsample:
            min_frame = int(np.min(data[:,0]))
            indexes = (data[:,0].astype(int) - min_frame) % 2 == 0
            data = data[indexes,:]
            data[:,0] -= min_frame
            data[:,0] /= 2

        if enable_rescale:
            if dim == 10:
                data[:,3:9] *= 0.3048
            else:
                data[:,3:5] *= 0.3048

        output_file_path = os.path.join(output_dir, data_file)
        if dim == 10:
            np.savetxt(output_file_path, data, delimiter=' ', fmt="%i %i %i %.4f %.4f %.4f %.4f %.4f %.4f %.4f")
        else:
            np.savetxt(output_file_path, data, delimiter=' ', fmt="%i %i %i %.4f %.4f %.4f %.4f %.4f")



ngsim_to_apolloscape("TrainSet.mat", "prediction_train")
ngsim_to_apolloscape("ValSet.mat", "prediction_val")
ngsim_to_apolloscape("TestSet.mat", "prediction_test")


for tag in ["train", "val", "test"]:
    preprocess("prediction_{}".format(tag), "prediction_{}.new".format(tag), True, True)

