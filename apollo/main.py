# from search import Searcher
from draw import Visualizer
import os, sys

cfg = {
    "root": "/apollo/eval_data",
    "history_length": 20,
    "prediction_length": 30,
    "record_path": "/apollo/modules/prediction/eval_data/test2.record",
    "cmd": "/apollo/bazel-bin/modules/prediction/evaluate_prediction",
    "point_num": 3
}
cfg["perturbation_path"] = os.path.join(cfg["root"], "perturbation.txt")
cfg["trajectory_path"] = os.path.join(cfg["root"], "trajectories")
cfg["history_trajectory_path"] = os.path.join(cfg["root"], "trajectories/history.pb.txt")
cfg["evaluate_trajectory_path"] = os.path.join(cfg["root"], "trajectories/evaluate.pb.txt")
cfg["predict_trajectory_path"] = os.path.join(cfg["root"], "trajectories/predict.pb.txt")

def main():
    # searcher = Searcher(cfg)
    visualizer = Visualizer(cfg)

    # print(searcher.run())


if __name__ == "__main__":
    main()
