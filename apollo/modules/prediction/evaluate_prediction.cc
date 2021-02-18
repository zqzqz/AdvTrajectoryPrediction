#include <math.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <string>

#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "cyber/record/record_viewer.h"
#include "modules/common/configs/config_gflags.h"
#include "modules/prediction/common/prediction_gflags.h"
#include "modules/prediction/common/semantic_map.h"
#include "modules/prediction/container/container_manager.h"
#include "modules/prediction/container/obstacles/obstacles_container.h"
#include "modules/prediction/container/pose/pose_container.h"
#include "modules/prediction/evaluator/evaluator_manager.h"
#include "modules/prediction/predictor/predictor_manager.h"

std::string default_map_dir = "/apollo/modules/prediction/eval_data";
std::string default_base_map_filename = "map.bin";
std::string default_record_file =
    "/apollo/modules/prediction/eval_data/test2.record";
std::string default_adapter_conf_file =
    "/apollo/modules/prediction/eval_data/adapter_conf.pb.txt";
std::string default_prediction_conf_file =
    "/apollo/modules/prediction/eval_data/prediction_conf.pb.txt";
std::string default_perturbation_file = "/apollo/eval_data/perturbation.txt";

using apollo::common::adapter::AdapterConfig;
using apollo::cyber::record::RecordMessage;
using apollo::cyber::record::RecordReader;
using apollo::cyber::record::RecordViewer;

#define PI 3.1415926

namespace apollo {
namespace prediction {

class Perturbation {
 public:
  std::vector<std::vector<double>> D;

  void load(std::string filename) {
    D.clear();
    std::ifstream f(filename);
    if (f.is_open()) {
      std::string line;
      while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::vector<double> p;
        std::string token;
        while (std::getline(ss, token, ',')) {
          p.push_back(std::stod(token));
        }
        D.push_back(p);
      }
    }
    AWARN << "Perturbation loaded " << D.size();
  }

  void init(unsigned int steps) {
    D.clear();
    for (unsigned int i = 0; i < steps; i++) {
      std::vector<double> line(3, 0);
      D.push_back(line);
    }
  }
};

class EvaluatePrediction {
 public:
  EvaluatePrediction() {
    FLAGS_ego_vehicle_id = -1;
    AWARN << "ego obstacle id" << FLAGS_ego_vehicle_id;
    FLAGS_half_vehicle_width = 0.1;

    cyber::common::GetProtoFromFile(default_adapter_conf_file, &adapter_conf_);
    cyber::common::GetProtoFromFile(default_prediction_conf_file,
                                    &prediction_conf_);
    // container manager
    ContainerManager::Instance()->Init(adapter_conf_);
    // evaluator manager
    EvaluatorManager::Instance()->Init(prediction_conf_);
    // predictor manager
    PredictorManager::Instance()->Init(prediction_conf_);
  }
  bool Forward(std::vector<RecordMessage> msgs, Perturbation delta,
               std::string mode, std::string label);
  common::adapter::AdapterManagerConfig adapter_conf_;
  PredictionConf prediction_conf_;
  // SemanticLSTMEvaluator semantic_lstm_evaluator;
};

bool EvaluatePrediction::Forward(std::vector<RecordMessage> msgs,
                                 Perturbation delta, std::string mode,
                                 std::string label="") {
  // get containers
  auto obstacles_container =
      ContainerManager::Instance()->GetContainer<ObstaclesContainer>(
          AdapterConfig::PERCEPTION_OBSTACLES);
  obstacles_container->Clear();
  auto ego_pose_container =
      ContainerManager::Instance()->GetContainer<PoseContainer>(
          AdapterConfig::LOCALIZATION);
  ego_pose_container->Clear();
  // extrapolation predictor does not use adc_trajectory container;
  // just a placeholder
  auto ego_trajectory_container =
      ContainerManager::Instance()->GetContainer<ADCTrajectoryContainer>(
          AdapterConfig::PLANNING_TRAJECTORY);

  // SemanticMap::Instance()->Init();
  apollo::common::Point3D last_pos;
  double original_theta = 0;
  Trajectory traj;
  apollo::localization::LocalizationEstimate pose;
  apollo::perception::PerceptionObstacles perception_obstacles;

  unsigned int msg_cnt_limit = delta.D.size();
  unsigned int msg_cnt = 0;
  for (RecordMessage msg : msgs) {
    if (msg.channel_name == "/apollo/localization/pose") {
      AWARN << "pose msg";
      if (!pose.ParseFromString(msg.content)) continue;
      ego_pose_container->Insert(pose);
    }
    if (msg.channel_name == "/apollo/perception/obstacles") {
      AWARN << "obstacle msg";
      obstacles_container->CleanUp();

      if (!perception_obstacles.ParseFromString(msg.content)) continue;

      const apollo::perception::PerceptionObstacle* ptr_ego_vehicle =
          ego_pose_container->ToPerceptionObstacle();
      if (ptr_ego_vehicle != nullptr) {
        double perception_obs_timestamp = ptr_ego_vehicle->timestamp();
        if (perception_obstacles.has_header() &&
            perception_obstacles.header().has_timestamp_sec()) {
          perception_obs_timestamp =
              perception_obstacles.header().timestamp_sec();
        }
        obstacles_container->InsertPerceptionObstacle(*ptr_ego_vehicle,
                                                      perception_obs_timestamp);
      }

      auto perception_obstacle_ptr =
          perception_obstacles.mutable_perception_obstacle()->Mutable(0);

      if (msg_cnt == 0) {
        original_theta = perception_obstacle_ptr->theta();
      }

      if (msg_cnt >= 1) {
        if (delta.D[msg_cnt][0] == 1) {
          double d = delta.D[msg_cnt][1];
          double t = delta.D[msg_cnt][2];
          double x_offset = cos(original_theta + t / 360 * 2 * PI) * d;
          double y_offset = sin(original_theta + t / 360 * 2 * PI) * d;

          auto pos_ptr = perception_obstacle_ptr->mutable_position();
          pos_ptr->set_x(pos_ptr->x() + x_offset);
          pos_ptr->set_y(pos_ptr->y() + y_offset);
          auto v_ptr = perception_obstacle_ptr->mutable_velocity();
          v_ptr->set_x((pos_ptr->x() - last_pos.x()) / 0.1);
          v_ptr->set_y((pos_ptr->y() - last_pos.y()) / 0.1);
          perception_obstacle_ptr->set_theta(atan(v_ptr->y() / v_ptr->x()));
        } else if (delta.D[msg_cnt][0] == 2) {
          double x_offset = delta.D[msg_cnt][1];
          double y_offset = delta.D[msg_cnt][2];

          auto pos_ptr = perception_obstacle_ptr->mutable_position();
          pos_ptr->set_x(pos_ptr->x() + x_offset);
          pos_ptr->set_y(pos_ptr->y() + y_offset);
          auto v_ptr = perception_obstacle_ptr->mutable_velocity();
          v_ptr->set_x((pos_ptr->x() - last_pos.x()) / 0.1);
          v_ptr->set_y((pos_ptr->y() - last_pos.y()) / 0.1);
          perception_obstacle_ptr->set_theta(atan(v_ptr->y() / v_ptr->x()));
        }
      }

      last_pos = perception_obstacle_ptr->position();
      obstacles_container->Insert(perception_obstacles);

      // record trajectory
      auto trajectory_point = traj.add_trajectory_point();
      auto path_point = trajectory_point->mutable_path_point();
      path_point->set_x(last_pos.x());
      path_point->set_y(last_pos.y());

      // count up
      msg_cnt++;
    }

    if (msg_cnt >= msg_cnt_limit) break;
  }

  std::cout << "Loaded " << std::to_string(msg_cnt_limit) << " messages\n";

  obstacles_container->BuildLaneGraph();

  // write trajectory
  cyber::common::SetProtoToASCIIFile(
      traj, "/apollo/eval_data/trajectories/history" + label + ".pb.txt");

  // do trajectory prediction
  std::cout << "Start prediction\n";
  EvaluatorManager::Instance()->Run(obstacles_container);
  Obstacle* obstacle_ptr = obstacles_container->GetObstacle(12345);
  // semantic_lstm_evaluator.Evaluate(obstacle_ptr, obstacles_container);
  std::cout << "Finish prediction\n";

  // fetch prediction results
  const Feature& feature = obstacle_ptr->latest_feature();
  const LaneGraph& lane_graph = feature.lane().lane_graph();

  if (mode == "mlp") {
    unsigned int lane_seq_id = 0;
    for (const auto& lane_sequence : lane_graph.lane_sequence()) {
      AWARN << lane_seq_id++ << " " << lane_sequence.probability();
    }
    while (lane_seq_id < 3) {
      lane_seq_id++;
    }
  } else if (mode == "lstm") {
    unsigned int traj_id = 0;
    for (const auto& trajectory : feature.predicted_trajectory()) {
      AWARN << traj_id++ << " " << trajectory.probability();
      cyber::common::SetProtoToASCIIFile(
          trajectory, "/apollo/eval_data/trajectories/evaluate" + label + ".pb.txt");
      break;
    }
  }

  // Make predictions
  PredictorManager::Instance()->Run(
      perception_obstacles, ego_trajectory_container, obstacles_container);
  // Get predicted obstacles
  // *prediction_obstacles =
  // PredictorManager::Instance()->prediction_obstacles();
  if (mode == "lstm") {
    unsigned int traj_id = 0;
    for (const auto& trajectory : feature.predicted_trajectory()) {
      AWARN << traj_id++ << " " << trajectory.probability();
      cyber::common::SetProtoToASCIIFile(
          trajectory, "/apollo/eval_data/trajectories/predict" + label + ".pb.txt");
      break;
    }
  }

  std::cout << "Result written\n";

  return true;
}

}  // namespace prediction
}  // namespace apollo

int main(int argc, char** argv) {
  auto PT = apollo::prediction::EvaluatePrediction();

  // load delta
  auto delta = apollo::prediction::Perturbation();
  // delta.load(default_perturbation_file);

  // set map
  FLAGS_map_dir = default_map_dir;
  FLAGS_base_map_filename = default_base_map_filename;

  // set preception results
  std::vector<RecordMessage> msgs;
  auto reader = std::make_shared<RecordReader>(default_record_file);
  RecordViewer viewer(reader);
  for (auto& msg : viewer) {
    if (msg.channel_name == "/apollo/localization/pose") {
      msgs.push_back(msg);
    }
    if (msg.channel_name == "/apollo/perception/obstacles") {
      msgs.push_back(msg);
    }
  }

  // do something
  // PT.Forward(msgs, delta, "lstm", "0");
  // for (int i = 1; i <= 10; i++) {
  //   std::vector<double> tmp = {1, 0, 0};
  //   delta.D.push_back(tmp);
  //   PT.Forward(msgs, delta, "lstm", std::to_string(i));
  // }
  delta.D.clear();
  std::ifstream f(default_perturbation_file);
  if (f.is_open()) {
    std::string line;
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::vector<double> p;
      std::string token;
      while (std::getline(ss, token, ',')) {
        p.push_back(std::stod(token));
      }
      delta.D.push_back(p);
      if (delta.D.size() >= 20) {
        PT.Forward(msgs, delta, "lstm", std::to_string(delta.D.size()));
      }
    }
  }

  return 1;
}
