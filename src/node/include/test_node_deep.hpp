#ifndef TEST_NODE_HPP_
#define TEST_NODE_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <rclcpp/rclcpp.hpp>

#include "camera.hpp"
#include "imu_driver.h"
#include "detect_node.hpp"
#include "solver_node.hpp"
#include "thread_safe_queue.hpp"
#include "tracker.hpp"
#include "armor.hpp"
#include "planner.hpp"
#include "guard_planner.hpp"
#include "autoaim_msgs/msg/debug.hpp"
#include "autoaim_msgs/msg/orienta.hpp"
#include "gimbal.hpp"

namespace Application
{

using Armors = armor_auto_aim::Armor;
using Visualization = armor_auto_aim::Visualization;

struct DebugPacket
{
  cv::Mat rgb_image;
  std::vector<Visualization> reprojected_armors;
  std::string tracker_state;
  bool valid{false};
};

class PipelineApp
{
public:
  explicit PipelineApp(const std::string & config_path);
  ~PipelineApp();

  int run();
  void request_stop();

private:
  void start_threads();
  void join_threads();
  void visualization_loop();
  void planner_loop();


  // 组件与配置
  std::string config_path_;
  std::unique_ptr<camera::Camera> camera_;
  std::unique_ptr<io::DmImu> dm_imu_;
  std::unique_ptr<armor_auto_aim::Detector> detector_;
  std::unique_ptr<solver::Solver> solver_;
  solver::YawOptimizer* yaw_optimizer_;
  std::unique_ptr<tracker::Tracker> tracker_;
  std::unique_ptr<plan::Planner> planner_;
  std::unique_ptr<guard::GuardPlanner> guard_planner_;
  std::unique_ptr<io::Gimbal> gimbal_;
  rclcpp::Node::SharedPtr ros_node_;
  rclcpp::Publisher<autoaim_msgs::msg::Debug>::SharedPtr debug_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::Orienta>::SharedPtr orientation_pub_;

  tools::ThreadSafeQueue<DebugPacket, true> visualization_queue{2};
  tools::ThreadSafeQueue<std::optional<tracker::TargetVariant>, true> target_queue{1};

  std::atomic<bool> quit_{false};
  std::thread visualization_thread_;
  std::thread planner_thread_;

  bool enable_visualization_{true};
  std::string visualization_window_name_{"armor_detection"};
  cv::Point visualization_center_point_{640, 384};
  std::atomic<int> visualization_frame_counter_{0};

  std::chrono::steady_clock::time_point start_time_;
  const double bullet_speed_{22.0};

  // // 系统延迟统计
  // std::deque<double> delay_window_;
  // const size_t delay_window_size_{100};
  // std::chrono::steady_clock::time_point last_delay_log_time_;
};

}  // namespace pipeline

#endif  // TEST_NODE_HPP_
