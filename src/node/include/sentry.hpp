#ifndef SENTRY_NODE_HPP_
#define SENTRY_NODE_HPP_

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
#include "detect_node.hpp"
#include "solver_node.hpp"
#include "thread_safe_queue.hpp"
#include "tracker.hpp"
#include "armor.hpp"
#include "planner.hpp"
#include "autoaim_msgs/msg/debug.hpp"
#include "autoaim_msgs/msg/gimbal_cmd.hpp"
#include "autoaim_msgs/msg/gimbal_state.hpp"
#include "autoaim_msgs/msg/orienta.hpp"
#include "autoaim_msgs/msg/outpost.hpp"
#include "lower_sentry.hpp"
#include "shooter.hpp"

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

enum class SentryRunMode
{
  Direct,
  Bridge,
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
  void gimbal_state_callback(const autoaim_msgs::msg::GimbalState::SharedPtr msg);
  bool has_runtime_state() const;
  io::GimbalState runtime_state() const;
  Eigen::Quaterniond runtime_orientation(std::chrono::steady_clock::time_point timestamp) const;
  void output_control(const plan::Plan & plan_result, const io::GimbalState & gs);

  // 组件与配置
  std::string config_path_;
  std::unique_ptr<camera::Camera> camera_;
  std::unique_ptr<armor_auto_aim::Detector> detector_;
  std::unique_ptr<solver::Solver> solver_;
  solver::YawOptimizer* yaw_optimizer_;
  std::unique_ptr<tracker::Tracker> tracker_;
  std::unique_ptr<plan::Planner> planner_;
  std::unique_ptr<io::Sentry> sentry_;
  std::unique_ptr<shooter::Shooter> shooter_;
  rclcpp::Node::SharedPtr ros_node_;

  // Debug所用的Topic
  rclcpp::Publisher<autoaim_msgs::msg::Debug>::SharedPtr debug_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::Orienta>::SharedPtr orientation_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::Outpost>::SharedPtr target_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::GimbalCmd>::SharedPtr gimbal_cmd_pub_;
  rclcpp::Subscription<autoaim_msgs::msg::GimbalState>::SharedPtr gimbal_state_sub_;

  tools::ThreadSafeQueue<DebugPacket, true> visualization_queue{2};
  tools::ThreadSafeQueue<std::optional<tracker::TargetVariant>, true> target_queue{1};
  // tools::ThreadSafeQueue<TargetPacket, true> target_queue{1};

  std::atomic<bool> quit_{false};
  std::thread visualization_thread_;
  std::thread planner_thread_;
  std::thread spin_thread_;

  bool enable_visualization_{true};
  std::string visualization_window_name_{"armor_detection"};
  cv::Point visualization_center_point_{640, 384};
  std::atomic<int> visualization_frame_counter_{0};

  std::chrono::steady_clock::time_point start_time_;
  const double bullet_speed_{20.0};

  // 系统延迟统计
  std::deque<double> delay_window_;
  const size_t delay_window_size_{100};
  std::chrono::steady_clock::time_point last_delay_log_time_;
  std::chrono::steady_clock::time_point last_state_wait_log_time_;

  // fire占比统计（滑动时间窗口）
  std::deque<std::pair<std::chrono::steady_clock::time_point, bool>> fire_window_;
  double fire_window_sec_{10.0};

  SentryRunMode run_mode_{SentryRunMode::Direct};
  std::string run_mode_name_{"direct"};

  mutable std::mutex bridge_state_mutex_;
  bool bridge_has_state_{false};
  io::GimbalState bridge_state_{};
  Eigen::Quaterniond bridge_orientation_{Eigen::Quaterniond::Identity()};
};

}  // namespace Application

#endif  // SENTRY_NODE_HPP_
