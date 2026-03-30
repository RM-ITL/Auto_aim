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
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>

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
#include "autoaim_msgs/msg/outpost.hpp"
#include "autoaim_msgs/msg/sentry_cmd.hpp"
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
  std::unique_ptr<io::Sentry> sentry_;
  std::unique_ptr<shooter::Shooter> shooter_;
  rclcpp::Node::SharedPtr ros_node_;

  // Debug所用的Topic
  rclcpp::Publisher<autoaim_msgs::msg::Debug>::SharedPtr debug_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::Orienta>::SharedPtr orientation_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::Outpost>::SharedPtr target_pub_;

  // 与导航通讯使用的Topic
  rclcpp::Publisher<autoaim_msgs::msg::SentryCmd>::SharedPtr cmd_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr tf_complete_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr gimbal_active_sub_;

  // 导航速度 (来自vel_sub_订阅)
  std::atomic<float> nav_vx_{0.0f};
  std::atomic<float> nav_vy_{0.0f};
  std::atomic<float> nav_w_{0.0f};

  // 云台可控状态 (来自gimbal_active_sub_订阅)
  std::atomic<bool> gimbal_active_{false};

  // 扫描参数 (从yaml读取)
  float scan_yaw_center_{0.0f};
  float scan_yaw_amplitude_{1.5f};
  float scan_yaw_period_{6.0f};
  float scan_pitch_center_{-0.05f};
  float scan_pitch_amplitude_{0.15f};
  float scan_pitch_period_{3.0f};

  tools::ThreadSafeQueue<DebugPacket, true> visualization_queue{2};
  tools::ThreadSafeQueue<std::optional<tracker::TargetVariant>, true> target_queue{1};

  std::atomic<bool> quit_{false};
  std::thread visualization_thread_;
  std::thread planner_thread_;
  std::thread spin_thread_;

  bool enable_visualization_{true};
  std::string visualization_window_name_{"armor_detection"};
  cv::Point visualization_center_point_{640, 384};
  std::atomic<int> visualization_frame_counter_{0};

  std::chrono::steady_clock::time_point start_time_;
  const double bullet_speed_{22.0};

  // 系统延迟统计
  std::deque<double> delay_window_;
  const size_t delay_window_size_{100};
  std::chrono::steady_clock::time_point last_delay_log_time_;

  // fire占比统计（滑动时间窗口）
  std::deque<std::pair<std::chrono::steady_clock::time_point, bool>> fire_window_;
  double fire_window_sec_{10.0};
};

}  // namespace Application

#endif  // SENTRY_NODE_HPP_
