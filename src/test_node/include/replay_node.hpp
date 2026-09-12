#ifndef REPLAY_NODE_HPP_
#define REPLAY_NODE_HPP_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

#include "app_config/app_config.hpp"
#include "detect_node.hpp"
#include "solver_node.hpp"

#include "autoaim_msgs/msg/orienta.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace Application
{

// 回放节点：订阅 `ros2 bag play` 重播出来的 /image_raw + /imu/quaternion，
// 喂进 Detector -> Solver（仅检测+解算，不含 tracker/planner），
// 弹可视化窗口显示检测框与解算后重投影的装甲板四边形。
//
// 数据来源与 record_node 对应：
//   - /image_raw       sensor_msgs/msg/Image      (bgr8, header.stamp = 采集时刻 steady 纳秒)
//   - /imu/quaternion  autoaim_msgs/msg/Orienta   (w/x/y/z 绝对姿态，无 header/stamp)
//
// 配对约定：record 端每帧「先发 image_N 再发 imu_N」严格 1:1。故本节点在 image 回调
// 缓存最新帧，在 imu 回调取缓存帧一起处理，使 image_N 与 orienta_N 对齐。
class ReplayApp
{
public:
  explicit ReplayApp(const app_config::AppConfig & app_config);
  ~ReplayApp();

  int run();
  void request_stop();

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void imu_callback(const autoaim_msgs::msg::Orienta::ConstSharedPtr & msg);

  // 组件
  std::unique_ptr<armor_auto_aim::Detector> detector_;
  std::unique_ptr<solver::Solver> solver_;
  solver::YawOptimizer * yaw_optimizer_{nullptr};

  rclcpp::Node::SharedPtr ros_node_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<autoaim_msgs::msg::Orienta>::SharedPtr imu_sub_;
  rclcpp::Publisher<autoaim_msgs::msg::Orienta>::SharedPtr orientation_pub_;

  std::string image_topic_{"/image_raw"};
  std::string imu_topic_{"/imu/quaternion"};

  // 缓存最新一帧（image 回调写，imu 回调读；单线程 spin，回调串行，无需锁）。
  cv::Mat last_bgr_;
  double last_ts_sec_{0.0};
  bool has_image_{false};

  // 用首帧 stamp 作零点，得到相对、单调递增的秒时间戳喂给 solver。
  std::int64_t first_stamp_ns_{0};
  bool first_stamp_set_{false};

  // 可视化
  std::string visualization_window_name_{"armor_detection"};
  cv::Point visualization_center_point_{640, 384};
  int frame_index_{0};

  // 统计（每秒一次）
  std::chrono::steady_clock::time_point last_log_time_;
  std::uint64_t frame_count_window_{0};
  std::uint64_t armor_count_window_{0};
  std::uint64_t pnp_success_window_{0};

  std::atomic<bool> quit_{false};
};

}  // namespace Application

#endif  // REPLAY_NODE_HPP_
