#ifndef RECORD_NODE_HPP_
#define RECORD_NODE_HPP_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

#include "app_config/app_config.hpp"
#include "camera.hpp"
#include "gimbal.hpp"
#include "imu_driver.h"

#include "autoaim_msgs/msg/orienta.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace Application
{

// 录制节点：把相机原始图像与同源时刻的 IMU 四元数发布到 ROS Topic，
// 供 `ros2 bag record` 落盘。图像与四元数共用同一个时间戳，保证回放侧可对齐。
//
// 发布话题：
//   - /image_raw       sensor_msgs/msg/Image        (bgr8, 1280x1024)
//   - /imu/quaternion  autoaim_msgs/msg/Orienta     (w/x/y/z; dm_* 恒为 0)
//
// 时间戳约定：header.stamp = 相机采集时刻的 steady_clock 值（纳秒）。
// 图像和四元数在同一帧内使用同一个 stamp，回放侧按 stamp 对齐即可。
class RecordApp
{
public:
  explicit RecordApp(const app_config::AppConfig & app_config);
  ~RecordApp();

  int run();
  void request_stop();

private:
  enum class ImuSource { Gimbal, DmImu };

  Eigen::Quaterniond pose_at(std::chrono::steady_clock::time_point timestamp);
  static rclcpp::Time to_ros_time(std::chrono::steady_clock::time_point timestamp);

  ImuSource imu_source_{ImuSource::Gimbal};
  std::string imu_source_name_;

  std::unique_ptr<camera::Camera> camera_;
  std::unique_ptr<io::Gimbal> gimbal_;
  std::unique_ptr<io::DmImu> dm_imu_;

  rclcpp::Node::SharedPtr ros_node_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::Orienta>::SharedPtr imu_pub_;

  std::string image_topic_{"/image_raw"};
  std::string imu_topic_{"/imu/quaternion"};
  std::string frame_id_{"camera_optical_frame"};

  std::atomic<bool> quit_{false};

  // 帧率统计
  std::chrono::steady_clock::time_point last_log_time_;
  std::uint64_t frame_count_{0};
  std::uint64_t frame_count_window_{0};
};

}  // namespace Application

#endif  // RECORD_NODE_HPP_
