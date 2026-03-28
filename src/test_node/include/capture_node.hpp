#ifndef CAPTURE_NODE_HPP_
#define CAPTURE_NODE_HPP_

#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include "camera.hpp"
#include "gimbal.hpp"
#include "imu_driver.h"

namespace Application
{

class CaptureApp
{
public:
  explicit CaptureApp(const std::string & config_path, const std::string & output_folder);
  ~CaptureApp();

  int run();
  void request_stop();

private:
  void write_q(const std::string & q_path, const Eigen::Quaterniond & q);
  void write_timestamp(const std::string & timestamp_path, std::chrono::steady_clock::time_point timestamp);

  std::string config_path_;
  std::string output_folder_;

  std::unique_ptr<camera::Camera> camera_;
  std::unique_ptr<io::Gimbal> gimbal_;
  std::unique_ptr<io::DmImu> dm_imu_;
  rclcpp::Node::SharedPtr ros_node_;

  std::atomic<bool> quit_{false};

  // 对称圆点板尺寸 (列数, 行数)
  const cv::Size pattern_size_{11, 8};
  const double circle_spacing_mm_{20.0};
  const double circle_diameter_mm_{15.0};
  const cv::Size board_size_mm_{210, 297};
  const double preview_scale_{0.5};
  const std::string window_name_{"Press s to save, q to quit"};
};

}  // namespace Application

#endif  // CAPTURE_NODE_HPP_
