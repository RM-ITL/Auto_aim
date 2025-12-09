#ifndef CAPTURE_NODE_HPP_
#define CAPTURE_NODE_HPP_

#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include "hikcamera.hpp"
#include "gimbal.hpp"

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

  std::string config_path_;
  std::string output_folder_;

  std::unique_ptr<camera::HikCamera> camera_;
  std::unique_ptr<io::Gimbal> gimbal_;
  rclcpp::Node::SharedPtr ros_node_;

  std::atomic<bool> quit_{false};

  // 棋盘格尺寸 (内角点数量)
  const cv::Size chessboard_size_{9, 6};
  const std::string window_name_{"Press s to save, q to quit"};
};

}  // namespace Application

#endif  // CAPTURE_NODE_HPP_
