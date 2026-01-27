#ifndef BASE_HIT_NODE_HPP_
#define BASE_HIT_NODE_HPP_

#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include "detector.hpp"
#include "hikcamera.hpp"
#include "performance_monitor.hpp"
#include "autoaim_msgs/msg/basehit.hpp"

namespace auto_base
{

class BaseHitNode
{
public:
  explicit BaseHitNode(const std::string & config_path);
  ~BaseHitNode();

  int run();
  void request_stop();

private:
  void visualize(const cv::Mat & img, const std::vector<Detector::GreenLight> & detections);

  std::string config_path_;

  std::unique_ptr<camera::HikCamera> camera_;
  std::unique_ptr<Detector> detector_;
  rclcpp::Node::SharedPtr ros_node_;
  rclcpp::Publisher<autoaim_msgs::msg::Basehit>::SharedPtr hit_pub_;

  utils::PerformanceMonitor perf_monitor_;

  bool enable_visualization_{true};
  std::string window_name_{"base_hit_pipeline"};

  std::atomic<bool> quit_{false};
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace base_hit

#endif  // BASE_HIT_NODE_HPP_
