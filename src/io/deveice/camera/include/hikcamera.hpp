#ifndef HIKCAMERA_HPP
#define HIKCAMERA_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include "MvCameraControl.h"
#include "performance_monitor.hpp"
#include "thread_safe_queue.hpp"

namespace camera
{

struct CameraData
{
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
};

class HikCamera
{
public:
  explicit HikCamera(const std::string & config_path);
  ~HikCamera();

  HikCamera(const HikCamera &) = delete;
  HikCamera & operator=(const HikCamera &) = delete;

  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);  // 对外接口，主要是一个流式的拿取图像
  const std::string & image_topic() const { return image_topic_; }  // ROS发布调式用

private:
  void daemon_loop();
  void capture_loop();
  bool start_capture();
  void stop_capture();
  void set_camera_parameters();
  cv::Mat convert_bayer(const cv::Mat & raw, unsigned int type);
  void reset_usb() const;
  bool load_config(const std::string & config_path);

  std::string config_path_;

  void * camera_handle_ = nullptr;
  unsigned int payload_size_ = 0;
  std::unique_ptr<unsigned char[]> raw_buffer_;

  std::thread daemon_thread_;
  std::thread capture_thread_;

  std::atomic<bool> daemon_quit_{false};
  std::atomic<bool> capture_quit_{false};
  std::atomic<bool> capturing_{false};
  std::atomic<bool> shutdown_{false};

  tools::ThreadSafeQueue<CameraData> queue_;
  utils::PerformanceMonitor perf_monitor_;

  double exposure_us_ = 0.0;
  double gain_ = 0.0;
  double fps_ = 0.0;
  cv::Size target_size_;
  std::string image_topic_;
  int vid_ = 0x2bdf;
  int pid_ = 0x0299;
};

}  // namespace camera

#endif
