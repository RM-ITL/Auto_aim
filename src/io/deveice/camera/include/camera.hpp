#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <chrono>
#include <memory>
#include <string>
#include <variant>

#include <opencv2/opencv.hpp>

#include "app_config/app_config.hpp"
#include "hikcamera.hpp"
#ifdef HAS_MINDVISION
#include "mindvision.hpp"
#endif

namespace camera
{

class Camera
{
public:
  explicit Camera(const app_config::CameraConfig & config);
  ~Camera() = default;

  Camera(const Camera &) = delete;
  Camera & operator=(const Camera &) = delete;

  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

  /// 停止相机，唤醒阻塞在 read() 上的线程
  void stop();

  const std::string & camera_type() const { return camera_type_; }

private:
  std::string camera_type_;

#ifdef HAS_MINDVISION
  std::variant<std::unique_ptr<HikCamera>, std::unique_ptr<MindVision>> camera_;
#else
  std::variant<std::unique_ptr<HikCamera>> camera_;
#endif
};

}  // namespace camera

#endif  // CAMERA_HPP
