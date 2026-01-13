#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <chrono>
#include <memory>
#include <string>
#include <variant>

#include <opencv2/opencv.hpp>

#include "hikcamera.hpp"
#include "mindvision.hpp"

namespace camera
{

class Camera
{
public:
  explicit Camera(const std::string & config_path);
  ~Camera() = default;

  Camera(const Camera &) = delete;
  Camera & operator=(const Camera &) = delete;

  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

  const std::string & camera_type() const { return camera_type_; }

private:
  bool load_config(const std::string & config_path);

  std::string config_path_;
  std::string camera_type_;

  std::variant<std::unique_ptr<HikCamera>, std::unique_ptr<MindVision>> camera_;
};

}  // namespace camera

#endif  // CAMERA_HPP
