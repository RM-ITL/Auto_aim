#include "camera.hpp"

#include <stdexcept>

#include "logger.hpp"

namespace camera
{

Camera::Camera(const app_config::CameraConfig & config)
{
  camera_type_ = config.type;
  utils::logger()->info("[Camera] camera.type          = {}", camera_type_);

  if (camera_type_ == "hik") {
    camera_ = std::make_unique<HikCamera>(config.hik);
    utils::logger()->info("Created HikCamera");
#ifdef HAS_MINDVISION
  } else if (camera_type_ == "mindvision") {
    // MindVision ctor 仍接 (exposure_ms, gamma, vid_pid) 三参数（结构化签名，1.C 不动）
    utils::logger()->info("[Camera] mindvision.exposure_ms = {:.3f} ms", config.mindvision.exposure_ms);
    utils::logger()->info("[Camera] mindvision.gamma       = {:.3f}", config.mindvision.gamma);
    utils::logger()->info("[Camera] mindvision.vid_pid     = {}", config.mindvision.vid_pid);

    camera_ = std::make_unique<MindVision>(
      config.mindvision.exposure_ms, config.mindvision.gamma, config.mindvision.vid_pid);
    utils::logger()->info("Created MindVision camera");
#endif
  } else {
    utils::logger()->error("Unknown camera type: {}", camera_type_);
    throw std::runtime_error("Unknown camera type: " + camera_type_);
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  std::visit(
    [&img, &timestamp](auto & cam) {
      if (cam) {
        cam->read(img, timestamp);
      }
    },
    camera_);
}

void Camera::stop()
{
  std::visit(
    [](auto & cam) {
      if (cam) {
        cam->stop();
      }
    },
    camera_);
}

}  // namespace camera
