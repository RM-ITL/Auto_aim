#include "camera.hpp"

#include <yaml-cpp/yaml.h>

#include "logger.hpp"

namespace camera
{

Camera::Camera(const std::string & config_path) : config_path_(config_path)
{
  if (!load_config(config_path)) {
    throw std::runtime_error("Failed to load camera config from: " + config_path);
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

bool Camera::load_config(const std::string & config_path)
{
  try {
    YAML::Node config = YAML::LoadFile(config_path);

    auto camera_node = config["camera"];
    if (!camera_node) {
      utils::logger()->error("Missing 'camera' section in config");
      return false;
    }

    // 读取相机类型，默认为hik
    camera_type_ = camera_node["type"].as<std::string>("hik");

    if (camera_type_ == "hik") {
      // 创建海康相机
      camera_ = std::make_unique<HikCamera>(config_path);
      utils::logger()->info("Created HikCamera");
    } else if (camera_type_ == "mindvision") {
      // 读取MindVision所需参数
      auto params = camera_node["parameters"];
      double exposure_ms = params["exposure_ms"].as<double>(5.0);
      double gamma = params["gamma"].as<double>(100.0);
      std::string vid_pid = params["vid_pid"].as<std::string>("2bdf:0283");

      camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
      utils::logger()->info("Created MindVision camera");
    } else {
      utils::logger()->error("Unknown camera type: {}", camera_type_);
      return false;
    }

    return true;
  } catch (const YAML::Exception & e) {
    utils::logger()->error("YAML parse error: {}", e.what());
    return false;
  } catch (const std::exception & e) {
    utils::logger()->error("Failed to create camera: {}", e.what());
    return false;
  }
}

}  // namespace camera
