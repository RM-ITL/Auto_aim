// USB 相机装甲板识别测试节点
//
// 目的：用普通 USB 相机 (cv::VideoCapture) 快速验证装甲板识别效果，
//       识别链路与哨兵 (sentry_node) 完全一致：
//         BGR 帧 -> cvtColor(BGR2RGB) -> Detector::detect -> Armor 列表
//       Detector 内部按检测器配置里的 yolo.yolo_name 选择 yolov5 / yolo11，
//       并做传统法二次角点矫正 + tiny_resnet 数字分类，与整车一致。
//
// 用法：
//   ros2 run test_pipeline usb_detect_node [usb-camera-yaml] [--no-show]
//   例：ros2 run test_pipeline usb_detect_node src/config/usb_camera.yaml
//
//   相机参数(设备号/分辨率/曝光/增益等)在 usb_camera.yaml 里配置。
//   曝光/增益等 V4L2 控制项通过 v4l2-ctl 应用(需安装 v4l-utils)。
//   --no-show    不弹窗，仅终端打印每帧检测数量与信息
//
// 说明：本节点只测识别，不涉及 solver / tracker / planner / 串口。

#include <atomic>
#include <cstdlib>
#include <csignal>
#include <exception>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "app_config/app_config.hpp"
#include "detect_node.hpp"
#include "draw_tools.hpp"
#include "logger.hpp"

namespace
{
std::atomic<bool> g_stop_requested{false};

void handle_signal(int) { g_stop_requested.store(true); }

// 若 yaml 中存在 key，则用 v4l2-ctl 设置对应的 V4L2 控制项
void apply_v4l2_ctrl(
  int device, const YAML::Node & cam, const std::string & yaml_key,
  const std::string & ctrl_name)
{
  if (!cam[yaml_key]) {
    return;
  }
  const int value = cam[yaml_key].as<int>();
  std::ostringstream cmd;
  cmd << "v4l2-ctl -d /dev/video" << device << " -c " << ctrl_name << "=" << value;
  const int ret = std::system(cmd.str().c_str());
  if (ret != 0) {
    utils::logger()->warn("[USBDetect] 设置 {} 失败 (v4l2-ctl 未安装?): {}", ctrl_name, cmd.str());
  } else {
    utils::logger()->info("[USBDetect] {} = {}", ctrl_name, value);
  }
}

int fourcc_from_string(const std::string & s)
{
  if (s.size() == 4) {
    return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
  }
  return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string cam_yaml_path =
    std::filesystem::current_path().string() + "/src/config/usb_camera.yaml";
  bool show = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--no-show") {
      show = false;
    } else if (!arg.empty() && arg[0] != '-') {
      cam_yaml_path = arg;
    }
  }

  std::signal(SIGINT, handle_signal);

  try {
    // 1. 读取 USB 相机配置
    const YAML::Node root = YAML::LoadFile(cam_yaml_path);
    const YAML::Node cam = root["usb_camera"];
    if (!cam) {
      utils::logger()->error("[USBDetect] 配置缺少 usb_camera 段: {}", cam_yaml_path);
      return 1;
    }
    const int device = cam["device"] ? cam["device"].as<int>() : 2;

    // 2. 加载检测器配置（复用哨兵完全相同的模型与检测参数）
    const std::string detector_cfg = root["detector_config"]
      ? root["detector_config"].as<std::string>()
      : (std::filesystem::current_path().string() + "/src/config/sentry.yaml");
    const auto app_config = app_config::AppConfig::load(detector_cfg);
    armor_auto_aim::Detector detector(app_config.detector);
    utils::logger()->info("[USBDetect] Detector 初始化完成, 检测器配置: {}", detector_cfg);

    // 3. 先用 v4l2-ctl 应用曝光/增益等控制项
    //    注意顺序: auto_* 开关要先于其控制的目标项
    apply_v4l2_ctrl(device, cam, "auto_exposure", "auto_exposure");
    apply_v4l2_ctrl(device, cam, "exposure", "exposure_time_absolute");
    apply_v4l2_ctrl(device, cam, "gain", "gain");
    apply_v4l2_ctrl(device, cam, "gamma", "gamma");
    apply_v4l2_ctrl(device, cam, "brightness", "brightness");
    apply_v4l2_ctrl(device, cam, "auto_white_balance", "white_balance_automatic");
    apply_v4l2_ctrl(device, cam, "white_balance_temperature", "white_balance_temperature");

    // 4. 打开 USB 相机并设置采集格式
    cv::VideoCapture capture(device, cv::CAP_V4L2);
    if (!capture.isOpened()) {
      capture.open(device);  // 回退到默认后端
    }
    if (!capture.isOpened()) {
      utils::logger()->error("[USBDetect] 打开 USB 相机失败: /dev/video{}", device);
      return 1;
    }

    if (cam["fourcc"]) {
      capture.set(cv::CAP_PROP_FOURCC, fourcc_from_string(cam["fourcc"].as<std::string>()));
    }
    if (cam["width"]) capture.set(cv::CAP_PROP_FRAME_WIDTH, cam["width"].as<double>());
    if (cam["height"]) capture.set(cv::CAP_PROP_FRAME_HEIGHT, cam["height"].as<double>());
    if (cam["fps"]) capture.set(cv::CAP_PROP_FPS, cam["fps"].as<double>());

    utils::logger()->info(
      "[USBDetect] USB 相机已打开: /dev/video{}, 实际分辨率={}x{}, FPS={}",
      device, capture.get(cv::CAP_PROP_FRAME_WIDTH),
      capture.get(cv::CAP_PROP_FRAME_HEIGHT), capture.get(cv::CAP_PROP_FPS));

    const std::string window_name = "usb_detect";
    if (show) {
      cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    }

    // 5. 主循环：读帧 -> 识别 -> 绘制
    int frame_index = 0;
    while (!g_stop_requested.load()) {
      cv::Mat bgr;
      if (!capture.read(bgr) || bgr.empty()) {
        utils::logger()->warn("[USBDetect] 读取到空帧，退出");
        break;
      }
      ++frame_index;

      // 与哨兵一致：detect 输入为 RGB
      cv::Mat rgb;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
      const auto armors = detector.detect(rgb);

      // 终端信息
      std::ostringstream oss;
      oss << "[USBDetect] frame " << frame_index << ": " << armors.size() << " armor(s)";
      for (const auto & a : armors) {
        oss << " | " << armor_auto_aim::armor_name_to_string(a.name) << " "
            << (a.color == armor_auto_aim::Color::red ? "red" : "blue") << " "
            << (a.type == armor_auto_aim::ArmorType::small ? "small" : "big")
            << " conf=" << a.confidence;
      }
      if (!armors.empty() || frame_index % 30 == 0) {
        utils::logger()->info(oss.str());
      }

      // 可视化：在 BGR 原图上画框
      if (show) {
        cv::Mat canvas = bgr.clone();
        for (const auto & a : armors) {
          if (a.points.size() == 4) {
            utils::draw_quadrangle_with_corners(
              canvas, a.points, utils::colors::GREEN, 2, 3);
          }
          std::ostringstream label;
          label << armor_auto_aim::armor_name_to_string(a.name) << " "
                << (a.type == armor_auto_aim::ArmorType::small ? "small" : "big")
                << " " << static_cast<int>(a.confidence * 100) << "%";
          utils::draw_detection_label(
            canvas, label.str(), a.center, utils::colors::GREEN, 0.6, 2);
        }
        utils::draw_frame_number(canvas, frame_index);

        cv::imshow(window_name, canvas);
        const int key = cv::waitKey(1);  // ESC 或 q 退出
        if (key == 27 || key == 'q') {
          break;
        }
      }
    }

    if (show) {
      cv::destroyWindow(window_name);
    }
    capture.release();
    utils::logger()->info("[USBDetect] 已退出");
    return 0;
  } catch (const std::exception & e) {
    utils::logger()->error("[USBDetect] 程序异常终止: {}", e.what());
  }
  return 1;
}
