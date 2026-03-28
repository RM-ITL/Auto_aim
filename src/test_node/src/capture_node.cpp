#include "capture_node.hpp"

#include <csignal>
#include <filesystem>
#include <fstream>

#include <fmt/core.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hpp"
#include "math_tools.hpp"
#include "draw_tools.hpp"

namespace Application
{

namespace
{
std::atomic<bool> g_stop_requested{false};

void handle_signal(int)
{
  g_stop_requested.store(true);
}
}  // namespace

CaptureApp::CaptureApp(const std::string & config_path, const std::string & output_folder)
: config_path_(config_path), output_folder_(output_folder)
{
  ros_node_ = std::make_shared<rclcpp::Node>("capture_node");

  camera_ = std::make_unique<camera::Camera>(config_path_);
  gimbal_ = std::make_unique<io::Gimbal>(config_path_);
  dm_imu_ = std::make_unique<io::DmImu>(config_path_);
  // 创建输出文件夹
  std::filesystem::create_directories(output_folder_);

  utils::logger()->info("[Capture] 模块初始化完成");
  utils::logger()->info("[Capture] 圆点板尺寸: {}x{}", pattern_size_.width, pattern_size_.height);
  utils::logger()->info(
    "[Capture] 圆点直径: {:.1f} mm, 圆心距: {:.1f} mm, 板尺寸: {}x{} mm",
    circle_diameter_mm_, circle_spacing_mm_, board_size_mm_.width, board_size_mm_.height);
  utils::logger()->info("[Capture] 输出文件夹: {}", output_folder_);
}

CaptureApp::~CaptureApp()
{
  request_stop();
}

void CaptureApp::write_q(const std::string & q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // 输出顺序为 wxyz
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

void CaptureApp::write_timestamp(
  const std::string & timestamp_path, std::chrono::steady_clock::time_point timestamp)
{
  std::ofstream timestamp_file(timestamp_path);
  auto timestamp_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count();
  timestamp_file << timestamp_ns;
  timestamp_file.close();
}

int CaptureApp::run()
{
  cv::namedWindow(window_name_, cv::WINDOW_NORMAL);

  int count = 0;

  while (!quit_.load()) {
    if (g_stop_requested.load()) {
      break;
    }

    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;

    camera_->read(img, timestamp);
    // Eigen::Quaterniond q = gimbal_->q(timestamp);
    Eigen::Quaterniond q = dm_imu_->imu_at(timestamp);

    cv::Mat img_bgr;
    if (camera_->camera_type() == "mindvision") {
      cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
    } else {
      img_bgr = img;
    }

    // 预览图缩放后再检测，保证实时显示流畅
    cv::Mat preview_img;
    cv::resize(img_bgr, preview_img, {}, preview_scale_, preview_scale_);
    auto img_with_ypr = preview_img.clone();
    Eigen::Vector3d zyx = utils::eulers(q, 2, 1, 0) * 57.3;  // 转换为角度
    utils::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {20, 20}, {0, 0, 255});
    utils::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {20, 50}, {0, 0, 255});
    utils::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {20, 80}, {0, 0, 255});

    std::vector<cv::Point2f> preview_centers;
    auto preview_success = cv::findCirclesGrid(
      preview_img, pattern_size_, preview_centers, 
      cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING);

    cv::drawChessboardCorners(img_with_ypr, pattern_size_, preview_centers, preview_success);

    cv::imshow(window_name_, img_with_ypr);
    auto key = cv::waitKey(1);

    if (key == 'q') {
      break;
    } else if (key != 's') {
      continue;
    }

    // 保存前做一次全分辨率圆点板检测
    std::vector<cv::Point2f> centers;
    auto success = cv::findCirclesGrid(
      img_bgr, pattern_size_, centers,
      cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING);

    if (!success) {
      utils::logger()->warn("[Capture] 当前帧未检测到圆点板，未保存");
      continue;
    }

    // 保存图片、四元数和时间戳
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder_, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder_, count);
    auto timestamp_path = fmt::format("{}/{}_timestamp.txt", output_folder_, count);
    cv::imwrite(img_path, img_bgr);
    write_q(q_path, q);
    write_timestamp(timestamp_path, timestamp);
    utils::logger()->info("[Capture] [{}] 已保存至 {}", count, output_folder_);
  }

  cv::destroyWindow(window_name_);
  utils::logger()->warn("[Capture] 注意四元数输出顺序为 wxyz");

  return 0;
}

void CaptureApp::request_stop()
{
  quit_.store(true);
}

}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? |                              | 输出命令行参数说明}"
    "{@config-path   |                              | YAML配置文件路径  }"
    "{output-folder o| assets/img_with_q            | 输出文件夹路径    }";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  std::string config_path = std::filesystem::current_path().string() + "/src/config/config.yaml";
  if (cli.has("@config-path")) {
    config_path = cli.get<std::string>("@config-path");
  }
  std::string output_folder = cli.get<std::string>("output-folder");

  rclcpp::init(argc, argv);
  std::signal(SIGINT, Application::handle_signal);

  try {
    Application::CaptureApp app(config_path, output_folder);
    int ret = app.run();
    rclcpp::shutdown();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[Capture] 程序异常终止: {}", e.what());
  }

  rclcpp::shutdown();
  return 1;
}
