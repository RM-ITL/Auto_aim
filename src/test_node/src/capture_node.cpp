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

  camera_ = std::make_unique<camera::HikCamera>(config_path_);
  gimbal_ = std::make_unique<io::Gimbal>(config_path_);

  // 创建输出文件夹
  std::filesystem::create_directories(output_folder_);

  utils::logger()->info("[Capture] 模块初始化完成");
  utils::logger()->info("[Capture] 棋盘格尺寸: {}x{}", chessboard_size_.width, chessboard_size_.height);
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
    Eigen::Quaterniond q = gimbal_->q(timestamp);

    // 在图像上显示欧拉角，用来判断 IMU 坐标系的 xyz 正方向，同时判断 IMU 是否存在零漂
    auto img_with_ypr = img.clone();
    Eigen::Vector3d zyx = utils::eulers(q, 2, 1, 0) * 57.3;  // 转换为角度
    utils::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    utils::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    utils::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});

    // 检测棋盘格角点
    std::vector<cv::Point2f> corners;
    auto success = cv::findChessboardCorners(
      img, chessboard_size_, corners,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

    // 如果找到角点，进行亚像素精确化
    if (success) {
      cv::Mat gray;
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
      cv::cornerSubPix(
        gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }

    // 绘制检测结果
    cv::drawChessboardCorners(img_with_ypr, chessboard_size_, corners, success);

    // 显示时缩小图片尺寸
    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);

    cv::imshow(window_name_, img_with_ypr);
    auto key = cv::waitKey(1);

    if (key == 'q') {
      break;
    } else if (key != 's') {
      continue;
    }

    // 保存图片和四元数
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder_, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder_, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q);
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

  std::string config_path = "/home/guo/ITL_Auto_aim/src/config/config.yaml";
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
