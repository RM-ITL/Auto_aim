#include "capture_node.hpp"

#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <fmt/core.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "app_config/app_config.hpp"
#include "calibration_common.hpp"
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

CaptureApp::CaptureApp(const app_config::AppConfig & app_config, const std::string & output_folder)
: output_folder_(output_folder)
{
  ros_node_ = std::make_shared<rclcpp::Node>("capture_node");
  pose_source_ = ros_node_->declare_parameter<std::string>("pose_source", "gimbal");
  diag_mode_ = ros_node_->declare_parameter<bool>("diag", false);

  // 把 SubConfig 的 vector<double>(9) 转 Eigen::Matrix3d；与原 read_r_gimbal_to_imu 行为字节级一致。
  auto unflatten_3x3 = [](const std::vector<double> & data) -> Eigen::Matrix3d {
    if (data.size() != 9) {
      throw std::runtime_error("rotation_matrix_gimbal_to_imu 数据长度不是 9");
    }
    return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(data.data());
  };

  if (diag_mode_) {
    pose_source_ = "gimbal";
    gimbal_ = std::make_unique<io::Gimbal>(app_config.gimbal);
    r_gimbal_to_imu_ = unflatten_3x3(app_config.solver.coord_converter.rotation_matrix_gimbal_to_imu);
    std::filesystem::create_directories(output_folder_);
    utils::logger()->info("[Capture/Diag] 进入诊断模式，不打开相机");
    utils::logger()->info("[Capture/Diag] 输出文件夹: {}", output_folder_);
    return;
  }

  if (pose_source_ != "gimbal" && pose_source_ != "dm_imu") {
    throw std::invalid_argument("pose_source must be 'gimbal' or 'dm_imu'");
  }

  camera_ = std::make_unique<camera::Camera>(app_config.camera);
  gimbal_ = std::make_unique<io::Gimbal>(app_config.gimbal);
  if (pose_source_ == "dm_imu") {
    dm_imu_ = std::make_unique<io::DmImu>(app_config.dm_imu);
  }
  // 创建输出文件夹
  std::filesystem::create_directories(output_folder_);

  utils::logger()->info("[Capture] 模块初始化完成");
  utils::logger()->info(
    "[Capture] 棋盘格内角点: {}x{} (列x行)", pattern_size_.width, pattern_size_.height);
  utils::logger()->info(
    "[Capture] 单格边长: {:.1f} mm, 板尺寸约: {}x{} mm",
    square_size_mm_, board_size_mm_.width, board_size_mm_.height);
  utils::logger()->info("[Capture] 姿态来源: {}", pose_source_);
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

Eigen::Quaterniond CaptureApp::pose_at(std::chrono::steady_clock::time_point timestamp)
{
  if (pose_source_ == "dm_imu") {
    return dm_imu_->imu_at(timestamp);
  }
  return gimbal_->q(timestamp);
}

int CaptureApp::run()
{
  if (diag_mode_) return run_diag();

  cv::namedWindow(window_name_, cv::WINDOW_NORMAL);

  int count = 0;
  const calibration::PatternConfig pattern{pattern_size_, square_size_mm_};

  while (!quit_.load()) {
    if (g_stop_requested.load()) {
      break;
    }

    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;

    camera_->read(img, timestamp);
    Eigen::Quaterniond q = pose_at(timestamp);

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
    const bool preview_success =
      calibration::find_chessboard_corners(preview_img, pattern, preview_centers, false);

    cv::drawChessboardCorners(img_with_ypr, pattern_size_, preview_centers, preview_success);

    cv::imshow(window_name_, img_with_ypr);
    auto key = cv::waitKey(1);

    if (key == 'q') {
      break;
    } else if (key != 's') {
      continue;
    }

    // 保存前做一次全分辨率棋盘格检测（仅作存帧闸门，无需亚像素精化）
    std::vector<cv::Point2f> centers;
    const bool success = calibration::find_chessboard_corners(img_bgr, pattern, centers, false);

    if (!success) {
      utils::logger()->warn("[Capture] 当前帧未检测到完整棋盘格，未保存");
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

void CaptureApp::write_diag_header(const std::string & csv_path)
{
  std::ofstream f(csv_path, std::ios::out | std::ios::trunc);
  f << "idx,t_ns,qw,qx,qy,qz,"
       "q_yaw_deg,q_pitch_deg,q_roll_deg,"
       "state_yaw_deg,state_pitch_deg,"
       "g2w_yaw_deg,g2w_pitch_deg,g2w_roll_deg\n";
}

void CaptureApp::append_diag_row(
  const std::string & csv_path,
  int idx,
  std::chrono::steady_clock::time_point t,
  const Eigen::Quaterniond & q,
  const Eigen::Vector3d & q_ypr_deg,
  const io::GimbalState & state,
  const Eigen::Vector3d & g2w_ypr_deg)
{
  std::ofstream f(csv_path, std::ios::app);
  f << std::fixed << std::setprecision(4);
  const auto t_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
  f << idx << "," << t_ns << ","
    << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
    << q_ypr_deg[0] << "," << q_ypr_deg[1] << "," << q_ypr_deg[2] << ","
    << (state.yaw  * 57.2957795) << "," << (state.pitch * 57.2957795) << ","
    << g2w_ypr_deg[0] << "," << g2w_ypr_deg[1] << "," << g2w_ypr_deg[2] << "\n";
}

int CaptureApp::run_diag()
{
  const std::string csv_path = output_folder_ + "/diag_snapshots.csv";
  write_diag_header(csv_path);

  cv::Mat hint(100, 480, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::putText(hint, "Press s: snapshot,  q/ESC: quit",
              {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
  const std::string win = "gimbal diag";
  cv::namedWindow(win, cv::WINDOW_NORMAL);
  cv::imshow(win, hint);

  int snapshot_idx = 0;
  while (!quit_.load() && !g_stop_requested.load()) {
    const auto t_now = std::chrono::steady_clock::now();
    const Eigen::Quaterniond q = gimbal_->q(t_now);
    const auto state = gimbal_->state();

    const Eigen::Matrix3d R_q = q.toRotationMatrix();
    const Eigen::Matrix3d R_g2w =
      r_gimbal_to_imu_.transpose() * R_q * r_gimbal_to_imu_;

    const Eigen::Vector3d q_ypr_deg = utils::eulers(R_q,   2, 1, 0) * 57.2957795;
    const Eigen::Vector3d g_ypr_deg = utils::eulers(R_g2w, 2, 1, 0) * 57.2957795;

    std::cout << "\033[H\033[2J"
              << std::fixed << std::setprecision(3)
              << "=== gimbal pose diag (press s: snapshot, q: quit) ===\n"
              << "raw q (wxyz)     : "
              << q.w() << "  " << q.x() << "  " << q.y() << "  " << q.z() << "\n"
              << "q.toR -> ZYX deg : yaw=" << q_ypr_deg[0]
              << "  pitch="                << q_ypr_deg[1]
              << "  roll="                 << q_ypr_deg[2] << "\n"
              << "state (rx) deg   : yaw=" << (state.yaw  * 57.2957795)
              << "  pitch="                << (state.pitch * 57.2957795) << "\n"
              << "R_g2w  -> ZYX deg: yaw=" << g_ypr_deg[0]
              << "  pitch="                << g_ypr_deg[1]
              << "  roll="                 << g_ypr_deg[2] << "\n"
              << "snapshots saved  : "    << snapshot_idx << "\n"
              << "csv path         : "    << csv_path     << "\n";
    std::cout.flush();

    cv::imshow(win, hint);
    const int key = cv::waitKey(50);
    if (key == 'q' || key == 27) break;
    if (key == 's') {
      append_diag_row(csv_path, ++snapshot_idx, t_now, q, q_ypr_deg, state, g_ypr_deg);
    }
  }
  cv::destroyWindow(win);
  return 0;
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
    Application::CaptureApp app(app_config::AppConfig::load(config_path), output_folder);
    int ret = app.run();
    rclcpp::shutdown();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[Capture] 程序异常终止: {}", e.what());
  }

  rclcpp::shutdown();
  return 1;
}
