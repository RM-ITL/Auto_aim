#include "record_node.hpp"

#include <csignal>
#include <filesystem>
#include <stdexcept>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include "app_config/app_config.hpp"
#include "logger.hpp"

namespace Application
{

namespace
{
std::atomic<bool> g_stop_requested{false};
RecordApp * g_app_instance{nullptr};

void handle_signal(int)
{
  g_stop_requested.store(true);
  if (g_app_instance) {
    g_app_instance->request_stop();
  }
}
}  // namespace

RecordApp::RecordApp(const app_config::AppConfig & app_config)
: last_log_time_(std::chrono::steady_clock::now())
{
  ros_node_ = std::make_shared<rclcpp::Node>("record_node");

  imu_source_name_ = ros_node_->declare_parameter<std::string>("imu_source", "gimbal");
  if (imu_source_name_ != "gimbal" && imu_source_name_ != "dm_imu") {
    throw std::invalid_argument("imu_source 只能是 gimbal 或 dm_imu");
  }
  imu_source_ = (imu_source_name_ == "dm_imu") ? ImuSource::DmImu : ImuSource::Gimbal;

  image_topic_ = ros_node_->declare_parameter<std::string>("image_topic", image_topic_);
  imu_topic_ = ros_node_->declare_parameter<std::string>("imu_topic", imu_topic_);
  frame_id_ = ros_node_->declare_parameter<std::string>("frame_id", frame_id_);

  // 录制用可靠 QoS + 深队列，避免 bag record 偶发丢帧。
  auto qos = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
  image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(image_topic_, qos);
  imu_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Orienta>(imu_topic_, qos);

  camera_ = std::make_unique<camera::Camera>(app_config.camera);
  gimbal_ = std::make_unique<io::Gimbal>(app_config.gimbal);
  if (imu_source_ == ImuSource::DmImu) {
    dm_imu_ = std::make_unique<io::DmImu>(app_config.dm_imu);
  }

  utils::logger()->info("[Record] 模块初始化完成");
  utils::logger()->info("[Record] 姿态来源: {}", imu_source_name_);
  utils::logger()->info("[Record] 图像话题: {}", image_topic_);
  utils::logger()->info("[Record] 四元数话题: {}", imu_topic_);
  utils::logger()->info(
    "[Record] 相机类型: {} (BGR8 直发，不做色彩转换)", camera_->camera_type());
  g_app_instance = this;
}

RecordApp::~RecordApp()
{
  g_app_instance = nullptr;
  request_stop();
}

Eigen::Quaterniond RecordApp::pose_at(std::chrono::steady_clock::time_point timestamp)
{
  if (imu_source_ == ImuSource::DmImu) {
    return dm_imu_->imu_at(timestamp);
  }
  return gimbal_->q(timestamp);
}

rclcpp::Time RecordApp::to_ros_time(std::chrono::steady_clock::time_point timestamp)
{
  // 用相机采集时刻的 steady_clock 纳秒作为 header.stamp。
  // 图像与四元数共用同一个值，回放侧按 stamp 对齐；单调时钟保证严格递增。
  const auto ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count();
  return rclcpp::Time(ns, RCL_STEADY_TIME);
}

int RecordApp::run()
{
  utils::logger()->info("[Record] 开始录制发布，Ctrl-C 停止");

  while (!quit_.load()) {
    if (g_stop_requested.load()) {
      break;
    }

    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;

    camera_->read(img, timestamp);

    // camera stop 后 read() 返回空帧，退出。
    if (quit_.load() || g_stop_requested.load() || img.empty()) {
      break;
    }

    // 与 test_node_deep 保持一致：四元数取采集时刻前 1ms，规避采集与 IMU 到达的相位差。
    const Eigen::Quaterniond q = pose_at(timestamp - std::chrono::milliseconds(1));

    const rclcpp::Time stamp = to_ros_time(timestamp);

    // 相机输出统一为 BGR8：hik 已是 BGR，mindvision 为 RGB，需转一次。
    cv::Mat bgr;
    if (camera_->camera_type() == "mindvision") {
      cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
    } else {
      bgr = img;
    }

    std_msgs::msg::Header header;
    header.stamp = stamp;
    header.frame_id = frame_id_;

    auto image_msg = cv_bridge::CvImage(header, "bgr8", bgr).toImageMsg();
    image_pub_->publish(*image_msg);

    autoaim_msgs::msg::Orienta imu_msg;
    imu_msg.w = static_cast<float>(q.w());
    imu_msg.x = static_cast<float>(q.x());
    imu_msg.y = static_cast<float>(q.y());
    imu_msg.z = static_cast<float>(q.z());
    imu_msg.dm_w = 0.0f;
    imu_msg.dm_x = 0.0f;
    imu_msg.dm_y = 0.0f;
    imu_msg.dm_z = 0.0f;
    imu_pub_->publish(imu_msg);

    frame_count_++;
    frame_count_window_++;

    const auto now = std::chrono::steady_clock::now();
    if (now - last_log_time_ >= std::chrono::seconds(1)) {
      const double elapsed = std::chrono::duration<double>(now - last_log_time_).count();
      const double fps = elapsed > 0.0 ? frame_count_window_ / elapsed : 0.0;
      utils::logger()->info(
        "[Record] 已发布 {} 帧，当前 {:.1f} fps", frame_count_, fps);
      last_log_time_ = now;
      frame_count_window_ = 0;
    }
  }

  request_stop();
  utils::logger()->info("[Record] 录制结束，共发布 {} 帧", frame_count_);
  utils::logger()->warn("[Record] 注意四元数顺序为 wxyz");
  return 0;
}

void RecordApp::request_stop()
{
  if (!quit_.exchange(true)) {
    if (gimbal_) {
      gimbal_->stop();
    }
    if (camera_) {
      camera_->stop();
    }
    utils::logger()->info("[Record] 正在停止...");
  }
}

}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | src/config/standard3.yaml | YAML配置文件路径 }";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  std::string config_path =
    std::filesystem::current_path().string() + "/src/config/standard3.yaml";
  if (cli.has("@config-path")) {
    config_path = cli.get<std::string>("@config-path");
  }

  rclcpp::init(argc, argv);
  std::signal(SIGINT, Application::handle_signal);

  try {
    Application::RecordApp app(app_config::AppConfig::load(config_path));
    int ret = app.run();
    rclcpp::shutdown();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[Record] 程序异常终止: {}", e.what());
  }

  rclcpp::shutdown();
  return 1;
}
