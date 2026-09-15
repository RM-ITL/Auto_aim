#include "replay_node.hpp"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <list>
#include <map>
#include <stdexcept>
#include <variant>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "app_config/app_config.hpp"
#include "armor.hpp"
#include "logger.hpp"

namespace Application
{

using Visualization = armor_auto_aim::Visualization;

namespace
{
std::atomic<bool> g_stop_requested{false};
ReplayApp * g_app_instance{nullptr};

void handle_signal(int)
{
  g_stop_requested.store(true);
  if (g_app_instance) {
    g_app_instance->request_stop();
  }
}
}  // namespace

ReplayApp::ReplayApp(const app_config::AppConfig & app_config)
: last_log_time_(std::chrono::steady_clock::now())
{
  ros_node_ = std::make_shared<rclcpp::Node>("replay_node");

  image_topic_ = ros_node_->declare_parameter<std::string>("image_topic", image_topic_);
  imu_topic_ = ros_node_->declare_parameter<std::string>("imu_topic", imu_topic_);

  detector_ = std::make_unique<armor_auto_aim::Detector>(app_config.detector);
  solver_ = std::make_unique<solver::Solver>(app_config.solver);
  yaw_optimizer_ = solver_->getYawOptimizer();
  // 接上 tracker，使解算结果流入 EKF，从而产出 NIS/归一化残差用于一致性对比。
  tracker_ = std::make_unique<tracker::Tracker>(app_config.tracker, *solver_);

  orientation_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Orienta>(
    "orientation", rclcpp::QoS(10));

  // 与 record_node 一致：可靠 QoS + 深队列，避免回放丢帧。
  auto qos = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
  image_sub_ = ros_node_->create_subscription<sensor_msgs::msg::Image>(
    image_topic_, qos,
    [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) { image_callback(msg); });
  imu_sub_ = ros_node_->create_subscription<autoaim_msgs::msg::Orienta>(
    imu_topic_, qos,
    [this](const autoaim_msgs::msg::Orienta::ConstSharedPtr msg) { imu_callback(msg); });

  try {
    cv::namedWindow(visualization_window_name_, cv::WINDOW_NORMAL);
  } catch (const cv::Exception & e) {
    utils::logger()->warn("[Replay] 创建可视化窗口失败: {}", e.what());
  }

  utils::logger()->info("[Replay] 模块初始化完成");
  utils::logger()->info("[Replay] 图像话题: {}", image_topic_);
  utils::logger()->info("[Replay] 四元数话题: {}", imu_topic_);
  utils::logger()->info("[Replay] 检测+解算+tracker(EKF)，输出 [NIS-STAT] 一致性统计");
  g_app_instance = this;
}

ReplayApp::~ReplayApp()
{
  g_app_instance = nullptr;
  request_stop();
  cv::destroyWindow(visualization_window_name_);
}

void ReplayApp::request_stop()
{
  quit_.store(true);
}

void ReplayApp::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    // record 端以 bgr8 编码发布，这里同样按 bgr8 取回。
    last_bgr_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
  } catch (const cv_bridge::Exception & e) {
    utils::logger()->warn("[Replay] cv_bridge 转换失败: {}", e.what());
    return;
  }

  const std::int64_t stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
  if (!first_stamp_set_) {
    first_stamp_ns_ = stamp_ns;
    first_stamp_set_ = true;
  }
  last_ts_sec_ = static_cast<double>(stamp_ns - first_stamp_ns_) / 1e9;
  has_image_ = true;
}

void ReplayApp::imu_callback(const autoaim_msgs::msg::Orienta::ConstSharedPtr & msg)
{
  // Orienta 无 header/stamp，靠 record 端「先 image 后 imu」的 1:1 顺序配对：
  // 此刻 last_bgr_ 即本帧图像。首个 imu 若早于任何 image 则跳过。
  if (!has_image_ || last_bgr_.empty()) {
    return;
  }

  const Eigen::Quaterniond orientation(msg->w, msg->x, msg->y, msg->z);  // wxyz

  // 回显姿态，便于 rqt/plotjuggler 对齐观察。
  orientation_pub_->publish(*msg);

  cv::Mat rgb;
  cv::cvtColor(last_bgr_, rgb, cv::COLOR_BGR2RGB);

  solver_->updateIMU(orientation, last_ts_sec_);

  const auto armors = detector_->detect(rgb);

  std::vector<Visualization> reprojected;
  reprojected.reserve(armors.size());
  std::uint64_t pnp_ok = 0;
  for (const auto & armor : armors) {
    const auto pose = solver_->processArmor(armor, last_ts_sec_);
    if (!solver_->getLastPnPResult().success) {
      continue;
    }
    ++pnp_ok;
    const auto corners = yaw_optimizer_->reproject_armor_out(
      pose.world_position, pose.world_orientation.yaw, armor.type, armor.name);
    if (corners.size() == 4) {
      Visualization vis;
      std::copy(corners.begin(), corners.end(), vis.corners.begin());
      vis.name = armor.name;
      vis.type = armor.type;
      reprojected.push_back(vis);
    }
  }

  try {
    cv::Mat canvas = rgb.clone();
    detector_->visualize_results(canvas, reprojected, visualization_center_point_, ++frame_index_);
    cv::imshow(visualization_window_name_, canvas);
    cv::waitKey(1);
  } catch (const std::exception & e) {
    utils::logger()->warn("[Replay] 可视化失败: {}", e.what());
  }

  // tracker/EKF：用包内相对时间戳（单调递增），保证 dt 与录制时一致、回放确定。
  const auto track_t = std::chrono::steady_clock::time_point(
    std::chrono::nanoseconds(static_cast<std::int64_t>(last_ts_sec_ * 1e9)));
  std::list<armor_auto_aim::Armor> armor_list(armors.begin(), armors.end());
  auto targets = tracker_->track(armor_list, track_t);

  if (!targets.empty()) {
    // 取当前锁定目标的 EKF 诊断数据（Target/OutpostTarget 均暴露 ekf()）。
    const auto & data = std::visit(
      [](auto & tgt) -> const std::map<std::string, double> & { return tgt.ekf().data; },
      targets.front());

    const double nis = data.at("nis");
    ++nis_frames_window_;
    nis_sum_window_ += nis;
    nis_max_window_ = std::max(nis_max_window_, nis);
    if (nis > 9.488) ++nis_over_window_;  // χ²(4) 95% 上分位

    const char * nres_keys[4] = {"nres_yaw", "nres_pitch", "nres_distance", "nres_angle"};
    for (int i = 0; i < 4; ++i) {
      const double v = data.at(nres_keys[i]);
      nres_sq_sum_[i] += v * v;
    }
  }

  // 统计
  ++frame_count_window_;
  armor_count_window_ += armors.size();
  pnp_success_window_ += pnp_ok;
  const auto now = std::chrono::steady_clock::now();
  if (now - last_log_time_ >= std::chrono::seconds(1)) {
    const double elapsed = std::chrono::duration<double>(now - last_log_time_).count();
    const double fps = elapsed > 0.0 ? frame_count_window_ / elapsed : 0.0;
    const double pnp_rate =
      armor_count_window_ > 0
        ? 100.0 * static_cast<double>(pnp_success_window_) / static_cast<double>(armor_count_window_)
        : 0.0;
    utils::logger()->info(
      "[Replay] {:.1f} fps | 本窗口装甲板 {} 个，PnP 成功 {} ({:.0f}%)",
      fps, armor_count_window_, pnp_success_window_, pnp_rate);

    // NIS 一致性聚合：一致时 nis_mean≈4、over9.488≈5%、各维 nres_rms≈1。
    // 若 R 距离/yaw 对调，nres_distance 与 nres_angle 会明显偏离 1（一大一小）。
    if (nis_frames_window_ > 0) {
      const double n = static_cast<double>(nis_frames_window_);
      utils::logger()->info(
        "[NIS-STAT] frames={} nis_mean={:.2f} nis_max={:.2f} over9.488={:.1f}% "
        "nres_rms yaw={:.2f} pitch={:.2f} dist={:.2f} angle={:.2f}",
        nis_frames_window_, nis_sum_window_ / n, nis_max_window_,
        100.0 * static_cast<double>(nis_over_window_) / n,
        std::sqrt(nres_sq_sum_[0] / n), std::sqrt(nres_sq_sum_[1] / n),
        std::sqrt(nres_sq_sum_[2] / n), std::sqrt(nres_sq_sum_[3] / n));
    }

    last_log_time_ = now;
    frame_count_window_ = 0;
    armor_count_window_ = 0;
    pnp_success_window_ = 0;
    nis_frames_window_ = 0;
    nis_sum_window_ = 0.0;
    nis_max_window_ = 0.0;
    nis_over_window_ = 0;
    nres_sq_sum_[0] = nres_sq_sum_[1] = nres_sq_sum_[2] = nres_sq_sum_[3] = 0.0;
  }
}

int ReplayApp::run()
{
  utils::logger()->info("[Replay] 开始回放，等待 `ros2 bag play` 数据；Ctrl-C 停止");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(ros_node_);
  while (rclcpp::ok() && !quit_.load() && !g_stop_requested.load()) {
    executor.spin_some(std::chrono::milliseconds(10));
  }
  utils::logger()->info("[Replay] 回放结束");
  return 0;
}

}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? |                           | 输出命令行参数说明}"
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
    Application::ReplayApp app(app_config::AppConfig::load(config_path));
    int ret = app.run();
    rclcpp::shutdown();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[Replay] 程序异常终止: {}", e.what());
  }

  rclcpp::shutdown();
  return 1;
}
