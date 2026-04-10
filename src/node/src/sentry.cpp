#include "sentry.hpp"

#include <csignal>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include "logger.hpp"
#include "draw_tools.hpp"

namespace Application
{
namespace
{
std::atomic<bool> g_stop_requested{false};
PipelineApp* g_app_instance{nullptr};

SentryRunMode parse_run_mode(const std::string & value)
{
  if (value == "direct") {
    return SentryRunMode::Direct;
  }
  if (value == "bridge") {
    return SentryRunMode::Bridge;
  }
  throw std::invalid_argument("run_mode 只能是 direct 或 bridge");
}

void handle_signal(int)
{
  g_stop_requested.store(true);
  if (g_app_instance) {
    g_app_instance->request_stop();
  }
}

}  // namespace

PipelineApp::PipelineApp(const std::string & config_path)
: config_path_(config_path),
  start_time_(std::chrono::steady_clock::now()),
  last_delay_log_time_(std::chrono::steady_clock::now()),
  last_state_wait_log_time_(std::chrono::steady_clock::now())
{
  ros_node_ = std::make_shared<rclcpp::Node>("sentry_node");
  run_mode_name_ = ros_node_->declare_parameter<std::string>("run_mode", "direct");
  run_mode_ = parse_run_mode(run_mode_name_);
  enable_visualization_ = ros_node_->declare_parameter<bool>(
    "enable_visualization", run_mode_ == SentryRunMode::Direct);

  // Debug Topics
  debug_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Debug>(
    "debug", rclcpp::QoS(10));
  orientation_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Orienta>(
    "orientation", rclcpp::QoS(10));
  target_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Outpost>(
    "target", rclcpp::QoS(10));

  camera_ = std::make_unique<camera::Camera>(config_path_);
  detector_ = std::make_unique<armor_auto_aim::Detector>(config_path_);
  solver_ = std::make_unique<solver::Solver>(config_path_);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(config_path_, *solver_);
  planner_ = std::make_unique<plan::Planner>(config_path_);
  shooter_ = std::make_unique<shooter::Shooter>(config_path_);

  if (run_mode_ == SentryRunMode::Direct) {
    sentry_ = std::make_unique<io::Sentry>(config_path_);
  } else {
    gimbal_cmd_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::GimbalCmd>(
      "/gimbal/cmd", rclcpp::QoS(10));
    gimbal_state_sub_ = ros_node_->create_subscription<autoaim_msgs::msg::GimbalState>(
      "/gimbal/state", rclcpp::SensorDataQoS(),
      [this](const autoaim_msgs::msg::GimbalState::SharedPtr msg) {
        gimbal_state_callback(msg);
      });
  }

  visualization_frame_counter_.store(0);

  if (enable_visualization_) {
    utils::logger()->info("[Pipeline] 启用实时可视化输出");
  } else {
    utils::logger()->info("[Pipeline] 可视化已关闭");
  }

  utils::logger()->info(
    "[Pipeline] 模块初始化完成，运行模式: {}", run_mode_name_);
  g_app_instance = this;
}

PipelineApp::~PipelineApp()
{
  g_app_instance = nullptr;
  request_stop();
  join_threads();
}

int PipelineApp::run()
{
  start_threads();
  std::string last_state = tracker_->state();

  while (!quit_.load()) {
    if (g_stop_requested.load()) {
      break;
    }

    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    double timestamp_sec{0.0};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};

    camera_->read(img, timestamp);

    // camera shutdown 后 read() 会返回空帧，检查退出
    if (quit_.load() || g_stop_requested.load() || img.empty()) {
      break;
    }

    if (!has_runtime_state()) {
      auto now = std::chrono::steady_clock::now();
      if (utils::delta_time(last_state_wait_log_time_, now) >= 1.0) {
        utils::logger()->warn("[Pipeline] 尚未收到有效状态输入，等待中...");
        last_state_wait_log_time_ = now;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    DebugPacket debug_packet;
    timestamp_sec = utils::delta_time(timestamp, start_time_);

    cv::cvtColor(img, debug_packet.rgb_image, cv::COLOR_BGR2RGB);

    orientation = runtime_orientation(timestamp);

    if (orientation_pub_) {
      auto msg = autoaim_msgs::msg::Orienta{};
      msg.w = orientation.w(),
      msg.x = orientation.x(),
      msg.y = orientation.y(),
      msg.z = orientation.z(),
      // msg.dm_w = dm_orientation.w(),
      // msg.dm_x = dm_orientation.x(),
      // msg.dm_y = dm_orientation.y(),
      // msg.dm_z = dm_orientation.z(),
      msg.dm_w = 0,
      msg.dm_x = 0,
      msg.dm_y = 0,
      msg.dm_z = 0,        
      orientation_pub_->publish(msg);
    }

    solver_->updateIMU(orientation, timestamp_sec);

    auto armor = detector_->detect(debug_packet.rgb_image);

    std::list<armor_auto_aim::Armor> armor_list(
      armor.begin(), armor.end());
    auto targets = tracker_->track(armor_list, timestamp);

    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);


    if (enable_visualization_) {
      debug_packet.reprojected_armors.reserve(targets.size() * 4);
    }

    bool is_first_target = true;
    for (const auto & target : targets) {
      const auto armor_xyza_list = std::visit(
        [](const auto & t) { return t.armor_xyza_list(); }, target);
      const auto armor_type = std::visit(
        [](const auto & t) { return t.armor_type; }, target);
      const auto target_name = std::visit(
        [](const auto & t) { return t.name; }, target);

      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        Eigen::Vector3d world_point(xyza.x(), xyza.y(), xyza.z());
        auto image_points =
          yaw_optimizer_->reproject_armor_out(world_point, xyza[3], armor_type, target_name);

        if (image_points.size() == 4) {
          if (is_first_target) {
            is_first_target = false;
          }

          if (enable_visualization_) {
            Visualization vis_armor;
            std::copy(image_points.begin(), image_points.end(), vis_armor.corners.begin());
            vis_armor.name = target_name;
            vis_armor.type = armor_type;
            debug_packet.reprojected_armors.push_back(vis_armor);
          }
        }
      }
    }

    debug_packet.tracker_state = tracker_->state();
    debug_packet.valid = true;

    if (enable_visualization_) {
      visualization_queue.push(debug_packet);
    }

    if (debug_packet.tracker_state != last_state) {
      utils::logger()->info(
        "[Pipeline] Tracker状态切换: {} -> {}", last_state, debug_packet.tracker_state);
      last_state = debug_packet.tracker_state;
    }
  }

  request_stop();
  join_threads();
  return 0;
}

void PipelineApp::request_stop()
{
  if (!quit_.exchange(true)) {
    // 关闭所有队列，唤醒阻塞在 pop/front 上的线程
    visualization_queue.shutdown();
    target_queue.shutdown();

    // 停止 sentry，唤醒阻塞在 q() 中的主线程
    if (run_mode_ == SentryRunMode::Direct && sentry_) {
      sentry_->stop();
    }

    // 关闭相机，使 camera_->read() 不再阻塞
    if (camera_) {
      camera_->stop();
    }

    utils::logger()->info("[Pipeline] 正在停止...");
  }
}

void PipelineApp::start_threads()
{
  quit_.store(false);
  visualization_frame_counter_.store(0);
  if (planner_) {
    planner_thread_ = std::thread(&PipelineApp::planner_loop, this);
  }
  if (enable_visualization_) {
    visualization_thread_ = std::thread(&PipelineApp::visualization_loop, this);
  }
  if (run_mode_ == SentryRunMode::Bridge) {
    spin_thread_ = std::thread([this]() {
      while (!quit_.load() && rclcpp::ok()) {
        rclcpp::spin_some(ros_node_);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }
}

void PipelineApp::join_threads()
{
  if (planner_thread_.joinable()) {
    planner_thread_.join();
  }
  if (visualization_thread_.joinable()) {
    visualization_thread_.join();
  }
  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }
}

void PipelineApp::visualization_loop()
{
  utils::logger()->info("[Pipeline] 可视化线程启动");
  try {
    cv::namedWindow(visualization_window_name_, cv::WINDOW_NORMAL);
  } catch (const cv::Exception & e) {
    utils::logger()->error("[Pipeline] 创建可视化窗口失败: {}", e.what());
    return;
  }

  while (!quit_.load()) {
    if (g_stop_requested.load()) {
      break;
    }

    DebugPacket packet;
    visualization_queue.pop(packet);

    if (!packet.valid) {
      break;
    }

    try {
      cv::Mat canvas = packet.rgb_image.clone();
      const int frame_index = visualization_frame_counter_.fetch_add(1) + 1;
      detector_->visualize_results(canvas, packet.reprojected_armors, visualization_center_point_, frame_index);

      cv::imshow(visualization_window_name_, canvas);
      cv::waitKey(1);
    } catch (const std::exception & e) {
      utils::logger()->warn("[Pipeline] 可视化帧处理失败: {}", e.what());
    }
  }

  cv::destroyWindow(visualization_window_name_);
  utils::logger()->info("[Pipeline] 可视化线程退出");
}

void PipelineApp::planner_loop()
{
  utils::logger()->info("[Pipeline] 规划线程启动");

  while (!quit_.load()) {
    if (g_stop_requested.load()) {
      break;
    }

    if (!planner_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    if (target_queue.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    auto target = target_queue.front();

    auto gs = runtime_state();

    auto plan_result = planner_->plan(target, gs.bullet_speed);

    if (target.has_value()) {
      bool enable_shoot = shooter_->checkfire(
        plan_result.yaw, plan_result.pitch, gs, target.value());
      plan_result.fire = plan_result.fire && enable_shoot;
    }

    // 安全兜底：control=true 但值含 NaN 时回退到 gs
    bool safe = plan_result.control
        && std::isfinite(plan_result.yaw)
        && std::isfinite(plan_result.pitch);

    const bool fire = plan_result.fire && safe;
    const float send_yaw = safe ? static_cast<float>(plan_result.yaw) : gs.yaw;
    const float send_pitch = safe ? static_cast<float>(plan_result.pitch) : gs.pitch;
    const float send_yaw_vel = safe ? static_cast<float>(plan_result.yaw_vel) : 0.0f;
    if (!safe && plan_result.control) {
      utils::logger()->warn(
        "[Pipeline] 安全回退: plan值异常 yaw={:.4f} pitch={:.4f}, 回退到gs yaw={:.4f} pitch={:.4f}",
        plan_result.yaw, plan_result.pitch, gs.yaw, gs.pitch);
    }

    output_control(plan_result, gs);

    // 统计滑动窗口内fire占比
    {
      auto now_fire = std::chrono::steady_clock::now();
      fire_window_.emplace_back(now_fire, fire);
      auto cutoff = now_fire - std::chrono::duration<double>(fire_window_sec_);
      while (!fire_window_.empty() && fire_window_.front().first < cutoff) {
        fire_window_.pop_front();
      }
    }

    if (debug_pub_) {
      // 计算fire_rate（窗口内fire帧占比）
      float fire_rate = 0.0f;
      if (!fire_window_.empty()) {
        int fire_count = 0;
        for (const auto & [t, f] : fire_window_) {
          if (f) fire_count++;
        }
        fire_rate = static_cast<float>(fire_count) / static_cast<float>(fire_window_.size());
      }

      auto msg = autoaim_msgs::msg::Debug{};
      msg.enable_control = safe;
      msg.fire = fire;
      msg.fire_rate = fire_rate;
      msg.yaw_offest = send_yaw - gs.yaw;
      msg.pitch_offset = send_pitch - gs.pitch;
      msg.yaw = send_yaw;
      msg.pitch = send_pitch;
      msg.yaw_gimbal = gs.yaw;
      msg.pitch_gimbal = gs.pitch;
      msg.bullet_speed = gs.bullet_speed;
      msg.yaw_vel = send_yaw_vel;
      debug_pub_->publish(msg);
    }


    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  utils::logger()->info("[Pipeline] 规划线程退出");
}

void PipelineApp::gimbal_state_callback(const autoaim_msgs::msg::GimbalState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(bridge_state_mutex_);

  bridge_state_.yaw = msg->yaw;
  bridge_state_.pitch = msg->pitch;
  bridge_state_.yaw_vel = msg->yaw_vel;
  bridge_state_.pitch_vel = msg->pitch_vel;
  bridge_state_.bullet_speed = msg->bullet_speed;
  bridge_state_.bullet_count = msg->bullet_count;

  Eigen::Quaterniond q(
    msg->orientation.w,
    msg->orientation.x,
    msg->orientation.y,
    msg->orientation.z);
  if (q.norm() < 1e-6) {
    bridge_orientation_ = Eigen::Quaterniond::Identity();
  } else {
    bridge_orientation_ = q.normalized();
  }

  bridge_has_state_ = true;
}

bool PipelineApp::has_runtime_state() const
{
  if (run_mode_ == SentryRunMode::Direct) {
    return sentry_ != nullptr;
  }
  std::lock_guard<std::mutex> lock(bridge_state_mutex_);
  return bridge_has_state_;
}

io::GimbalState PipelineApp::runtime_state() const
{
  if (run_mode_ == SentryRunMode::Direct) {
    return sentry_->state();
  }
  std::lock_guard<std::mutex> lock(bridge_state_mutex_);
  return bridge_state_;
}

Eigen::Quaterniond PipelineApp::runtime_orientation(std::chrono::steady_clock::time_point timestamp) const
{
  if (run_mode_ == SentryRunMode::Direct) {
    return sentry_->q(timestamp);
  }
  std::lock_guard<std::mutex> lock(bridge_state_mutex_);
  return bridge_orientation_;
}

void PipelineApp::output_control(const plan::Plan & plan_result, const io::GimbalState & gs)
{
  const bool safe = plan_result.control &&
    std::isfinite(plan_result.yaw) &&
    std::isfinite(plan_result.pitch);

  const bool fire = plan_result.fire && safe;
  const float send_yaw = safe ? static_cast<float>(plan_result.yaw) : gs.yaw;
  const float send_pitch = safe ? static_cast<float>(plan_result.pitch) : gs.pitch;
  const float send_yaw_vel = safe ? static_cast<float>(plan_result.yaw_vel) : 0.0f;
  const float send_pitch_vel = safe ? static_cast<float>(plan_result.pitch_vel) : 0.0f;
  const float send_yaw_acc = safe ? static_cast<float>(plan_result.yaw_acc) : 0.0f;
  const float send_pitch_acc = safe ? static_cast<float>(plan_result.pitch_acc) : 0.0f;

  if (run_mode_ == SentryRunMode::Direct) {
    const uint8_t send_mode = safe ? static_cast<uint8_t>(fire ? 2 : 1) : static_cast<uint8_t>(0);
    sentry_->send(send_mode, send_yaw, send_pitch, 0.0f, 0.0f, 0.0f);
    return;
  }

  auto gimbal_cmd_msg = autoaim_msgs::msg::GimbalCmd{};
  gimbal_cmd_msg.header.stamp = ros_node_->now();
  gimbal_cmd_msg.control = safe;
  gimbal_cmd_msg.fire_advice = fire;
  gimbal_cmd_msg.yaw = send_yaw;
  gimbal_cmd_msg.pitch = send_pitch;
  gimbal_cmd_msg.yaw_vel = send_yaw_vel;
  gimbal_cmd_msg.pitch_vel = send_pitch_vel;
  gimbal_cmd_msg.yaw_acc = send_yaw_acc;
  gimbal_cmd_msg.pitch_acc = send_pitch_acc;
  gimbal_cmd_pub_->publish(gimbal_cmd_msg);
}

}  // namespace Application

int main(int argc, char ** argv)
{
  bool show_help = false;
  std::string config_path = std::filesystem::current_path().string() + "/src/config/sentry.yaml";
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h" || arg == "help") {
      show_help = true;
      break;
    }
    if (arg == "--ros-args") {
      break;
    }
    if (!arg.empty() && arg[0] != '-') {
      config_path = arg;
      break;
    }
  }

  if (show_help) {
    std::cout << "用法: ros2 run pipeline sentry_node [config-path] [--ros-args -p run_mode:=direct|bridge]\n";
    return 0;
  }

  rclcpp::init(argc, argv);
  std::signal(SIGINT, Application::handle_signal);

  try {
    Application::PipelineApp app(config_path);
    int ret = app.run();
    rclcpp::shutdown();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[Pipeline] 程序异常终止: {}", e.what());
  }

  rclcpp::shutdown();
  return 1;
}
