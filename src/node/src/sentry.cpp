#include "sentry.hpp"

#include <csignal>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include "logger.hpp"
#include "draw_tools.hpp"
#include "yaml.hpp"

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

PipelineApp::PipelineApp(const std::string & config_path)
: config_path_(config_path),
  start_time_(std::chrono::steady_clock::now()),
  last_delay_log_time_(std::chrono::steady_clock::now())
{
  ros_node_ = std::make_shared<rclcpp::Node>("sentry_node");

  // Debug Topics
  debug_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Debug>(
    "debug", rclcpp::QoS(10));
  orientation_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Orienta>(
    "orientation", rclcpp::QoS(10));
  target_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Outpost>(
    "target", rclcpp::QoS(10));

  // 导航通讯 Topics
  cmd_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::SentryCmd>(
    "sentry_cmd", rclcpp::QoS(10));
  tf_complete_pub_ = ros_node_->create_publisher<sensor_msgs::msg::JointState>(
    "joint_states", rclcpp::QoS(10));

  vel_sub_ = ros_node_->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", rclcpp::QoS(10),
    [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
      nav_vx_.store(static_cast<float>(msg->linear.x));
      nav_vy_.store(static_cast<float>(msg->linear.y));
      nav_w_.store(static_cast<float>(msg->angular.z));
    });

  gimbal_active_sub_ = ros_node_->create_subscription<std_msgs::msg::Bool>(
    "gimbal_active", rclcpp::QoS(10),
    [this](const std_msgs::msg::Bool::SharedPtr msg) {
      gimbal_active_.store(msg->data);
    });

  camera_ = std::make_unique<camera::Camera>(config_path_);
  // dm_imu_ = std::make_unique<io::DmImu>(config_path_);
  detector_ = std::make_unique<armor_auto_aim::Detector>(config_path_);
  solver_ = std::make_unique<solver::Solver>(config_path_);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(config_path_, *solver_);
  planner_ = std::make_unique<plan::Planner>(config_path_);
  sentry_ = std::make_unique<io::Sentry>(config_path_);

  // 注册下位机接收回调：发布SentryCmd和JointState
  sentry_->set_receive_callback([this](const io::lowerToSentry& data) {
    if (cmd_pub_) {
      auto msg = autoaim_msgs::msg::SentryCmd{};
      msg.start_nav = (data.sentry_nav == 1);
      msg.low_health = (data.low_health == 1);
      msg.resupply_done = (data.resupply_done == 1);
      cmd_pub_->publish(msg);
    }
    if (tf_complete_pub_) {
      auto msg = sensor_msgs::msg::JointState{};
      msg.header.stamp = ros_node_->now();
      msg.name = {"gimbal_yaw_odom_joint", "gimbal_yaw_joint", "gimbal_pitch_joint"};
      msg.position = {
        static_cast<double>(data.yaw_odom),
        static_cast<double>(data.yaw),
        static_cast<double>(data.pitch)
      };
      tf_complete_pub_->publish(msg);
    }
  });

  // 读取扫描参数
  auto yaml = utils::load(config_path_);
  auto scan_node = yaml["Scan"];
  if (scan_node) {
    scan_yaw_center_ = scan_node["yaw_center"].as<float>(0.0f);
    scan_yaw_amplitude_ = scan_node["yaw_amplitude"].as<float>(1.5f);
    scan_yaw_period_ = scan_node["yaw_period"].as<float>(6.0f);
    scan_pitch_center_ = scan_node["pitch_center"].as<float>(-0.05f);
    scan_pitch_amplitude_ = scan_node["pitch_amplitude"].as<float>(0.15f);
    scan_pitch_period_ = scan_node["pitch_period"].as<float>(3.0f);
    utils::logger()->info(
      "[Pipeline] 扫描参数: yaw[{:.2f}±{:.2f}, T={:.1f}s] pitch[{:.2f}±{:.2f}, T={:.1f}s]",
      scan_yaw_center_, scan_yaw_amplitude_, scan_yaw_period_,
      scan_pitch_center_, scan_pitch_amplitude_, scan_pitch_period_);
  }

  visualization_frame_counter_.store(0);

  if (enable_visualization_) {
    utils::logger()->info("[Pipeline] 启用实时可视化输出");
  } else {
    utils::logger()->info("[Pipeline] 可视化已关闭");
  }

  utils::logger()->info("[Pipeline] 模块初始化完成");
}

PipelineApp::~PipelineApp()
{
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
    auto t0 = timestamp;

    DebugPacket debug_packet;
    timestamp_sec = utils::delta_time(timestamp, start_time_);

    cv::cvtColor(img, debug_packet.rgb_image, cv::COLOR_BGR2RGB);

    // orientation = dm_imu_->imu_at(timestamp);
    orientation = sentry_->q(timestamp);

    solver_->updateIMU(orientation, timestamp_sec);

    auto armor = detector_->detect(debug_packet.rgb_image);

    std::list<armor_auto_aim::Armor> armor_list(
      armor.begin(), armor.end());
    auto targets = tracker_->track(armor_list, timestamp);

    auto t5 = std::chrono::steady_clock::now();

    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    // 计算系统总延迟并统计
    double total_delay_ms = std::chrono::duration<double, std::milli>(t5 - t0).count();
    delay_window_.push_back(total_delay_ms);
    if (delay_window_.size() > delay_window_size_) {
      delay_window_.pop_front();
    }

    // 每5秒输出一次统计信息
    auto now = std::chrono::steady_clock::now();
    if (now - last_delay_log_time_ > std::chrono::seconds(5) && delay_window_.size() >= 10) {
      std::vector<double> sorted_delays(delay_window_.begin(), delay_window_.end());
      std::sort(sorted_delays.begin(), sorted_delays.end());
      size_t p95_idx = static_cast<size_t>(sorted_delays.size() * 0.95);
      double p95_delay = sorted_delays[p95_idx];
      double avg_delay = std::accumulate(sorted_delays.begin(), sorted_delays.end(), 0.0) / sorted_delays.size();
      utils::logger()->info(
        "[Pipeline] 系统延迟统计 - 平均: {:.2f}ms, 95分位: {:.2f}ms, 最大: {:.2f}ms",
        avg_delay, p95_delay, sorted_delays.back());
      last_delay_log_time_ = now;
    }

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
    if (enable_visualization_) {
      visualization_queue.push(DebugPacket{});
    }
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
  // ROS spin线程，用于接收订阅消息
  spin_thread_ = std::thread([this]() {
    while (!quit_.load() && rclcpp::ok()) {
      rclcpp::spin_some(ros_node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
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

  auto last_log_time = std::chrono::steady_clock::now();
  bool was_scanning = false;
  std::chrono::steady_clock::time_point scan_start_time;

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

    auto plan_result = planner_->plan(target, bullet_speed_);
    auto gs = sentry_->state();

    // 开火判断
    bool enable_shoot = std::abs(plan_result.yaw - gs.yaw) < 0.02
                     && std::abs(plan_result.pitch - gs.pitch) < 0.015;
    if (enable_shoot) {
      plan_result.fire = true;
    }

    // 确定mode和yaw/pitch
    uint8_t mode = 0;
    float send_yaw = 0.0f, send_pitch = 0.0f;
    bool active = gimbal_active_.load();

    if (!active) {
      // 导航说不可控 → 静止
      mode = 0;
      send_yaw = 0.0f;
      send_pitch = 0.0f;
      was_scanning = false;
    } else if (!plan_result.control) {
      // 可控但没有目标 → 扫描
      mode = 4;
      if (!was_scanning) {
        scan_start_time = std::chrono::steady_clock::now();
        was_scanning = true;
      }
      double t = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - scan_start_time).count();
      send_yaw = scan_yaw_center_ +
          scan_yaw_amplitude_ * static_cast<float>(std::sin(2.0 * M_PI / scan_yaw_period_ * t));
      send_pitch = scan_pitch_center_ +
          scan_pitch_amplitude_ * static_cast<float>(std::sin(2.0 * M_PI / scan_pitch_period_ * t));
    } else {
      // 有目标 → 控制/开火
      was_scanning = false;
      mode = plan_result.fire ? 2 : 1;
      send_yaw = plan_result.yaw;
      send_pitch = plan_result.pitch;
    }

    // 发送到下位机，vx/vy/w始终透传导航速度
    // sentry_->send(mode, send_yaw, send_pitch,
    //               nav_vx_.load(), nav_vy_.load(), nav_w_.load());
    
    sentry_->send(mode, 0.0, 0.0,
                  0.0, 0.0, 0.0);

    if (debug_pub_) {
      auto msg = autoaim_msgs::msg::Debug{};
      msg.enable_control = plan_result.control;
      msg.fire = plan_result.fire;
      msg.yaw_offest = plan_result.yaw - gs.yaw;
      msg.yaw = plan_result.yaw;
      msg.pitch = plan_result.pitch;
      msg.yaw_gimbal = gs.yaw;
      msg.pitch_gimbal = gs.pitch;
      debug_pub_->publish(msg);
    }

   // 发布Target状态消息
    if (target_pub_ && target.has_value()) {
      auto target_msg = autoaim_msgs::msg::Outpost{};
      std::visit([&target_msg](const auto & t) {
        const auto & ekf = t.ekf();
        target_msg.h1 = static_cast<float>(ekf.x[9]);
        target_msg.p_h1 = static_cast<float>(ekf.P(9, 9));
        target_msg.h2 = static_cast<float>(ekf.x[10]);
        target_msg.p_h2 = static_cast<float>(ekf.P(10, 10));
      }, target.value());
      target_pub_->publish(target_msg);
    }

    auto now = std::chrono::steady_clock::now();
    if (
      plan_result.control && now - last_log_time >
      std::chrono::milliseconds(200)) {
      last_log_time = now;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  utils::logger()->info("[Pipeline] 规划线程退出");
}

}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | YAML配置文件路径}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  std::string config_path = "/home/guo/ITL_Auto_aim/src/config/sentry.yaml";
  if (cli.has("@config-path")) {
    config_path = cli.get<std::string>("@config-path");
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
