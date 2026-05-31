#include "test_node_deep.hpp"

#include <cmath>
#include <csignal>
#include <algorithm>
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

#include "app_config/app_config.hpp"
#include "logger.hpp"
#include "draw_tools.hpp"

namespace Application
{
namespace
{
std::atomic<bool> g_stop_requested{false};
PipelineApp* g_app_instance{nullptr};

void handle_signal(int)
{
  g_stop_requested.store(true);
  if (g_app_instance) {
    g_app_instance->request_stop();
  }
}

void log_app_config(const std::string & prefix, const app_config::AppConfig & app_config)
{
  utils::logger()->info("[{}] config.absolute_path = {}", prefix, app_config.source_path);
  utils::logger()->info("[{}] config.file_size     = {} bytes", prefix, app_config.source_file_size);
  utils::logger()->info(
    "[{}] config.last_write_time_raw = {}", prefix, app_config.source_last_write_raw);
}


}  // namespace

PipelineApp::PipelineApp(const app_config::AppConfig & app_config)
: start_time_(std::chrono::steady_clock::now()),
  last_delay_log_time_(std::chrono::steady_clock::now())
{
  ros_node_ = std::make_shared<rclcpp::Node>("pipeline_debug_node");
  debug_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Debug>(
    "debug", rclcpp::QoS(10));
  orientation_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Orienta>(
    "orientation", rclcpp::QoS(10));
  target_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Target>(
    "target",rclcpp::QoS(10)
  );

  imu_source_name_ = ros_node_->declare_parameter<std::string>("imu_source", "gimbal");
  if (imu_source_name_ != "gimbal" && imu_source_name_ != "dm_imu") {
    throw std::invalid_argument("imu_source 只能是 gimbal 或 dm_imu");
  }
  imu_source_ = (imu_source_name_ == "dm_imu") ? ImuSource::DmImu : ImuSource::Gimbal;
  utils::logger()->info("[Pipeline] 姿态来源: {}", imu_source_name_);
  log_app_config("Pipeline", app_config);

  camera_ = std::make_unique<camera::Camera>(app_config.camera);
  if (imu_source_ == ImuSource::DmImu) {
    dm_imu_ = std::make_unique<io::DmImu>(app_config.dm_imu);
  }
  detector_ = std::make_unique<armor_auto_aim::Detector>(app_config.detector);
  solver_ = std::make_unique<solver::Solver>(app_config.solver);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(app_config.tracker, *solver_);
  planner_ = std::make_unique<plan::Planner>(app_config.planner);
  gimbal_ = std::make_unique<io::Gimbal>(app_config.gimbal);
  shooter_ = std::make_unique<shooter::Shooter>(app_config.shooter);

  // enable_visualization_ = detector_->config().enable_visualization;
  // visualization_center_point_ = detector_->config().center_point;
  visualization_frame_counter_.store(0);

  if (enable_visualization_) {
    utils::logger()->info("[Pipeline] 启用实时可视化输出");
  } else {
    utils::logger()->info("[Pipeline] 可视化已关闭");
  }

  utils::logger()->info("[Pipeline] 模块初始化完成");
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

    auto t0 = timestamp;  // 图像采集时间戳

    DebugPacket debug_packet;
    timestamp_sec = utils::delta_time(timestamp, start_time_);

    cv::cvtColor(img, debug_packet.rgb_image, cv::COLOR_BGR2RGB);

    orientation = (imu_source_ == ImuSource::DmImu)
                  ? dm_imu_->imu_at(timestamp)
                  : gimbal_->q(timestamp);

    if (orientation_pub_) {
      auto msg = autoaim_msgs::msg::Orienta{};
      msg.w = orientation.w(),
      msg.x = orientation.x(),
      msg.y = orientation.y(),
      msg.z = orientation.z(),
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

    auto t5 = std::chrono::steady_clock::now();  // 送入队列前的时间

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
      // 使用 std::visit 访问 variant 成员
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

        // utils::logger()->debug(
        //   "当前识别到的目标的yaw姿态为:{:.2f}",
        //   xyza[3]
        // );
        
        if (image_points.size() == 4) {
          // 如果是第一个target（即queue的front），计算并打印中心点
          if (is_first_target) {
            cv::Point2f center(0, 0);
            for (const auto& pt : image_points) {
              center += pt;
            }
            center.x /= 4.0f;
            center.y /= 4.0f;

            // utils::logger()->debug(
            //   "[Pipeline] Target queue front 重投影中心点: ({:.2f}, {:.2f})",
            //   center.x, center.y);
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

    // 停止 gimbal，唤醒阻塞在 q() 中的主线程
    if (gimbal_) {
      gimbal_->stop();
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
}

void PipelineApp::join_threads()
{
  if (planner_thread_.joinable()) {
    planner_thread_.join();
  }
  if (visualization_thread_.joinable()) {
    visualization_thread_.join();
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

      // if (canvas.channels() == 3) {
      //   cv::cvtColor(canvas, canvas, cv::COLOR_RGB2BGR);
      // }

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
  auto profile_window_start = std::chrono::steady_clock::now();
  int profile_loop_count = 0;
  int profile_empty_count = 0;
  double profile_front_sum_ms = 0.0;
  double profile_state_sum_ms = 0.0;
  double profile_plan_sum_ms = 0.0;
  double profile_checkfire_sum_ms = 0.0;
  double profile_send_sum_ms = 0.0;
  double profile_debug_sum_ms = 0.0;
  double profile_target_pub_sum_ms = 0.0;
  double profile_loop_sum_ms = 0.0;
  double profile_front_max_ms = 0.0;
  double profile_state_max_ms = 0.0;
  double profile_plan_max_ms = 0.0;
  double profile_checkfire_max_ms = 0.0;
  double profile_send_max_ms = 0.0;
  double profile_debug_max_ms = 0.0;
  double profile_target_pub_max_ms = 0.0;
  double profile_loop_max_ms = 0.0;

  auto update_profile = [](double value, double & sum, double & max) {
    sum += value;
    if (value > max) max = value;
  };

  auto elapsed_ms = [](const auto & begin, const auto & end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  };

  while (!quit_.load()) {
    const auto loop_start_time = std::chrono::steady_clock::now();

    if (g_stop_requested.load()) {
      break;
    }

    if (!planner_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    if (target_queue.empty()) {
      profile_empty_count++;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    const auto front_start_time = std::chrono::steady_clock::now();
    auto target = target_queue.front();
    const auto front_end_time = std::chrono::steady_clock::now();


    const auto state_start_time = std::chrono::steady_clock::now();
    auto gs = gimbal_->state();
    const auto state_end_time = std::chrono::steady_clock::now();

    const auto plan_start_time = std::chrono::steady_clock::now();
    auto plan_result = planner_->plan(target, gs.bullet_speed);
    const auto plan_end_time = std::chrono::steady_clock::now();


    const auto checkfire_start_time = std::chrono::steady_clock::now();
    if (target.has_value()) {
      bool enable_shoot = shooter_->checkfire(
        plan_result.yaw, plan_result.pitch, gs, target.value());
      plan_result.fire = plan_result.fire && enable_shoot;
    }
    const auto checkfire_end_time = std::chrono::steady_clock::now();

    const auto send_start_time = std::chrono::steady_clock::now();
    if (plan_result.control) {
      gimbal_->send(
        plan_result.control, plan_result.fire, plan_result.yaw, plan_result.yaw_vel,
        plan_result.yaw_acc, plan_result.pitch, plan_result.pitch_vel, plan_result.pitch_acc);
        // gimbal_->send_simple(plan_result.control, plan_result.fire, plan_result.yaw, plan_result.pitch);
    } else {
      gimbal_->send(false, false, gs.yaw, 0, 0, gs.pitch, 0, 0);
      // if (servo_compensator_) servo_compensator_->reset();
    }
    const auto send_end_time = std::chrono::steady_clock::now();

    //验证通讯帧率

    {
      static bool timers_initialized = false;
      static std::chrono::steady_clock::time_point last_send_time;
      static std::chrono::steady_clock::time_point window_start_time;
      static int send_count = 0;

      const auto send_time = std::chrono::steady_clock::now();

      if (!timers_initialized) {
        timers_initialized = true;
        last_send_time = send_time;
        window_start_time = send_time;
        send_count = 1;
      } else {
        auto dt_us =
          std::chrono::duration_cast<std::chrono::microseconds>(send_time - last_send_time);
        const double dt_ms = dt_us.count() / 1000.0;
        if (dt_ms > 20.0) {
          utils::logger()->debug("[Pipeline] gimbal send dt = {:.3f} ms", dt_ms);
        }
        last_send_time = send_time;
        send_count++;
      }

      auto window_elapsed = send_time - window_start_time;
      if (window_elapsed >= std::chrono::seconds(1)) {
        const double elapsed_sec = std::chrono::duration<double>(window_elapsed).count();
        const double freq_hz = elapsed_sec > 0.0 ? send_count / elapsed_sec : 0.0;
        utils::logger()->debug("[Pipeline] gimbal send freq = {:.1f} Hz", freq_hz);
        window_start_time = send_time;
        send_count = 0;
      }
    }
      
    // 统计滑动窗口内fire占比 和 offset
    {
      auto now_fire = std::chrono::steady_clock::now();
      fire_window_.emplace_back(now_fire, plan_result.fire);
      offset_window_.push_back({now_fire,
        static_cast<double>(plan_result.yaw - gs.yaw),
        static_cast<double>(plan_result.pitch - gs.pitch)});
      // 移除超出时间窗口的旧记录
      auto cutoff = now_fire - std::chrono::duration<double>(fire_window_sec_);
      while (!fire_window_.empty() && fire_window_.front().first < cutoff) {
        fire_window_.pop_front();
      }
      while (!offset_window_.empty() && offset_window_.front().time < cutoff) {
        offset_window_.pop_front();
      }
    }

    const auto debug_start_time = std::chrono::steady_clock::now();
    if (debug_pub_) {
      // 计算fire_rate
      float fire_rate = 0.0f;
      if (!fire_window_.empty()) {
        int fire_count = 0;
        for (const auto & [t, f] : fire_window_) {
          if (f) fire_count++;
        }
        fire_rate = static_cast<float>(fire_count) / static_cast<float>(fire_window_.size());
      }

      auto now = std::chrono::steady_clock::now();
      float yaw_acc_gimbal = 0.0f;
      float pitch_acc_gimbal = 0.0f;
      if (gs_initialized_) {
        auto dt = std::chrono::duration<float>(now - last_gs_time_).count();
        if (dt > 0.001f) {
          yaw_acc_gimbal = (gs.yaw_vel - last_gs_yaw_vel_) / dt;
          pitch_acc_gimbal = (gs.pitch_vel - last_gs_pitch_vel_) / dt;
        }
      }
      last_gs_yaw_vel_ = gs.yaw_vel;
      last_gs_pitch_vel_ = gs.pitch_vel;
      last_gs_time_ = now;
      gs_initialized_ = true;

      auto msg = autoaim_msgs::msg::Debug{};
      msg.enable_control = plan_result.control;
      msg.fire = plan_result.fire;
      msg.yaw_offest = plan_result.yaw - gs.yaw;
      msg.pitch_offset = plan_result.pitch - gs.pitch;
      msg.yaw = plan_result.yaw;
      msg.pitch = plan_result.pitch;
      msg.yaw_gimbal = gs.yaw;
      msg.pitch_gimbal = gs.pitch;
      msg.fire_rate = fire_rate;
      msg.bullet_speed = gs.bullet_speed;
      msg.yaw_vel = plan_result.yaw_vel;
      msg.pitch_vel = plan_result.pitch_vel;
      msg.yaw_acc = plan_result.yaw_acc;
      msg.pitch_acc = plan_result.pitch_acc;
      msg.yaw_vel_gimbal = gs.yaw_vel;
      msg.pitch_vel_gimbal = gs.pitch_vel;
      msg.yaw_acc_gimbal = yaw_acc_gimbal;
      msg.pitch_acc_gimbal = pitch_acc_gimbal;
      debug_pub_->publish(msg);
    }
    const auto debug_end_time = std::chrono::steady_clock::now();

    // 发布Target状态消息
    const auto target_pub_start_time = std::chrono::steady_clock::now();
    if (target_pub_ && target.has_value()) {
      auto target_msg = autoaim_msgs::msg::Target{};
      std::visit([&target_msg](const auto & t) {
        const auto & ekf = t.ekf();
        target_msg.vx = static_cast<float>(ekf.x[1]);
        target_msg.x = static_cast<float>(ekf.x[0]);
        target_msg.vy = static_cast<float>(ekf.x[3]);
        target_msg.y = static_cast<float>(ekf.x[2]);
        target_msg.vz = static_cast<float>(ekf.x[5]);
        target_msg.z = static_cast<float>(ekf.x[4]);
      }, target.value());
      target_pub_->publish(target_msg);
    }
    const auto target_pub_end_time = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    if (
      plan_result.control && now - last_log_time >
      std::chrono::milliseconds(200)) {
      // utils::logger()->debug(
      //   "[Pipeline] 规划输出: yaw={:.3f} pitch={:.3f} fire={}"
      //   "下位机Gimbal_yaw={:.3f} 下位机Gimbal_pitch={:.3f}",
      //   plan_result.yaw, plan_result.pitch, plan_result.fire,
      //   gs.yaw, gs.pitch);
      last_log_time = now;
    }

    const auto loop_profile_end_time = std::chrono::steady_clock::now();
    const double front_ms = elapsed_ms(front_start_time, front_end_time);
    const double state_ms = elapsed_ms(state_start_time, state_end_time);
    const double plan_ms = elapsed_ms(plan_start_time, plan_end_time);
    const double checkfire_ms = elapsed_ms(checkfire_start_time, checkfire_end_time);
    const double send_ms = elapsed_ms(send_start_time, send_end_time);
    const double debug_ms = elapsed_ms(debug_start_time, debug_end_time);
    const double target_pub_ms = elapsed_ms(target_pub_start_time, target_pub_end_time);
    const double loop_ms = elapsed_ms(loop_start_time, loop_profile_end_time);

    profile_loop_count++;
    update_profile(front_ms, profile_front_sum_ms, profile_front_max_ms);
    update_profile(state_ms, profile_state_sum_ms, profile_state_max_ms);
    update_profile(plan_ms, profile_plan_sum_ms, profile_plan_max_ms);
    update_profile(checkfire_ms, profile_checkfire_sum_ms, profile_checkfire_max_ms);
    update_profile(send_ms, profile_send_sum_ms, profile_send_max_ms);
    update_profile(debug_ms, profile_debug_sum_ms, profile_debug_max_ms);
    update_profile(target_pub_ms, profile_target_pub_sum_ms, profile_target_pub_max_ms);
    update_profile(loop_ms, profile_loop_sum_ms, profile_loop_max_ms);

    const auto profile_now = std::chrono::steady_clock::now();
    if (profile_now - profile_window_start >= std::chrono::seconds(1) && profile_loop_count > 0) {
      const double count = static_cast<double>(profile_loop_count);
      utils::logger()->info(
        "[PlannerProfile] loops={} empty={} "
        "front={:.3f}/{:.3f}ms state={:.3f}/{:.3f}ms plan={:.3f}/{:.3f}ms "
        "checkfire={:.3f}/{:.3f}ms send={:.3f}/{:.3f}ms "
        "debug={:.3f}/{:.3f}ms target_pub={:.3f}/{:.3f}ms loop={:.3f}/{:.3f}ms",
        profile_loop_count,
        profile_empty_count,
        profile_front_sum_ms / count, profile_front_max_ms,
        profile_state_sum_ms / count, profile_state_max_ms,
        profile_plan_sum_ms / count, profile_plan_max_ms,
        profile_checkfire_sum_ms / count, profile_checkfire_max_ms,
        profile_send_sum_ms / count, profile_send_max_ms,
        profile_debug_sum_ms / count, profile_debug_max_ms,
        profile_target_pub_sum_ms / count, profile_target_pub_max_ms,
        profile_loop_sum_ms / count, profile_loop_max_ms);

      profile_window_start = profile_now;
      profile_loop_count = 0;
      profile_empty_count = 0;
      profile_front_sum_ms = 0.0;
      profile_state_sum_ms = 0.0;
      profile_plan_sum_ms = 0.0;
      profile_checkfire_sum_ms = 0.0;
      profile_send_sum_ms = 0.0;
      profile_debug_sum_ms = 0.0;
      profile_target_pub_sum_ms = 0.0;
      profile_loop_sum_ms = 0.0;
      profile_front_max_ms = 0.0;
      profile_state_max_ms = 0.0;
      profile_plan_max_ms = 0.0;
      profile_checkfire_max_ms = 0.0;
      profile_send_max_ms = 0.0;
      profile_debug_max_ms = 0.0;
      profile_target_pub_max_ms = 0.0;
      profile_loop_max_ms = 0.0;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  utils::logger()->info("[Pipeline] 规划线程退出");
}

}  // namespace pipeline

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

  std::string config_path = std::filesystem::current_path().string() + "/src/config/standard4.yaml";
  if (cli.has("@config-path")) {
    config_path = cli.get<std::string>("@config-path");
  }

  rclcpp::init(argc, argv);
  std::signal(SIGINT, Application::handle_signal);

  try {
    Application::PipelineApp app(app_config::AppConfig::load(config_path));
    int ret = app.run();
    rclcpp::shutdown();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[Pipeline] 程序异常终止: {}", e.what());
  }

  rclcpp::shutdown();
  return 1;
}
