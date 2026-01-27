#include "test_node_deep.hpp"

#include <csignal>
#include <algorithm>
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
  ros_node_ = std::make_shared<rclcpp::Node>("pipeline_debug_node");
  debug_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Debug>(
    "debug", rclcpp::QoS(10));
  orientation_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Orienta>(
    "orientation", rclcpp::QoS(10));
  target_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Outpost>(
    "target",rclcpp::QoS(10)
  );

  camera_ = std::make_unique<camera::Camera>(config_path_);
  dm_imu_ = std::make_unique<io::DmImu>(config_path_);
  detector_ = std::make_unique<armor_auto_aim::Detector>(config_path_);
  solver_ = std::make_unique<solver::Solver>(config_path_);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(config_path_, *solver_);
  planner_ = std::make_unique<plan::Planner>(config_path_);
  // guard_planner_ = std::make_unique<guard::GuardPlanner>(config_path_); // 守株待兔模式的planner
  gimbal_ = std::make_unique<io::Gimbal>(config_path_);

  // enable_visualization_ = detector_->config().enable_visualization;
  // visualization_center_point_ = detector_->config().center_point;
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
    Eigen::Quaterniond dm_orientation{Eigen::Quaterniond::Identity()};

    camera_->read(img, timestamp);
    auto t0 = timestamp;  // 图像采集时间戳

    DebugPacket debug_packet;
    timestamp_sec = utils::delta_time(timestamp, start_time_);

    cv::cvtColor(img, debug_packet.rgb_image, cv::COLOR_BGR2RGB);

    dm_orientation = dm_imu_->imu_at(timestamp);
    orientation = gimbal_->q(timestamp);
    utils::logger()->debug(
      "[Pipeline] DM_IMU四元数: w={:.6f}, x={:.6f}, y={:.6f}, z={:.6f}",
      dm_orientation.w(), dm_orientation.x(), dm_orientation.y(), dm_orientation.z());
    utils::logger()->debug(
    "[Pipeline] 下位机的四元数: w={:.6f}, x={:.6f}, y={:.6f}, z={:.6f}",
    orientation.w(), orientation.x(), orientation.y(), orientation.z());

    if (orientation_pub_) {
      auto msg = autoaim_msgs::msg::Orienta{};
      msg.w = orientation.w(),
      msg.x = orientation.x(),
      msg.y = orientation.y(),
      msg.z = orientation.z(),
      msg.dm_w = dm_orientation.w(),
      msg.dm_x = dm_orientation.x(),
      msg.dm_y = dm_orientation.y(),
      msg.dm_z = dm_orientation.z(),      
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

      // 【前哨站估计诊断】输出三个装甲板的估计位置
      if (target_name == armor_auto_aim::ArmorName::outpost && armor_xyza_list.size() == 3) {
        const auto & ekf_x = std::visit([](const auto & t) { return t.ekf_x(); }, target);
        double h1 = ekf_x[9];
        double h2 = ekf_x[10];
        double omega = ekf_x[7];

        // utils::logger()->info(
        //   "[前哨站估计] h1={:.3f}, h2={:.3f}, ω={:.3f} | "
        //   "A0:[{:.2f},{:.2f},{:.2f}] A1:[{:.2f},{:.2f},{:.2f}] A2:[{:.2f},{:.2f},{:.2f}]",
        //   h1, h2, omega,
        //   armor_xyza_list[0][0], armor_xyza_list[0][1], armor_xyza_list[0][2],
        //   armor_xyza_list[1][0], armor_xyza_list[1][1], armor_xyza_list[1][2],
        //   armor_xyza_list[2][0], armor_xyza_list[2][1], armor_xyza_list[2][2]
        // );
      }

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
    //auto plan_result = guard_planner_->plan(target, bullet_speed_);
    auto gs = gimbal_->state();
    bool enable_shoot = std::abs(plan_result.yaw - gs.yaw) < 0.02 
                    && std::abs(plan_result.pitch - gs.pitch) < 0.015;
    if(enable_shoot) {
        plan_result.fire = true;
    }
    if (plan_result.control) {
      gimbal_->send(
        plan_result.control, plan_result.fire, plan_result.yaw, plan_result.yaw_vel,
        plan_result.yaw_acc, plan_result.pitch, plan_result.pitch_vel, plan_result.pitch_acc);
        // gimbal_->send_simple(plan_result.control, plan_result.fire, plan_result.yaw, plan_result.pitch);
    } else {
      gimbal_->send(false, false, 0, 0, 0, 0, 0, 0);
    }

    // 验证通讯帧率

    // {
    //   static bool timers_initialized = false;
    //   static std::chrono::steady_clock::time_point last_send_time;
    //   static std::chrono::steady_clock::time_point window_start_time;
    //   static int send_count = 0;

    //   const auto send_time = std::chrono::steady_clock::now();

    //   if (!timers_initialized) {
    //     timers_initialized = true;
    //     last_send_time = send_time;
    //     window_start_time = send_time;
    //     send_count = 1;
    //   } else {
    //     auto dt_us =
    //       std::chrono::duration_cast<std::chrono::microseconds>(send_time - last_send_time);
    //     utils::logger()->debug(
    //       "[Pipeline] gimbal send dt = {:.3f} ms", dt_us.count() / 1000.0);
    //     last_send_time = send_time;
    //     send_count++;
    //   }

    //   auto window_elapsed = send_time - window_start_time;
    //   if (window_elapsed >= std::chrono::seconds(1)) {
    //     const double elapsed_sec = std::chrono::duration<double>(window_elapsed).count();
    //     const double freq_hz = elapsed_sec > 0.0 ? send_count / elapsed_sec : 0.0;
    //     utils::logger()->debug("[Pipeline] gimbal send freq = {:.1f} Hz", freq_hz);
    //     window_start_time = send_time;
    //     send_count = 0;
    //   }
    // }
      
    if (debug_pub_) {
      auto msg = autoaim_msgs::msg::Debug{};
      msg.enable_control = plan_result.control;
      msg.fire = plan_result.fire;
      msg.yaw_offest = plan_result.yaw - gs.yaw;
      msg.target_pitch = plan_result.target_pitch;
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
        // target_msg.w = static_cast<float>(ekf.x[7]);
        // target_msg.r = static_cast<float>(ekf.x[8]);
      }, target.value());
      target_pub_->publish(target_msg);
    }

    auto now = std::chrono::steady_clock::now();
    if (
      plan_result.control && now - last_log_time >
      std::chrono::milliseconds(200)) {
      auto  yaw_offest = plan_result.target_yaw - gs.yaw;
      // utils::logger()->debug(
      //   "[Pipeline] 规划输出: yaw={:.3f} pitch={:.3f} fire={}"
      //   "下位机Gimbal_yaw={:.3f} 下位机Gimbal_pitch={:.3f}",
      //   plan_result.yaw, plan_result.pitch, plan_result.fire,
      //   gs.yaw, gs.pitch);
      last_log_time = now;
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

  std::string config_path = "/home/guo/ITL_Auto_aim/src/config/config.yaml";
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
