#include "standard3.hpp"

#include <csignal>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hpp"

namespace Application
{
namespace
{
std::atomic<bool> g_stop_requested{false};
Standard3App* g_app_instance{nullptr};

void handle_signal(int)
{
  g_stop_requested.store(true);
  if (g_app_instance) {
    g_app_instance->request_stop();
  }
}

}  // namespace

Standard3App::Standard3App(const std::string & config_path)
: config_path_(config_path),
  start_time_(std::chrono::steady_clock::now())
{
  utils::logger()->info("[Standard3] 正在初始化，配置文件: {}", config_path_);

  camera_ = std::make_unique<camera::Camera>(config_path_);
  utils::logger()->info("[Standard3] 相机初始化完成");

  detector_ = std::make_unique<armor_auto_aim::Detector>(config_path_);
  solver_ = std::make_unique<solver::Solver>(config_path_);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(config_path_, *solver_);
  planner_ = std::make_unique<plan::Planner>(config_path_);

  gimbal_ = std::make_unique<io::Gimbal>(config_path_);
  utils::logger()->info("[Standard3] 云台串口初始化完成");

  shooter_ = std::make_unique<shooter::Shooter>(config_path_);
  utils::logger()->info("[Standard3] Shooter初始化完成");

  utils::logger()->info("[Standard3] 所有模块初始化完成，准备进入主循环");
  g_app_instance = this;
}

Standard3App::~Standard3App()
{
  g_app_instance = nullptr;
  request_stop();
  if (planner_thread_.joinable()) {
    planner_thread_.join();
  }
}

int Standard3App::run()
{
  utils::logger()->info("[Standard3] 主循环启动");
  quit_.store(false);
  if (planner_) {
    planner_thread_ = std::thread(&Standard3App::planner_loop, this);
  }

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

    timestamp_sec = utils::delta_time(timestamp, start_time_);

    cv::Mat rgb_image;
    cv::cvtColor(img, rgb_image, cv::COLOR_BGR2RGB);

    orientation = gimbal_->q(timestamp);
    solver_->updateIMU(orientation, timestamp_sec);

    auto armor = detector_->detect(rgb_image);

    std::list<armor_auto_aim::Armor> armor_list(armor.begin(), armor.end());
    auto targets = tracker_->track(armor_list, timestamp);

    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    std::string current_state = tracker_->state();
    if (current_state != last_state) {
      utils::logger()->info(
        "[Standard3] Tracker状态切换: {} -> {}", last_state, current_state);
      last_state = current_state;
    }
  }

  request_stop();
  if (planner_thread_.joinable()) {
    planner_thread_.join();
  }
  utils::logger()->info("[Standard3] 程序正常退出");
  return 0;
}

void Standard3App::request_stop()
{
  if (!quit_.exchange(true)) {
    // 关闭队列，唤醒阻塞在 pop/front 上的线程
    target_queue.shutdown();

    // 停止 gimbal，唤醒阻塞在 q() 中的主线程
    if (gimbal_) {
      gimbal_->stop();
    }

    // 关闭相机，使 camera_->read() 不再阻塞
    if (camera_) {
      camera_->stop();
    }

    utils::logger()->info("[Standard3] 正在停止...");
  }
}

void Standard3App::planner_loop()
{
  utils::logger()->info("[Standard3] 规划线程启动");

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

    auto gs = gimbal_->state();
    auto plan_result = planner_->plan(target, gs.bullet_speed);


    if (target.has_value()) {
      bool enable_shoot = shooter_->checkfire(
        plan_result.yaw, plan_result.pitch, gs, target.value());
      plan_result.fire = plan_result.fire && enable_shoot;
    }
    if (plan_result.control) {
      gimbal_->send(
        plan_result.control, plan_result.fire, plan_result.yaw, plan_result.yaw_vel,
        plan_result.yaw_acc, plan_result.pitch, plan_result.pitch_vel, plan_result.pitch_acc);
    } else {
      gimbal_->send(false, false, gs.yaw, 0, 0, gs.pitch, 0, 0);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  utils::logger()->info("[Standard3] 规划线程退出");
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

  std::string config_path = std::filesystem::current_path().string() + "/src/config/sentry.yaml";
  if (cli.has("@config-path")) {
    config_path = cli.get<std::string>("@config-path");
  }

  std::signal(SIGINT, Application::handle_signal);

  try {
    Application::Standard3App app(config_path);
    int ret = app.run();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[Standard3] 程序异常终止: {}", e.what());
  }

  return 1;
}
