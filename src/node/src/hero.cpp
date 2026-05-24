#include "standard3.hpp"

#include <csignal>
#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include "app_config/app_config.hpp"
#include "logger.hpp"

namespace Application
{
namespace
{
std::atomic<bool> g_stop_requested{false};

void handle_signal(int)
{
  g_stop_requested.store(true);
}

void log_app_config(const std::string & prefix, const app_config::AppConfig & app_config)
{
  utils::logger()->info("[{}] config.absolute_path = {}", prefix, app_config.source_path);
  utils::logger()->info("[{}] config.file_size     = {} bytes", prefix, app_config.source_file_size);
  utils::logger()->info(
    "[{}] config.last_write_time_raw = {}", prefix, app_config.source_last_write_raw);
}

}  // namespace

Standard3App::Standard3App(const app_config::AppConfig & app_config)
: start_time_(std::chrono::steady_clock::now())
{
  utils::logger()->info("[Standard3] 正在初始化，配置文件: {}", app_config.source_path);
  log_app_config("Hero", app_config);

  camera_ = std::make_unique<camera::Camera>(app_config.camera);
  utils::logger()->info("[Standard3] 相机初始化完成");

  detector_ = std::make_unique<armor_auto_aim::Detector>(app_config.detector);
  solver_ = std::make_unique<solver::Solver>(app_config.solver);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(app_config.tracker, *solver_);
  planner_ = std::make_unique<plan::Planner>(app_config.planner);

  gimbal_ = std::make_unique<io::Gimbal>(app_config.gimbal);
  utils::logger()->info("[Standard3] 云台串口初始化完成");

  shooter_ = std::make_unique<shooter::Shooter>(app_config.shooter);
  utils::logger()->info("[Standard3] Shooter初始化完成");

  utils::logger()->info("[Standard3] 所有模块初始化完成，准备进入主循环");
}

Standard3App::~Standard3App()
{
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
  quit_.store(true);
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

  std::string config_path = std::filesystem::current_path().string() + "/src/config/hero.yaml";
  if (cli.has("@config-path")) {
    config_path = cli.get<std::string>("@config-path");
  }

  std::signal(SIGINT, Application::handle_signal);

  try {
    const auto app_config = app_config::AppConfig::load(config_path);
    Application::Standard3App app(app_config);
    int ret = app.run();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[Standard3] 程序异常终止: {}", e.what());
  }

  return 1;
}
