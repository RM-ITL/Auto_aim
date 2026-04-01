#ifndef STANDARD3_HPP_
#define STANDARD3_HPP_

#include <atomic>
#include <chrono>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "camera.hpp"
#include "detect_node.hpp"
#include "solver_node.hpp"
#include "thread_safe_queue.hpp"
#include "tracker.hpp"
#include "armor.hpp"
#include "planner.hpp"
#include "gimbal.hpp"
#include "shooter.hpp"

namespace Application
{

class Standard3App
{
public:
  explicit Standard3App(const std::string & config_path);
  ~Standard3App();

  int run();
  void request_stop();

private:
  void planner_loop();

  // 组件与配置
  std::string config_path_;
  std::unique_ptr<camera::Camera> camera_;
  std::unique_ptr<armor_auto_aim::Detector> detector_;
  std::unique_ptr<solver::Solver> solver_;
  solver::YawOptimizer* yaw_optimizer_;
  std::unique_ptr<tracker::Tracker> tracker_;
  std::unique_ptr<plan::Planner> planner_;
  std::unique_ptr<io::Gimbal> gimbal_;
  std::unique_ptr<shooter::Shooter> shooter_;

  tools::ThreadSafeQueue<std::optional<tracker::TargetVariant>, true> target_queue{1};

  std::atomic<bool> quit_{false};
  std::thread planner_thread_;

  std::chrono::steady_clock::time_point start_time_;
  const double bullet_speed_{12.0};
};

}  // namespace Application

#endif  // STANDARD3_HPP_
