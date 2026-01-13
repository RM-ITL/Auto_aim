#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include <list>
#include <string>
#include <variant>

#include "armor.hpp"
#include "module/solver.hpp"
#include "solver_node.hpp"
#include "target.hpp"
#include "outpost_target.hpp"
#include "math_tools.hpp"
#include "logger.hpp"

namespace tracker
{

using Armors = armor_auto_aim::Armor;
using TargetVariant = std::variant<predict::Target, predict::OutpostTarget>;

class Tracker
{
public:
  Tracker(const std::string & config_path, solver::Solver & solver);

  std::string state() const;

  // // 获取最近1秒的识别率 (0.0 ~ 1.0)
  // double detection_rate() const;

  std::list<TargetVariant> track(
    std::list<Armors> & armors,
    std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

private:
  solver::Solver & solver_;
  armor_auto_aim::Color enemy_color_;

  // 跟踪状态机参数
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  // 前哨站特化参数
  int outpost_min_detect_count_;
  int outpost_detect_fail_tolerance_;
  int detect_fail_count_;

  // 状态管理
  std::string state_;
  std::string pre_state_;

  // 目标管理 - 使用 variant 支持两种目标类型
  TargetVariant target_;
  bool is_tracking_outpost_ = false;
  std::chrono::steady_clock::time_point last_timestamp_;

  // // 识别率统计 (滑动窗口)
  // std::deque<bool> detection_history_;
  // static constexpr int DETECTION_WINDOW_SIZE = 120;  // 约1秒 (假设30fps)

  // 内部方法
  void state_machine(bool found);
  bool set_target(std::list<Armors> & armors,
                  std::chrono::steady_clock::time_point t);
  bool update_target(std::list<Armors> & armors,
                     std::chrono::steady_clock::time_point t);
};

}  // namespace tracker

#endif // TRACKER_HPP