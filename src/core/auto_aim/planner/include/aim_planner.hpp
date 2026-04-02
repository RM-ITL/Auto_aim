#ifndef AUTO_AIM__AIM_PLANNER_HPP
#define AUTO_AIM__AIM_PLANNER_HPP

#include <Eigen/Dense>
#include <optional>
#include <variant>

#include "planner.hpp"  // 复用 Plan, DT, HORIZON 等定义

namespace plan
{

class AimPlanner
{
public:
  Eigen::Vector4d debug_xyza;
  AimPlanner(const std::string & config_path);

  // 子弹飞行时间补偿
  Plan plan(predict::Target target, double bullet_speed);
  // 系统延迟时间的补偿
  Plan plan(std::optional<predict::Target> target, double bullet_speed);
  // variant 重载（兼容 TargetVariant）
  Plan plan(std::optional<std::variant<predict::Target, predict::OutpostTarget>> target, double bullet_speed);

private:
  // 基础参数
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double armor_hysteresis_;  // 装甲板选择滞后系数

  // 高速模式参数
  double omega_threshold_;   // 角速度阈值，超过此值进入高速模式
  double window_angle_;      // 射击窗口半角（弧度）

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  // 工作模式枚举
  enum class ArmorMode { LOW_SPEED, HIGH_SPEED };

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  // 低速模式方法（原有逻辑）
  Eigen::Matrix<double, 2, 1> aim(const predict::Target & target, double bullet_speed);
  Trajectory get_trajectory(predict::Target & target, double yaw0, double bullet_speed);

  // 高速模式方法（新增）
  ArmorMode get_armor_mode(double omega) const;
  int select_armor_high_speed(const predict::Target & target) const;
  double compute_yaw_high_speed(const predict::Target & target) const;
  double compute_pitch_high_speed(const predict::Target & target, int armor_id, double bullet_speed) const;
  bool should_fire_high_speed(const predict::Target & target, int armor_id) const;

  // 辅助方法
  double compute_facing_angle(const Eigen::Vector4d & xyza, const Eigen::Vector3d & xyz) const;
};

}  // namespace plan

#endif  // AUTO_AIM__AIM_PLANNER_HPP
