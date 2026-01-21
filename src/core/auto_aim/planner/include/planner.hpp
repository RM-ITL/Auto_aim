#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <list>
#include <optional>
#include <variant>

#include "target.hpp"
#include "outpost_target.hpp"
#include "tinympc/tiny_api.hpp"

namespace plan
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel
using TargetVariant = std::variant<predict::Target, predict::OutpostTarget>;

struct Plan
{
  bool control = false;
  bool fire = false;
  float target_yaw = 0.F;
  float target_pitch = 0.F;
  float yaw = 0.F;
  float yaw_vel = 0.F;
  float yaw_acc = 0.F;
  float pitch = 0.F;
  float pitch_vel = 0.F;
  float pitch_acc = 0.F;
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  Planner(const std::string & config_path);

  // 子弹飞行时间补偿
  Plan plan(predict::Target target, double bullet_speed);
  // 系统延迟时间的补偿
  Plan plan(std::optional<predict::Target> target, double bullet_speed);

  // OutpostTarget 接口
  Plan plan(predict::OutpostTarget target, double bullet_speed);

  // 新增 TargetVariant 接口
  Plan plan(TargetVariant target, double bullet_speed);
  Plan plan(std::optional<TargetVariant> target, double bullet_speed);

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;  // 系统延迟时间的补偿

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  Eigen::Matrix<double, 2, 1> aim(const predict::Target & target, double bullet_speed);
  // 轨迹生成过程中的连续预测
  Trajectory get_trajectory(predict::Target & target, double yaw0, double bullet_speed);

  // OutpostTarget 版本
  Eigen::Matrix<double, 2, 1> aim(const predict::OutpostTarget & target, double bullet_speed);
  Trajectory get_trajectory(predict::OutpostTarget & target, double yaw0, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
