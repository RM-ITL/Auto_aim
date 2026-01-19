#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <optional>
#include <variant>


#include "target.hpp"
#include "outpost_target.hpp"
#include "armor.hpp"
#include "gimbal.hpp"

namespace aimer
{

using ArmorType = armor_auto_aim::ArmorType;
using ArmorName = armor_auto_aim::ArmorName;
using TargetVariant = std::variant<predict::Target, predict::OutpostTarget>;


struct AimPoint
{
  bool valid;
  Eigen::Vector4d xyza;
};

class Aimer
{
public:
  AimPoint debug_aim_point;
  explicit Aimer(const std::string & config_path);

  // 新接口：支持 TargetVariant
  io::GimbalCommand aim(
    std::optional<TargetVariant> target, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  // 保留旧接口用于兼容
  io::GimbalCommand aim(
    std::list<predict::Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  // io::GimbalCommand aim(
  //   std::list<predict::Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  //   io::ShootMode shoot_mode, bool to_now = true);

private:
  double yaw_offset_;
  std::optional<double> left_yaw_offset_, right_yaw_offset_;
  double pitch_offset_;
  double comming_angle_;
  double leaving_angle_;
  double lock_id_ = -1;
  double high_speed_delay_time_;
  double low_speed_delay_time_;
  double decision_speed_;

  AimPoint choose_aim_point(const predict::Target & target);
  AimPoint choose_aim_point(const predict::OutpostTarget & target);

  // 内部辅助函数，用于处理单个 target
  io::GimbalCommand aim_single_target(
    const predict::Target & target, std::chrono::steady_clock::time_point timestamp,
    double bullet_speed, bool to_now);
  io::GimbalCommand aim_single_target(
    const predict::OutpostTarget & target, std::chrono::steady_clock::time_point timestamp,
    double bullet_speed, bool to_now);
};

}  // namespace aimer

#endif  // AUTO_AIM__AIMER_HPP
