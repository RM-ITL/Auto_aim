#include "shooter.hpp"

#include "logger.hpp"
#include "math_tools.hpp"

namespace shooter
{
Shooter::Shooter(const app_config::ShooterConfig & config) : last_command_{0, 0, 0}
{
  // 字段从 SubConfig 注入。tolerance 为 deg，模块 / 57.3 转 rad。
  first_tolerance_ = config.first_tolerance / 57.3;
  second_tolerance_ = config.second_tolerance / 57.3;
  judge_distance_ = config.judge_distance;
  auto_fire_ = config.auto_fire;

  utils::logger()->info("[Shooter] first_tolerance  = {:.3f} deg ({:.6f} rad)", first_tolerance_ * 57.3, first_tolerance_);
  utils::logger()->info("[Shooter] second_tolerance = {:.3f} deg ({:.6f} rad)", second_tolerance_ * 57.3, second_tolerance_);
  utils::logger()->info("[Shooter] judge_distance   = {:.3f}", judge_distance_);
  utils::logger()->info("[Shooter] auto_fire        = {}", auto_fire_);
}

bool Shooter::shoot(
  const io::GimbalCommand & command, const aimer::Aimer & aimer,
  const TargetVariant & target, const Eigen::Vector3d & gimbal_pos)
{
  if (!command.control || !auto_fire_) return false;

  // 使用 std::visit 访问 variant 中的 target
  auto ekf_x = std::visit([](const auto & t) { return t.ekf_x(); }, target);

  auto target_x = ekf_x[0];
  auto target_y = ekf_x[2];
  auto tolerance = std::sqrt(utils::square(target_x) + utils::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;

  // 获取aimer的debug_aim_point来判断瞄准点是否有效
  const auto & aim_point = aimer.debug_aim_point;

  // tools::logger()->debug("d(command.yaw) is {:.4f}", std::abs(last_command_.yaw - command.yaw));
  if (
    std::abs(last_command_.yaw - command.yaw) < tolerance * 2 &&  //此时认为command突变不应该射击
    std::abs(gimbal_pos[0] - last_command_.yaw) < tolerance &&    //应该减去上一次command的yaw值
    aim_point.valid) {
    last_command_ = command;
    return true;
  }

  last_command_ = command;
  return false;
}

bool Shooter::checkfire(
  double cmd_yaw, double cmd_pitch,
  const io::GimbalState & gs,
  const TargetVariant & target)
{
  auto ekf_x = std::visit([](const auto & t) { return t.ekf_x(); }, target);
  double distance = std::sqrt(utils::square(ekf_x[0]) + utils::square(ekf_x[2]));
  double tolerance = distance > judge_distance_ ? second_tolerance_ : first_tolerance_;

  double yaw_offset = cmd_yaw - gs.yaw;
  double pitch_offset = cmd_pitch - gs.pitch;
  double normalized_error =
    (yaw_offset * yaw_offset + pitch_offset * pitch_offset) / (tolerance * tolerance);
  return normalized_error < 1.0;
}

}  // namespace shooter
