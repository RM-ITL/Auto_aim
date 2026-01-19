#include "shooter.hpp"

#include <yaml-cpp/yaml.h>

#include "logger.hpp"
#include "math_tools.hpp"

namespace shooter
{
Shooter::Shooter(const std::string & config_path) : last_command_{0, 0, 0}
{
  auto yaml = YAML::LoadFile(config_path);
  first_tolerance_ = yaml["Shooter"]["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["Shooter"]["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["Shooter"]["judge_distance"].as<double>();
  auto_fire_ = yaml["Shooter"]["auto_fire"].as<bool>();
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

}  // namespace shooter