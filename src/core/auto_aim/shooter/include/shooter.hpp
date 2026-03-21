#ifndef AUTO_AIM__SHOOTER_HPP
#define AUTO_AIM__SHOOTER_HPP

#include <string>
#include <variant>
#include <Eigen/Dense>

#include "gimbal.hpp"
#include "target.hpp"
#include "outpost_target.hpp"
#include "aimer.hpp"

namespace shooter
{
using TargetVariant = std::variant<predict::Target, predict::OutpostTarget>;

class Shooter
{
public:
  Shooter(const std::string & config_path);

  bool shoot(
    const io::GimbalCommand & command, const aimer::Aimer & aimer,
    const TargetVariant & target, const Eigen::Vector3d & gimbal_pos);

  /// 检查伺服跟踪误差是否在允许范围内（normalized error 椭圆判断）
  bool checkServoReady(
    double cmd_yaw, double cmd_pitch,
    const io::GimbalState & gs,
    const TargetVariant & target);

private:
  io::GimbalCommand last_command_;
  double judge_distance_;
  double first_tolerance_;
  double second_tolerance_;
  bool auto_fire_;
};
}  // namespace shooter

#endif  // AUTO_AIM__SHOOTER_HPP