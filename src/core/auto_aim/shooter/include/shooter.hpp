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

private:
  io::GimbalCommand last_command_;
  double judge_distance_;
  double first_tolerance_;
  double second_tolerance_;
  bool auto_fire_;
};
}  // namespace shooter

#endif  // AUTO_AIM__SHOOTER_HPP