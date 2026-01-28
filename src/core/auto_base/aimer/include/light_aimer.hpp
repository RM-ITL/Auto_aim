#ifndef AUTO_BASE__LIGHT_AIMER_HPP_
#define AUTO_BASE__LIGHT_AIMER_HPP_

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <string>

#include "light_model.hpp"
#include "lower_dart.hpp"

namespace auto_base
{

class LightAimer
{
public:
  explicit LightAimer(const std::string & config_path);

  double aim(
    LightTarget* target,
    const io::DartToVision & dart_data);

private:
  // 配置参数
  double begin_x_;                    // 基准点x坐标
  double base_offset_;                // 基础补偿
  std::map<int, double> offset_map_;  // number -> offset映射表

  double calculate_yaw_error(
    double begin_x,
    double current_center_x,
    double offset);
};

}  // namespace auto_base

#endif  // AUTO_BASE__LIGHT_AIMER_HPP_
