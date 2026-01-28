#include "light_aimer.hpp"

#include <yaml-cpp/yaml.h>
#include <iostream>

namespace auto_base
{

LightAimer::LightAimer(const std::string & config_path)
: begin_x_(0.0), base_offset_(0.0)
{
  auto yaml = YAML::LoadFile(config_path);

  // 加载基准点x坐标
  if (yaml["LightAimer"]["begin_x"].IsDefined()) {
    begin_x_ = yaml["LightAimer"]["begin_x"].as<double>();
  } else {
    std::cerr << "[LightAimer] Warning: begin_x not defined in config, using default 0.0"
              << std::endl;
  }

  // 加载基础补偿
  if (yaml["LightAimer"]["base_offset"].IsDefined()) {
    base_offset_ = yaml["LightAimer"]["base_offset"].as<double>();
  } else {
    std::cerr << "[LightAimer] Warning: base_offset not defined in config, using default 0.0"
              << std::endl;
  }

  // 加载根据number的补偿表
  if (yaml["LightAimer"]["offsets"].IsDefined()) {
    auto offsets_node = yaml["LightAimer"]["offsets"];
    for (auto it = offsets_node.begin(); it != offsets_node.end(); ++it) {
      int number = std::stoi(it->first.as<std::string>());
      double offset = it->second.as<double>();
      offset_map_[number] = offset;
    }
  } else {
    std::cerr << "[LightAimer] Warning: offsets not defined in config" << std::endl;
  }

  std::cout << "[LightAimer] Initialized with begin_x=" << begin_x_
            << ", base_offset=" << base_offset_ << ", offset_map size=" << offset_map_.size()
            << std::endl;
}

double LightAimer::aim(
  LightTarget* target,
  const io::DartToVision & dart_data)
{
  if (!target) {
    std::cerr << "[LightAimer] Error: target is null" << std::endl;
    return 0.0;
  }

  // 获取目标的中心点x坐标
  Eigen::VectorXd ekf_x = target->ekf_x();
  double current_center_x = ekf_x(0);

  // 根据number字段查表获取对应的offset
  double number_offset = 0.0;
  if (offset_map_.find(dart_data.number) != offset_map_.end()) {
    number_offset = offset_map_[dart_data.number];
  } else {
    std::cerr << "[LightAimer] Warning: number " << static_cast<int>(dart_data.number)
              << " not found in offset_map, using 0.0" << std::endl;
  }

  double total_offset = base_offset_ + number_offset;

  // 计算yaw_error
  double yaw_error = calculate_yaw_error(begin_x_, current_center_x, total_offset);

  return yaw_error;
}

double LightAimer::calculate_yaw_error(
  double begin_x,
  double current_center_x,
  double offset)
{
  // 公式：yaw_error = -(begin_x - current_center_x - offset)
  return -(begin_x - current_center_x - offset);
}

}  // namespace auto_base
