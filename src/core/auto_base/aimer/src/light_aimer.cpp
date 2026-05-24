#include "light_aimer.hpp"

#include <iostream>

#include "logger.hpp"

namespace auto_base
{

LightAimer::LightAimer(const app_config::LightAimerConfig & config)
: begin_x_(config.begin_x), base_offset_(config.base_offset)
{
  // 字段从 SubConfig 注入。SubConfig 加载阶段已处理 IsDefined 守卫与 map 遍历，
  // 此处仅赋值。原代码缺字段时打 warn——SubConfig 加载阶段不再打 warn（值是默认 0.0/空 map），
  // 但模块构造日志仍会暴露最终值，调试者可对照判断字段是否生效。
  offset_map_ = config.offsets;

  for (const auto & [number, offset] : offset_map_) {
    utils::logger()->info("[LightAimer] offsets[{}] = {:.3f}", number, offset);
  }

  utils::logger()->info("[LightAimer] begin_x         = {:.3f}", begin_x_);
  utils::logger()->info("[LightAimer] base_offset     = {:.3f}", base_offset_);
  utils::logger()->info("[LightAimer] offset_map.size = {}", offset_map_.size());

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
