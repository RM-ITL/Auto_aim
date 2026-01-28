#include "dart_simulator.hpp"

#include "logger.hpp"

namespace io
{

DartSimulator::DartSimulator()
{
  utils::logger()->info("[DartSimulator] 初始化模拟器");
  utils::logger()->info(
    "[DartSimulator] 初始配置: number={}, mode={}, dune={}",
    static_cast<int>(dart_number_),
    static_cast<int>(dart_mode_),
    static_cast<int>(dune_status_));
}

DartToVision DartSimulator::get_nearest_state(std::chrono::steady_clock::time_point t)
{
  // 构造模拟的下位机数据
  DartToVision data;
  data.head[0] = 'D';
  data.head[1] = 'V';
  data.mode = dart_mode_;
  data.status = 0;
  data.number = dart_number_;
  data.dune = dune_status_;
  data.tail = 'D';

  return data;
}

void DartSimulator::send(float yaw_error, int target_status)
{
  send_count_++;
  last_yaw_error_ = yaw_error;
  last_target_status_ = target_status;

  // 每帧都记录调试信息
  utils::logger()->debug(
    "[DartSimulator] 发送数据 #{}: yaw_error={:.3f}, target_status={}",
    send_count_, yaw_error, target_status);

  // 每100帧打印一次统计信息
  if (send_count_ % 100 == 0) {
    utils::logger()->info(
      "[DartSimulator] 统计信息 - 已发送{}条命令, 最后yaw_error={:.3f}, "
      "target_status={}",
      send_count_, yaw_error, target_status);
  }
}

}  // namespace io
