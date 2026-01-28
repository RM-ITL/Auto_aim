#ifndef IO__DART_SIMULATOR_HPP
#define IO__DART_SIMULATOR_HPP

#include <chrono>
#include <atomic>
#include <cstdint>

#include "lower_dart.hpp"

namespace io
{

class DartSimulator
{
public:
  DartSimulator();
  ~DartSimulator() = default;

  DartToVision get_nearest_state(std::chrono::steady_clock::time_point t);

  void send(float yaw_error, int target_status);
  void set_dart_number(uint8_t number)
  {
    if (number >= 1 && number <= 4) {
      dart_number_ = number;
    }
  }
  void set_dart_mode(uint8_t mode)
  {
    if (mode <= 2) {
      dart_mode_ = mode;
    }
  }

  void set_dune_status(uint8_t status)
  {
    if (status <= 2) {
      dune_status_ = status;
    }
  }



  int get_send_count() const { return send_count_; }


  float get_last_yaw_error() const { return last_yaw_error_; }


  int get_last_target_status() const { return last_target_status_; }


  void reset_stats()
  {
    send_count_ = 0;
    last_yaw_error_ = 0.0f;
    last_target_status_ = 0;
  }

private:
  // 模拟参数
  uint8_t dart_number_ = 1;      // 飞镖号（1-4）
  uint8_t dart_mode_ = 1;        // 模式（0: 不开自瞄, 1: 开自瞄, 2: 录像）
  uint8_t dune_status_ = 0;      // 舱门状态（0: 展开, 1: 关闭, 2: 进行中）

  // 统计信息
  std::atomic<int> send_count_{0};
  float last_yaw_error_ = 0.0f;
  int last_target_status_ = 0;
};

}  // namespace io

#endif  // IO__DART_SIMULATOR_HPP
