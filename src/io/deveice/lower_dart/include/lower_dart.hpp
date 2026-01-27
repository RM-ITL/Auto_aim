#ifndef IO__LOWERDART_HPP
#define IO__LOWERDART_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "thread_safe_queue.hpp"

namespace io
{


struct __attribute__ ((packed)) DartToVision
{
  uint8_t head[2] = {'D', 'V'};
  uint8_t mode;   // 0：不开自瞄，不录像 1：开自瞄且录像 2：录像
  uint8_t status;
  uint8_t number; // 第number号飞镖
  uint8_t dune; //0：完全展开 1：完全关闭 2：正在进行时
  uint8_t tail = 'D';
};


static_assert(sizeof(DartToVision) <= 64);

struct __attribute__ ((packed)) VisionToDart
{
  uint8_t head[2] = {'V', 'D'};
  float yaw_error;
  uint8_t target_status;  // 0：Lost, 1:found
  uint8_t tail = 'V';
};

static_assert(sizeof(VisionToDart) <= 64);

enum class DartMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  RECORD
};

struct DartState
{
  uint8_t mode;
  uint16_t number_count;   // 累计发射的飞镖计数
  uint8_t dune_status;  // 舱门开启情况
};

class Dart
{
public:
  Dart(const std::string & config_path);

  ~Dart();

  DartMode mode() const;
  DartState state() const;
  std::string str(DartMode mode) const;

  void send(float yaw_error, int target_status);

  // 根据时间戳查询最近邻的飞镖状态
  DartToVision get_nearest_state(std::chrono::steady_clock::time_point t);

private:
  serial::Serial serial_;
  std::mutex serial_mutex_;  // 保护串口读写操作
  std::atomic<bool> reconnecting_ = false;  // 重连标志

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  DartToVision rx_data_;
  VisionToDart tx_data_;

  DartMode mode_ ;
  DartState state_;

  // 维护飞镖状态历史队列，用于时间戳查询
  tools::ThreadSafeQueue<std::tuple<DartToVision, std::chrono::steady_clock::time_point>>
    rx_queue_{1000};

  // 辅助：维护最近接收的数据和时间戳（用于快速查询）
  DartToVision latest_rx_data_;
  std::chrono::steady_clock::time_point latest_timestamp_;

  // 辅助：维护一个有序的时间戳缓存（用于近邻查询）
  std::deque<std::pair<std::chrono::steady_clock::time_point, DartToVision>> timestamp_cache_;
  static constexpr size_t CACHE_SIZE = 100;  // 缓存最近100条记录

  bool read_serial(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();
  void clean_old_data();  // 清理过时数据
};

}  // namespace io

#endif  // IO__LOWERDART_HPP
