#ifndef IO__LOWERSENTRY_HPP
#define IO__LOWERSENTRY_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "thread_safe_queue.hpp"

namespace io
{



struct __attribute__((packed)) lowerToSentry
{
  uint8_t head[2] = {'G', 'V'};
  float q[4];    // wxyz顺序
  float yaw;
  float yaw_odom;
  float pitch;
  float bullet_speed;
  uint8_t sentry_nav;   // 标志位1，初始导航标志位
  uint8_t low_health;   // 标志位2，回家补血判断
  uint8_t resupply_done;// 标志位3，补给完毕
  uint16_t bullet_count;  // 子弹累计发送次数
  uint8_t tail = 'G';
};


static_assert(sizeof(lowerToSentry) <= 64);


struct __attribute__((packed)) SentryTolower
{
  uint8_t head[2] = {'V', 'G'};
  uint8_t mode;  // 0:不控制静止, 1: 控制云台但不开火，2: 控制云台且开火，4：扫描
  float yaw;
  float pitch;
  float vx;
  float vy;
  float w;
  uint8_t tail= 'V';
};


static_assert(sizeof(SentryTolower) <= 64);

struct GimbalState
{
  float yaw;
  float pitch;
  float bullet_speed;
  uint16_t bullet_count;
};

class Sentry
{
public:
  Sentry(const std::string & config_path);

  ~Sentry();

  GimbalState state() const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  void send(uint8_t mode, float yaw, float pitch, float vx, float vy, float w);

  void push_quaternion(const Eigen::Quaterniond& q, std::chrono::steady_clock::time_point t);

  using ReceiveCallback = std::function<void(const lowerToSentry&)>;
  void set_receive_callback(ReceiveCallback cb);

private:
  serial::Serial serial_;
  std::mutex serial_mutex_;  // 保护串口读写操作
  std::mutex callback_mutex_;
  ReceiveCallback on_receive_;
  std::atomic<bool> reconnecting_ = false;  // 重连标志

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  lowerToSentry rx_data_;
  SentryTolower tx_data_;

  GimbalState state_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  // IMU外参标定四元数：q_corrected = q_calib_ * q_lower
  Eigen::Quaterniond q_calib_;

  bool read_serial(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();
};

}  // namespace io

#endif  // IO__LOWERSENTRY_HPP
