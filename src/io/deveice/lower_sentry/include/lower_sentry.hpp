#ifndef IO__LOWERSENTRY_HPP
#define IO__LOWERSENTRY_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>

#include "app_config/app_config.hpp"
#include "imu_snapshot_buffer.hpp"
#include "serial/serial.h"
#include "thread_safe_queue.hpp"
#include "gimbal.hpp"

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
  uint8_t die_flag;
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

class Sentry
{
public:
  Sentry(const app_config::SentryConfig & config);

  ~Sentry();

  /// 停止 sentry，唤醒阻塞在 q() 中的线程
  void stop();

  GimbalState state() const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  /// 用图像时间戳查询曝光时刻姿态：内部按 (t - image_imu_offset_) 二分插值。
  /// 无有效样本时返回 std::nullopt —— 调用方应丢弃该帧。
  std::optional<Eigen::Quaterniond> q_for_image(std::chrono::steady_clock::time_point image_t);

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
  // 姿态插值缓冲：非破坏性二分查找 + 小间隙 clamp，取代旧的破坏性队列 + identity 兜底。
  tools::ImuSnapshotBuffer imu_buffer_{1000, 0.010};

  // IMU外参标定四元数：q_corrected = q_calib_ * q_lower
  Eigen::Quaterniond q_calib_;

  // 图像<->IMU 相对时间偏移 Δt，q_for_image 查询时从图像戳里减去。
  std::chrono::steady_clock::duration image_imu_offset_{std::chrono::steady_clock::duration::zero()};

  // q_for_image 插值失败(丢帧)统计与限频告警。
  uint64_t interp_fail_count_{0};
  std::chrono::steady_clock::time_point last_interp_warn_{};

  // 返回值: 1=成功, 0=超时无数据, -1=设备异常
  int read_serial(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();
};

}  // namespace io

#endif  // IO__LOWERSENTRY_HPP
