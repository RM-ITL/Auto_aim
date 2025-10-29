#ifndef IO_OLD_GIMBAL_HPP
#define IO_OLD_GIMBAL_HPP

#include <atomic>
#include <cmath>
#include <mutex>
#include <string>

#include "serial/serial.h"

namespace io
{

// 老接口数据结构（12字节，单向发送）
struct __attribute__((packed)) OldVisionToGimbal
{
  static constexpr size_t FRAME_SIZE = 12;
  static constexpr float ANGLE_SCALE = 1000.0f;
  static constexpr uint8_t FRAME_HEADER = 0xFF;
  static constexpr uint8_t FRAME_TAIL = 0xFE;
  static constexpr uint8_t CONTROL_MODE_AUTOAIM = 0x08;

  uint8_t head;           // [0] 帧头 0xFF
  int16_t pitch;          // [1-2] Pitch数据（小端序，度*1000）
  int16_t yaw;            // [3-4] Yaw数据（小端序，度*1000）
  uint8_t sign_flag;      // [5] 符号位 0x00-0x04
  uint8_t distance;       // [6] 距离（当前为0）
  uint8_t vx;             // [7] 导航Vx（当前为0）
  uint8_t vy;             // [8] 导航Vy（当前为0）
  uint8_t wz;             // [9] 导航Wz（当前为0）
  uint8_t control_mode;   // [10] 控制模式 0x08
  uint8_t tail;           // [11] 帧尾 0xFE

  // 符号位枚举
  enum SignFlag : uint8_t {
    SIGN_ALL_NEGATIVE = 0x00,    // pitch和yaw都为负
    SIGN_YAW_POSITIVE = 0x01,    // yaw为正，pitch为负
    SIGN_PITCH_POSITIVE = 0x02,  // pitch为正，yaw为负
    SIGN_ALL_POSITIVE = 0x03,    // 都为正
    SIGN_NONE = 0x04             // 无效/零值
  };

  // 默认构造函数，初始化为零帧
  OldVisionToGimbal()
  : head(FRAME_HEADER),
    pitch(0),
    yaw(0),
    sign_flag(SIGN_NONE),
    distance(0),
    vx(0),
    vy(0),
    wz(0),
    control_mode(CONTROL_MODE_AUTOAIM),
    tail(FRAME_TAIL)
  {}

  // 设置pitch和yaw增量（单位：度）
  void set_delta_angle(float delta_yaw_deg, float delta_pitch_deg)
  {
    // 转换为绝对值的整数表示
    pitch = static_cast<int16_t>(std::round(std::abs(delta_pitch_deg) * ANGLE_SCALE));
    yaw = static_cast<int16_t>(std::round(std::abs(delta_yaw_deg) * ANGLE_SCALE));

    // 设置符号位
    if (delta_yaw_deg < 0 && delta_pitch_deg < 0) {
      sign_flag = SIGN_ALL_NEGATIVE;
    } else if (delta_yaw_deg > 0 && delta_pitch_deg < 0) {
      sign_flag = SIGN_YAW_POSITIVE;
    } else if (delta_yaw_deg < 0 && delta_pitch_deg > 0) {
      sign_flag = SIGN_PITCH_POSITIVE;
    } else if (delta_yaw_deg > 0 && delta_pitch_deg > 0) {
      sign_flag = SIGN_ALL_POSITIVE;
    } else {
      sign_flag = SIGN_NONE;
    }
  }
};

static_assert(sizeof(OldVisionToGimbal) == 12, "OldVisionToGimbal must be 12 bytes");

// 简化版老云台类（只发送，不接收）
class OldGimbal
{
public:
  OldGimbal(const std::string & config_path);
  ~OldGimbal();

  // 发送角度增量（单位：度）
  void send_delta_angle(float delta_yaw_deg, float delta_pitch_deg);

  // 发送原始数据包
  void send(const OldVisionToGimbal & data);

private:
  serial::Serial serial_;
  mutable std::mutex mutex_;
  OldVisionToGimbal tx_data_;

  std::atomic<uint64_t> sent_count_{0};
  std::atomic<uint64_t> error_count_{0};
};

}  // namespace io

#endif  // IO_OLD_GIMBAL_HPP
