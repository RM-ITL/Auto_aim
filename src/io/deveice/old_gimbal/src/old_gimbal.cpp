#include "old_gimbal.hpp"

#include "logger.hpp"
#include "yaml.hpp"

namespace io
{

OldGimbal::OldGimbal(const std::string & config_path)
{
  auto yaml = utils::load(config_path);
  auto com_port = utils::read<std::string>(yaml, "com_port");

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(115200);  // 老接口默认波特率
    serial_.setTimeout(serial::Timeout::max(), 1000, 0, 1000, 0);
    serial_.open();
    utils::logger()->info("[OldGimbal] Serial port opened: {}", com_port);
  } catch (const std::exception & e) {
    utils::logger()->error("[OldGimbal] Failed to open serial port {}: {}", com_port, e.what());
    throw;
  }

  utils::logger()->info(
    "[OldGimbal] Initialized with frame size: {} bytes, angle scale: {}",
    OldVisionToGimbal::FRAME_SIZE,
    OldVisionToGimbal::ANGLE_SCALE);
}

OldGimbal::~OldGimbal()
{
  try {
    if (serial_.isOpen()) {
      serial_.close();
      utils::logger()->info("[OldGimbal] Serial port closed");
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[OldGimbal] Error closing serial port: {}", e.what());
  }

  if (sent_count_ > 0 || error_count_ > 0) {
    double success_rate = (sent_count_ + error_count_) > 0
      ? (static_cast<double>(sent_count_) / (sent_count_ + error_count_)) * 100.0
      : 0.0;
    utils::logger()->info(
      "[OldGimbal] Statistics - Sent: {}, Errors: {}, Success rate: {:.1f}%",
      sent_count_.load(), error_count_.load(), success_rate);
  }
}

void OldGimbal::send_delta_angle(float delta_yaw_deg, float delta_pitch_deg)
{
  // 数据验证
  if (std::isnan(delta_yaw_deg) || std::isnan(delta_pitch_deg)) {
    utils::logger()->warn(
      "[OldGimbal] Invalid angle data - Yaw: {}, Pitch: {}",
      delta_yaw_deg, delta_pitch_deg);
    error_count_++;
    return;
  }

  // 范围检查
  if (std::abs(delta_yaw_deg) > 180.0f || std::abs(delta_pitch_deg) > 180.0f) {
    utils::logger()->warn(
      "[OldGimbal] Angle out of range [-180°, 180°] - Yaw: {:.2f}°, Pitch: {:.2f}°",
      delta_yaw_deg, delta_pitch_deg);
    error_count_++;
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // 设置角度数据
  tx_data_.set_delta_angle(delta_yaw_deg, delta_pitch_deg);

  try {
    // 发送12字节数据
    size_t bytes_written = serial_.write(
      reinterpret_cast<const uint8_t*>(&tx_data_),
      OldVisionToGimbal::FRAME_SIZE);

    if (bytes_written == OldVisionToGimbal::FRAME_SIZE) {
      sent_count_++;

      // 调试日志（每100帧输出一次）
      if (sent_count_ % 100 == 0) {
        utils::logger()->debug(
          "[OldGimbal] Sent {} frames - Latest: Yaw={:.3f}°, Pitch={:.3f}°",
          sent_count_.load(), delta_yaw_deg, delta_pitch_deg);
      }
    } else {
      utils::logger()->warn(
        "[OldGimbal] Incomplete write - Expected: {}, Actual: {}",
        OldVisionToGimbal::FRAME_SIZE, bytes_written);
      error_count_++;
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[OldGimbal] Failed to write serial: {}", e.what());
    error_count_++;
  }
}

void OldGimbal::send(const OldVisionToGimbal & data)
{
  std::lock_guard<std::mutex> lock(mutex_);
  tx_data_ = data;

  try {
    size_t bytes_written = serial_.write(
      reinterpret_cast<const uint8_t*>(&tx_data_),
      OldVisionToGimbal::FRAME_SIZE);

    if (bytes_written == OldVisionToGimbal::FRAME_SIZE) {
      sent_count_++;
    } else {
      utils::logger()->warn(
        "[OldGimbal] Incomplete write - Expected: {}, Actual: {}",
        OldVisionToGimbal::FRAME_SIZE, bytes_written);
      error_count_++;
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[OldGimbal] Failed to write serial: {}", e.what());
    error_count_++;
  }
}

}  // namespace io
