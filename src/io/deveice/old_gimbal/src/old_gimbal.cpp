#include "old_gimbal.hpp"

#include <vector>

#include "logger.hpp"
#include "yaml.hpp"

namespace io
{

OldGimbal::OldGimbal(const std::string & config_path)
{
  auto yaml = utils::load(config_path);
  primary_port_ = utils::read<std::string>(yaml, "com_port");
  backup_port_ = utils::read<std::string>(yaml, "com_port_1");

  bool port_opened = false;

  // 首先尝试打开主端口
  try {
    serial_.setPort(primary_port_);
    serial_.setBaudrate(115200);  // 老接口默认波特率
    serial_.setTimeout(serial::Timeout::max(), 1000, 0, 1000, 0);
    serial_.open();
    current_port_ = primary_port_;
    port_opened = true;
    utils::logger()->info("[OldGimbal] Serial port opened: {}", primary_port_);
  } catch (const std::exception & e) {
    utils::logger()->warn("[OldGimbal] Failed to open primary port {}: {}", primary_port_, e.what());

    // 尝试打开备用端口
    try {
      utils::logger()->info("[OldGimbal] Trying backup port: {}", backup_port_);
      serial_.setPort(backup_port_);
      serial_.setBaudrate(115200);
      serial_.setTimeout(serial::Timeout::max(), 1000, 0, 1000, 0);
      serial_.open();
      current_port_ = backup_port_;
      port_opened = true;
      utils::logger()->info("[OldGimbal] Backup serial port opened: {}", backup_port_);
    } catch (const std::exception & backup_error) {
      utils::logger()->error(
        "[OldGimbal] Failed to open backup port {}: {}", backup_port_, backup_error.what());
    }
  }

  if (!port_opened) {
    utils::logger()->error("[OldGimbal] All serial ports failed to open");
    throw std::runtime_error("Failed to open any serial port");
  }

  last_reconnect_attempt_ = std::chrono::steady_clock::now();

  utils::logger()->info(
    "[OldGimbal] Initialized with frame size: {} bytes, angle scale: {}, active port: {}",
    OldVisionToGimbal::FRAME_SIZE,
    OldVisionToGimbal::ANGLE_SCALE,
    current_port_);
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
      "[OldGimbal] Statistics - Sent: {}, Errors: {}, Reconnects: {}, Success rate: {:.1f}%",
      sent_count_.load(), error_count_.load(), reconnect_count_.load(), success_rate);
  }
}

void OldGimbal::send_delta_angle(float delta_yaw_deg, float delta_pitch_deg)
{
  // 如果需要重连，先尝试重连
  if (need_reconnect_.load()) {
    if (try_reconnect()) {
      utils::logger()->info("[OldGimbal] Reconnection successful, resuming data transmission");
    } else {
      return;  // 重连失败，跳过本次发送
    }
  }

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
      need_reconnect_ = true;
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[OldGimbal] Failed to write serial: {}", e.what());
    error_count_++;
    need_reconnect_ = true;
  }
}

void OldGimbal::send(const OldVisionToGimbal & data)
{
  // 如果需要重连，先尝试重连
  if (need_reconnect_.load()) {
    if (try_reconnect()) {
      utils::logger()->info("[OldGimbal] Reconnection successful, resuming data transmission");
    } else {
      return;  // 重连失败，跳过本次发送
    }
  }

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
      need_reconnect_ = true;
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[OldGimbal] Failed to write serial: {}", e.what());
    error_count_++;
    need_reconnect_ = true;
  }
}

void OldGimbal::send(const SimpleVisionToGimbal & data)
{
  // 如果需要重连，先尝试重连
  if (need_reconnect_.load()) {
    if (try_reconnect()) {
      utils::logger()->info("[OldGimbal] Reconnection successful, resuming data transmission");
    } else {
      return;  // 重连失败，跳过本次发送
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  simple_tx_data_ = data;

  try {
    size_t bytes_written = serial_.write(
      reinterpret_cast<const uint8_t*>(&simple_tx_data_),
      SimpleVisionToGimbal::FRAME_SIZE);

    if (bytes_written == SimpleVisionToGimbal::FRAME_SIZE) {
      sent_count_++;
    } else {
      utils::logger()->warn(
        "[OldGimbal] Incomplete write - Expected: {}, Actual: {}",
        SimpleVisionToGimbal::FRAME_SIZE, bytes_written);
      error_count_++;
      need_reconnect_ = true;
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[OldGimbal] Failed to write serial: {}", e.what());
    error_count_++;
    need_reconnect_ = true;
  }
}

void OldGimbal::send_simple(bool control, bool fire, float yaw, float pitch)
{
  // 如果需要重连，先尝试重连
  if (need_reconnect_.load()) {
    if (try_reconnect()) {
      utils::logger()->info("[OldGimbal] Reconnection successful, resuming data transmission");
    } else {
      return;  // 重连失败，跳过本次发送
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);

  simple_tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  simple_tx_data_.yaw = yaw;
  // simple_tx_data_.yaw_vel = yaw_vel;
  simple_tx_data_.pitch = pitch;
  // simple_tx_data_.pitch_vel = pitch_vel;

  try {
    size_t bytes_written = serial_.write(
      reinterpret_cast<const uint8_t*>(&simple_tx_data_),
      SimpleVisionToGimbal::FRAME_SIZE);

    if (bytes_written == SimpleVisionToGimbal::FRAME_SIZE) {
      sent_count_++;

      if (sent_count_ % 100 == 0) {
        utils::logger()->debug(
          "[OldGimbal] Sent {} frames - Latest: Yaw={:.3f}rad, Pitch={:.3f}rad",
          sent_count_.load(), yaw, pitch);
      }
    } else {
      utils::logger()->warn(
        "[OldGimbal] Incomplete write - Expected: {}, Actual: {}",
        SimpleVisionToGimbal::FRAME_SIZE, bytes_written);
      error_count_++;
      need_reconnect_ = true;
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[OldGimbal] Failed to write serial: {}", e.what());
    error_count_++;
    need_reconnect_ = true;
  }
}

bool OldGimbal::try_reconnect()
{
  // 检查重连间隔，避免频繁重连
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
    now - last_reconnect_attempt_).count();

  if (elapsed < RECONNECT_INTERVAL_MS) {
    return false;  // 还没到重连时间
  }

  last_reconnect_attempt_ = now;

  // 尝试关闭旧连接
  try {
    if (serial_.isOpen()) {
      serial_.close();
    }
  } catch (const std::exception & e) {
    utils::logger()->debug("[OldGimbal] Error closing port during reconnect: {}", e.what());
  }

  // 尝试按顺序重连：先尝试主端口，再尝试备用端口
  std::vector<std::string> ports_to_try;

  // 如果当前用的是备用端口，先尝试主端口（可能设备重新插回了）
  if (current_port_ == backup_port_) {
    ports_to_try = {primary_port_, backup_port_};
  } else {
    ports_to_try = {backup_port_, primary_port_};
  }

  for (const auto & port : ports_to_try) {
    try {
      utils::logger()->info("[OldGimbal] Attempting to reconnect to port: {}", port);
      serial_.setPort(port);
      serial_.setBaudrate(115200);
      serial_.setTimeout(serial::Timeout::max(), 1000, 0, 1000, 0);
      serial_.open();

      current_port_ = port;
      reconnect_count_++;
      need_reconnect_ = false;

      utils::logger()->info(
        "[OldGimbal] Reconnected successfully to port: {} (reconnect #{})",
        port, reconnect_count_.load());
      return true;
    } catch (const std::exception & e) {
      utils::logger()->debug("[OldGimbal] Failed to reconnect to {}: {}", port, e.what());
    }
  }

  utils::logger()->warn("[OldGimbal] All reconnect attempts failed");
  return false;
}

}  // namespace io
