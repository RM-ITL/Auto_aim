#include "lower_sentry.hpp"

#include "logger.hpp"
#include "math_tools.hpp"

namespace io
{
Sentry::Sentry(const app_config::SentryConfig & config)
{
  const std::string & com_port = config.com_port;

  utils::logger()->info("[Sentry] com_port              = {}", com_port);
  utils::logger()->info("[Sentry] baud                  = {} (from yaml)", config.baud);

  // q_calib: 与 Gimbal 完全等价（SentryConfig 复用 yaml["Gimbal"] 段）。
  if (config.q_calib) {
    q_calib_ = config.q_calib->normalized();
    utils::logger()->info("[Sentry] q_calib.w             = {:.6f}", q_calib_.w());
    utils::logger()->info("[Sentry] q_calib.x             = {:.6f}", q_calib_.x());
    utils::logger()->info("[Sentry] q_calib.y             = {:.6f}", q_calib_.y());
    utils::logger()->info("[Sentry] q_calib.z             = {:.6f}", q_calib_.z());
  } else {
    // 默认单位四元数（不校正）
    q_calib_ = Eigen::Quaterniond::Identity();
    utils::logger()->warn("[Sentry] q_calib               = identity (missing q_calib)");
  }

  try {
    serial_.setPort(com_port);
    auto timeout = serial::Timeout::simpleTimeout(50);
    serial_.setTimeout(timeout);
    serial_.open();
    serial_.setBaudrate(config.baud);
  } catch (const std::exception & e) {
    utils::logger()->error("[Sentry] Failed to open serial: {}", e.what());
    throw;
  }

  image_imu_offset_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(config.image_imu_offset));
  utils::logger()->info("[Sentry] image_imu_offset      = {:.6f} s", config.image_imu_offset);

  thread_ = std::thread(&Sentry::read_thread, this);
  utils::logger()->info("[Sentry] Sentry initialized.");
}

Sentry::~Sentry()
{
  stop();
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

void Sentry::stop()
{
  quit_ = true;  // read_thread 在 read_serial 超时后退出；q() 不再阻塞，无需唤醒队列。
}


GimbalState Sentry::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}


Eigen::Quaterniond Sentry::q(std::chrono::steady_clock::time_point t)
{
  // 遗留接口：二分插值，失败退到最近样本(而非 identity)。
  auto o = imu_buffer_.interpolate(t);
  return o ? *o : imu_buffer_.newest_or_identity();
}

std::optional<Eigen::Quaterniond> Sentry::q_for_image(std::chrono::steady_clock::time_point image_t)
{
  // 图像<->IMU 偏移校正 + 二分插值；无有效样本返回 nullopt(不注入 identity)，计数 + 限频告警。
  auto o = imu_buffer_.interpolate(image_t - image_imu_offset_);
  if (!o) {
    interp_fail_count_++;
    auto now = std::chrono::steady_clock::now();
    if (now - last_interp_warn_ > std::chrono::seconds(1)) {
      utils::logger()->warn(
        "[Sentry] q_for_image 无有效 IMU 样本，丢帧(累计 {} 次)——检查 IMU 是否迟到/断流",
        interp_fail_count_);
      last_interp_warn_ = now;
    }
  }
  return o;
}


void Sentry::push_quaternion(const Eigen::Quaterniond& q, std::chrono::steady_clock::time_point t)
{
  imu_buffer_.push(q, t);
}

void Sentry::set_receive_callback(ReceiveCallback cb)
{
  std::lock_guard<std::mutex> lock(callback_mutex_);
  on_receive_ = std::move(cb);
}

void Sentry::send(
  uint8_t mode, float yaw, float pitch, float vx, float vy, float w)
{
  tx_data_.mode = mode;
  tx_data_.yaw = yaw;
  tx_data_.pitch = pitch;
  tx_data_.vx = vx;
  tx_data_.vy = vy;
  tx_data_.w = w;

  if (reconnecting_ || !serial_.isOpen()) return;

  std::lock_guard<std::mutex> lock(serial_mutex_);
  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    utils::logger()->warn("[Sentry] Failed to write serial: {}", e.what());
  }
}

int Sentry::read_serial(uint8_t * buffer, size_t size)
{
  std::lock_guard<std::mutex> lock(serial_mutex_);
  try {
    return serial_.read(buffer, size) == size ? 1 : 0;
  } catch (const std::exception & e) {
    return -1;
  }
}

void Sentry::read_thread()
{
  utils::logger()->info("[Sentry] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 100) {
      error_count = 0;
      utils::logger()->warn("[Sentry] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    // 读取帧头
    int ret = read_serial(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head));
    if (ret == -1) {
      error_count++;  // 设备异常（拔掉、断开），累加错误触发重连
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    if (ret == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // 验证帧头为 'G', 'V'
    if (rx_data_.head[0] != 'G' || rx_data_.head[1] != 'V') {
      error_count++;
      continue;
    }

    // 读取除帧头外的剩余数据
    if (read_serial(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head)) != 1) {
      error_count++;
      continue;
    }

    // 验证帧尾为 'G'
    if (rx_data_.tail != 'G') {
      error_count++;
      continue;
    }

    // 数据包有效，清空错误计数
    error_count = 0;

    // 更新云台状态
    {
      std::lock_guard<std::mutex> lock(mutex_);

      state_.yaw = rx_data_.yaw;
      state_.yaw_vel = 0.0f;
      state_.pitch = rx_data_.pitch;
      state_.pitch_vel = 0.0f;
      state_.bullet_speed = rx_data_.bullet_speed;
      state_.bullet_count = rx_data_.bullet_count;

      // 读取四元数 (wxyz顺序) 并校正
      Eigen::Quaterniond q_converted(
          rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);

      Eigen::Quaterniond q_corrected = (q_calib_ * q_converted).normalized();

      // 推入校正后的四元数供 q()/q_for_image() 插值使用
      imu_buffer_.push(q_corrected, std::chrono::steady_clock::now());
    }

    // 调用接收回调 (在mutex_锁外部)
    {
      std::lock_guard<std::mutex> lock(callback_mutex_);
      if (on_receive_) on_receive_(rx_data_);
    }
  }

  utils::logger()->info("[Sentry] read_thread stopped.");
}

void Sentry::reconnect()
{
  reconnecting_ = true;
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    utils::logger()->warn("[Sentry] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      {
        std::lock_guard<std::mutex> lock(serial_mutex_);
        serial_.close();
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      {
        std::lock_guard<std::mutex> lock(serial_mutex_);
        serial_.open();
      }
      utils::logger()->info("[Sentry] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      utils::logger()->warn("[Sentry] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  reconnecting_ = false;
}

}  // namespace io
