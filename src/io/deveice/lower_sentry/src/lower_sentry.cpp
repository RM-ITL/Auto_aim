#include "lower_sentry.hpp"

#include "logger.hpp"
#include "math_tools.hpp"
#include "yaml.hpp"

namespace io
{
Sentry::Sentry(const std::string & config_path)
{
  auto yaml = utils::load(config_path);
  auto com_port = utils::read<std::string>(yaml, "com_port");

  // 读取IMU外参标定四元数（注意：q_calib是顶级节点，在Sentry同级）
  auto q_calib_node = yaml["q_calib"];
  if (q_calib_node) {
    double qx = q_calib_node["x"].as<double>();
    double qy = q_calib_node["y"].as<double>();
    double qz = q_calib_node["z"].as<double>();
    double qw = q_calib_node["w"].as<double>();
    q_calib_ = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
    utils::logger()->info("[Sentry] Loaded q_calib: w={:.6f}, x={:.6f}, y={:.6f}, z={:.6f}",
                          q_calib_.w(), q_calib_.x(), q_calib_.y(), q_calib_.z());
  } else {
    // 默认单位四元数（不校正）
    q_calib_ = Eigen::Quaterniond::Identity();
    utils::logger()->warn("[Sentry] q_calib not found, using identity quaternion.");
  }

  try {
    serial_.setPort(com_port);
    serial_.open();
    serial_.setBaudrate(115200);
  } catch (const std::exception & e) {
    utils::logger()->error("[Sentry] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Sentry::read_thread, this);
  utils::logger()->info("[Sentry] Sentry initialized.");
}

Sentry::~Sentry()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}


GimbalState Sentry::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}


Eigen::Quaterniond Sentry::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = utils::delta_time(t_a, t_b);
    auto t_ac = utils::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}


void Sentry::push_quaternion(const Eigen::Quaterniond& q, std::chrono::steady_clock::time_point t)
{
  queue_.push({q, t});
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

bool Sentry::read_serial(uint8_t * buffer, size_t size)
{
  std::lock_guard<std::mutex> lock(serial_mutex_);
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    return false;
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
    if (!read_serial(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // 验证帧头为 'G', 'V'
    if (rx_data_.head[0] != 'G' || rx_data_.head[1] != 'V') {
      error_count++;
      continue;
    }

    // 读取除帧头外的剩余数据
    if (!read_serial(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
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
      state_.pitch = rx_data_.pitch;
      state_.bullet_speed = rx_data_.bullet_speed;
      state_.bullet_count = rx_data_.bullet_count;

      // 读取四元数 (wxyz顺序) 并校正
      Eigen::Quaterniond q_converted(
          rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);

      Eigen::Quaterniond q_corrected = (q_calib_ * q_converted).normalized();

      // 推入校正后的四元数供 q() 方法使用
      queue_.push({q_corrected, std::chrono::steady_clock::now()});
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
