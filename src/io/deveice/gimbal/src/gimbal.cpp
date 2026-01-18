#include "gimbal.hpp"

#include "logger.hpp"
#include "math_tools.hpp"
#include "yaml.hpp"

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = utils::load(config_path);
  auto com_port = utils::read<std::string>(yaml, "com_port");

  // 读取IMU外参标定四元数（注意：q_calib是顶级节点，在Gimbal同级）
  auto q_calib_node = yaml["q_calib"];
  if (q_calib_node) {
    double qx = q_calib_node["x"].as<double>();
    double qy = q_calib_node["y"].as<double>();
    double qz = q_calib_node["z"].as<double>();
    double qw = q_calib_node["w"].as<double>();
    q_calib_ = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
    utils::logger()->info("[Gimbal] Loaded q_calib: w={:.6f}, x={:.6f}, y={:.6f}, z={:.6f}",
                          q_calib_.w(), q_calib_.x(), q_calib_.y(), q_calib_.z());
  } else {
    // 默认单位四元数（不校正）
    q_calib_ = Eigen::Quaterniond::Identity();
    utils::logger()->warn("[Gimbal] q_calib not found, using identity quaternion.");
  }

  try {
    serial_.setPort(com_port);
    serial_.open();
    serial_.setBaudrate(115200);
  } catch (const std::exception & e) {
    utils::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);
  utils::logger()->info("[Gimbal] Gimbal initialized.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
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


void Gimbal::push_quaternion(const Eigen::Quaterniond& q, std::chrono::steady_clock::time_point t)
{
  queue_.push({q, t});
}

// size_t Gimbal::queue_size() const
// {
//   return queue_.size();  // ThreadSafeQueue需要有size()方法
// }

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;

  if (reconnecting_ || !serial_.isOpen()) return;  // 重连中或串口未打开时直接返回

  std::lock_guard<std::mutex> lock(serial_mutex_);  // 加锁保护串口写
  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    utils::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;

  if (reconnecting_ || !serial_.isOpen()) return;  // 重连中或串口未打开时直接返回

  std::lock_guard<std::mutex> lock(serial_mutex_);  // 加锁保护串口写
  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    utils::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read_serial(uint8_t * buffer, size_t size)
{
  std::lock_guard<std::mutex> lock(serial_mutex_);  // 加锁保护串口读
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    return false;
  }
}

void Gimbal::read_thread()
{
  utils::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 100) {  // 降低阈值，只有真正的错误才累加
      error_count = 0;
      utils::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    // 读取帧头
    if (!read_serial(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) {
      // 读不到数据是正常的（电控未发送），不累加错误
      std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 等待数据
      continue;
    }

    // 验证帧头为 'G', 'V'
    if (rx_data_.head[0] != 'G' || rx_data_.head[1] != 'V') {
      error_count++;  // 收到错误数据，累加错误
      continue;
    }

    // 读取除帧头外的剩余数据
    if (!read_serial(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++;  // 帧头对了但读不到后续数据，累加错误
      continue;
    }

    // 验证帧尾为 'G'
    if (rx_data_.tail != 'G') {
      error_count++;  // 帧尾校验失败，累加错误
      // 由于 packed 结构体，需要先复制到临时变量
      float yaw_tmp = rx_data_.yaw;
      float pitch_tmp = rx_data_.pitch;
      // utils::logger()->debug(
      //   "[Gimbal] Frame tail check failed. Expected 'G'(0x47), got 0x{:02X}. "
      //   "Received data size: {} bytes. Data: mode={}, yaw={:.3f}, pitch={:.3f}",
      //   static_cast<unsigned char>(rx_data_.tail), sizeof(rx_data_),
      //   rx_data_.mode, yaw_tmp, pitch_tmp);
      continue;
    }

    // 数据包有效，清空错误计数
    error_count = 0;

    // 更新云台状态
    std::lock_guard<std::mutex> lock(mutex_);

    state_.yaw = rx_data_.yaw;
    state_.yaw_vel = rx_data_.yaw_vel;
    state_.pitch = rx_data_.pitch;
    state_.pitch_vel = rx_data_.pitch_vel;
    state_.bullet_speed = rx_data_.bullet_speed;
    state_.bullet_count = rx_data_.bullet_count;

    // 读取四元数 (wxyz顺序)
    float q0 = rx_data_.q[0];  // w
    float q1 = rx_data_.q[1];  // x
    float q2 = rx_data_.q[2];  // y
    float q3 = rx_data_.q[3];  // z

    Eigen::Quaterniond q_converted(
        q0,
        q1,
        q2,
        q3
    );

    // 使用标定四元数校正下位机IMU四元数: q_corrected = q_lower * q_calib^(-1)
    Eigen::Quaterniond q_corrected = (q_calib_ * q_converted).normalized();

    // 推入校正后的四元数供 q() 方法使用
    queue_.push({q_corrected, std::chrono::steady_clock::now()});

    // utils::logger()->debug(
    //     "[Gimbal] 原始: w:{:.3f}, x:{:.3f}, y:{:.3f}, z:{:.3f} -> 校正后: w:{:.3f}, x:{:.3f}, y:{:.3f}, z:{:.3f}",
    //     q0, q1, q2, q3,
    //     q_corrected.w(), q_corrected.x(), q_corrected.y(), q_corrected.z());
    

    switch (rx_data_.mode) {
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      default:
        mode_ = GimbalMode::IDLE;
        utils::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
        break;
    }
  }

  utils::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  reconnecting_ = true;  // 设置重连标志
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    utils::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
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
      utils::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      utils::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  reconnecting_ = false;  // 清除重连标志
}

}  // namespace Gimbal