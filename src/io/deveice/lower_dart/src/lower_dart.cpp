#include "lower_dart.hpp"

#include "logger.hpp"
#include "math_tools.hpp"
#include "yaml.hpp"

namespace io
{
Dart::Dart(const std::string & config_path)
{
  auto yaml = utils::load(config_path);
  auto com_port = utils::read<std::string>(yaml, "com_port");

  try {
    serial_.setPort(com_port);
    serial_.open();
    serial_.setBaudrate(115200);
  } catch (const std::exception & e) {
    utils::logger()->error("[Dart] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Dart::read_thread, this);
  utils::logger()->info("[Dart] Dart initialized.");
}

Dart::~Dart()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

DartMode Dart::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

DartState Dart::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Dart::str(DartMode mode) const
{
  switch (mode) {
    case DartMode::IDLE:
      return "IDLE";
    case DartMode::AUTO_AIM:
      return "AUTO_AIM";
    case DartMode::RECORD:
      return "RECORD";
    default:
      return "INVALID";
  }
}



// void Dart::push_quaternion(const Eigen::Quaterniond& q, std::chrono::steady_clock::time_point t)
// {
//   queue_.push({q, t});
// }

// size_t Dart::queue_size() const
// {
//   return queue_.size();  // ThreadSafeQueue需要有size()方法
// }


void Dart::send(float yaw_error, int target_status)
{
  tx_data_.yaw_error = yaw_error;
  tx_data_.target_status = target_status;


  if (reconnecting_ || !serial_.isOpen()) return;  // 重连中或串口未打开时直接返回

  std::lock_guard<std::mutex> lock(serial_mutex_);  // 加锁保护串口写
  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    utils::logger()->warn("[Dart] Failed to write serial: {}", e.what());
  }
}

bool Dart::read_serial(uint8_t * buffer, size_t size)
{
  std::lock_guard<std::mutex> lock(serial_mutex_);  // 加锁保护串口读
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    return false;
  }
}

void Dart::read_thread()
{
  utils::logger()->info("[Dart] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 100) {  // 降低阈值，只有真正的错误才累加
      error_count = 0;
      utils::logger()->warn("[Dart] Too many errors, attempting to reconnect...");
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
    if (rx_data_.head[0] != 'D' || rx_data_.head[1] != 'V') {
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
    if (rx_data_.tail != 'D') {
      error_count++;  // 帧尾校验失败，累加错误
      // 由于 packed 结构体，需要先复制到临时变量
    //   float yaw_tmp = rx_data_.yaw;
    //   float pitch_tmp = rx_data_.pitch;
      // utils::logger()->debug(
      //   "[Dart] Frame tail check failed. Expected 'G'(0x47), got 0x{:02X}. "
      //   "Received data size: {} bytes. Data: mode={}, yaw={:.3f}, pitch={:.3f}",
      //   static_cast<unsigned char>(rx_data_.tail), sizeof(rx_data_),
      //   rx_data_.mode, yaw_tmp, pitch_tmp);
      continue;
    }

    // 数据包有效，清空错误计数
    error_count = 0;

    // 获取当前时间戳
    auto now = std::chrono::steady_clock::now();

    // 更新飞镖状态
    {
      std::lock_guard<std::mutex> lock(mutex_);

      state_.number_count = rx_data_.number;
      state_.dune_status = rx_data_.dune;
      state_.mode = rx_data_.mode;

      // 更新最新数据和时间戳
      latest_rx_data_ = rx_data_;
      latest_timestamp_ = now;

      // 添加到时间戳缓存
      timestamp_cache_.push_back({now, rx_data_});

      // 定期清理过时数据（每100次成功读取清理一次）
      static int push_count = 0;
      if (++push_count % 100 == 0) {
        clean_old_data();
      }

      switch (rx_data_.mode) {
      case 0:
        mode_ = DartMode::IDLE;
        break;
      case 1:
        mode_ = DartMode::AUTO_AIM;
        break;
      case 2:
        mode_ = DartMode::RECORD;
        break;
      default:
        mode_ = DartMode::IDLE;
        utils::logger()->warn("[Dart] Invalid mode: {}", rx_data_.mode);
        break;
      }
    }
  }

  utils::logger()->info("[Dart] read_thread stopped.");
}

void Dart::reconnect()
{
  reconnecting_ = true;  // 设置重连标志
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    utils::logger()->warn("[Dart] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
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
      utils::logger()->info("[Dart] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      utils::logger()->warn("[Dart] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  reconnecting_ = false;  // 清除重连标志
}

DartToVision Dart::get_nearest_state(std::chrono::steady_clock::time_point t)
{
  std::lock_guard<std::mutex> lock(mutex_);

  // 如果缓存为空，返回最新数据
  if (timestamp_cache_.empty()) {
    return latest_rx_data_;
  }

  // 在缓存中查找距离时间戳 t 最近的数据
  auto min_diff = std::chrono::steady_clock::duration::max();
  DartToVision result = latest_rx_data_;

  for (const auto& [ts, data] : timestamp_cache_) {
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::abs(ts - t));
    if (diff < min_diff) {
      min_diff = diff;
      result = data;
    }
  }

  return result;
}

void Dart::clean_old_data()
{
  auto now = std::chrono::steady_clock::now();
  auto threshold = std::chrono::seconds(5);  // 5秒过期时间

  // 移除超过5秒的旧数据
  while (!timestamp_cache_.empty() &&
         (now - timestamp_cache_.front().first) > threshold) {
    timestamp_cache_.pop_front();
  }

  // 如果缓存超过最大大小，移除最旧的元素
  while (timestamp_cache_.size() > CACHE_SIZE) {
    timestamp_cache_.pop_front();
  }
}

}  // namespace io