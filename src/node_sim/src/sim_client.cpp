// sim_client.cpp — 见 sim_client.hpp。低层文件 IO 复用仿真官方示例的安全约定：
//   目录属本 UID / 0700、独占 client.lock、O_NOFOLLOW、先写 .tmp 再原子 rename、严格大小校验。
#include "sim_client.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <regex>
#include <stdexcept>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

namespace sim
{
namespace
{
void read_all(int fd, void * output, size_t bytes)
{
  auto * data = static_cast<char *>(output);
  while (bytes) {
    const auto count = ::read(fd, data, bytes);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::runtime_error("Incomplete file read");
    data += count;
    bytes -= static_cast<size_t>(count);
  }
}
}  // namespace

Eigen::Quaterniond quaternion_from(const Json & gimbal)
{
  if (!gimbal.contains("quaternion_wxyz")) return Eigen::Quaterniond::Identity();
  const auto q = gimbal.at("quaternion_wxyz").get<std::vector<double>>();
  if (q.size() != 4) return Eigen::Quaterniond::Identity();
  return Eigen::Quaterniond(q[0], q[1], q[2], q[3]).normalized();  // wxyz
}

SimClient::SimClient(const std::string & channel)
{
  if (!std::regex_match(channel, std::regex("[A-Za-z0-9_-]{1,80}")))
    throw std::runtime_error("Invalid channel");
  const auto path = "/dev/shm/rm_aim_sim_" + channel;
  directory_ = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory_ < 0) throw std::runtime_error("Start the simulator first; cannot open " + path);
  struct stat info{};
  if (::fstat(directory_, &info) != 0 || info.st_uid != ::getuid() || (info.st_mode & 0777) != 0700) {
    ::close(directory_);
    throw std::runtime_error("IPC directory must belong to this UID with mode 0700");
  }
  lock_ = ::openat(directory_, "client.lock", O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (lock_ < 0 || ::flock(lock_, LOCK_EX | LOCK_NB) != 0) {
    if (lock_ >= 0) ::close(lock_);
    ::close(directory_);
    throw std::runtime_error("Cannot lock client.lock; another client may own this channel");
  }
  token_ = std::to_string(::getpid()) + "-" + std::to_string(Clock::now().time_since_epoch().count());
}

SimClient::~SimClient()
{
  try {
    send(false, false, 0.0, 0.0);
  } catch (...) {
  }
  if (lock_ >= 0) ::close(lock_);
  if (directory_ >= 0) ::close(directory_);
}

double SimClient::num(const Json & node, const char * key)
{
  if (node.is_object() && node.contains(key) && node.at(key).is_number())
    return node.at(key).get<double>();
  return 0.0;
}

Eigen::Quaterniond SimClient::quaternion() const
{
  return state_.contains("gimbal") ? quaternion_from(state_.at("gimbal"))
                                   : Eigen::Quaterniond::Identity();
}

std::string SimClient::robot() const
{
  if (state_.contains("appearance") && state_["appearance"].contains("robot"))
    return state_["appearance"]["robot"].get<std::string>();
  return "";
}

std::string SimClient::team() const
{
  if (state_.contains("appearance") && state_["appearance"].contains("team"))
    return state_["appearance"]["team"].get<std::string>();
  return "";
}

bool SimClient::paused() const { return state_.value("paused", false); }

bool SimClient::enabled() const
{
  return state_.contains("ipc") && state_["ipc"].value("enabled", false);
}

bool SimClient::fresh() const { return has_state() && Clock::now() - last_state_ <= std::chrono::milliseconds(200); }

// 上下文指纹：publisher/session/外观/暂停/启用/相机标定变化即视为新上下文，需重握手。
Json SimClient::context(const Json & packet) const
{
  const auto & camera = packet.at("camera");
  return Json::array(
    {packet.at("publisher_id"), packet.at("session_id"), packet.at("appearance"),
     packet.at("paused"), packet.at("ipc").at("enabled"), camera.at("width"), camera.at("height"),
     camera.at("K"), camera.at("distortion"), camera.at("R_camera2gimbal"),
     camera.at("t_camera2gimbal")});
}

bool SimClient::poll_state()
{
  auto value = read_json("state.json");
  if (value.value("protocol", "") != "rm_aim_sim/2" || value.value("type", "") != "state")
    return false;
  const double wall =
    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  const double age = wall - value.value("wall_unix_sec", 0.0);
  if (!std::isfinite(age) || age < -0.1 || age > 0.25) return false;
  if (!state_.empty() && value.at("publisher_id") == state_.at("publisher_id") &&
      value.at("publication") == state_.at("publication"))
    return false;
  if (state_.empty() || context(value) != context(state_)) request_.clear();
  state_ = std::move(value);
  last_state_ = Clock::now();
  return true;
}

void SimClient::send(
  bool control, bool fire, double yaw, double pitch, double yaw_vel, double pitch_vel,
  double yaw_acc, double pitch_acc)
{
  if (state_.empty()) return;
  for (double number : {yaw, pitch, yaw_vel, pitch_vel, yaw_acc, pitch_acc})
    if (!std::isfinite(number) || std::abs(number) > 1000)
      throw std::runtime_error("Command contains invalid angle/feedforward");
  if (sequence_session_ != state_.at("session_id")) {
    sequence_session_ = state_.at("session_id").get<std::string>();
    sequence_ = 0;
  }
  sequence_ = std::max(sequence_, state_.at("ipc").at("last_seq").get<std::int64_t>() + 1);
  const bool live = control && fresh() && !state_.at("paused").get<bool>() &&
                    state_.at("ipc").at("enabled").get<bool>();
  write_json(
    "command.json",
    {{"protocol", "rm_aim_sim/2"}, {"type", "command"}, {"publisher_id", state_.at("publisher_id")},
     {"session_id", state_.at("session_id")}, {"seq", sequence_++}, {"control", live},
     {"fire", live && fire}, {"yaw", yaw}, {"pitch", pitch}, {"yaw_vel", yaw_vel},
     {"pitch_vel", pitch_vel}, {"yaw_acc", yaw_acc}, {"pitch_acc", pitch_acc}});
}

void SimClient::request_frame()
{
  if (!request_.empty() || !fresh()) return;
  request_ = token_ + "-" + std::to_string(request_count_++);
  requested_ = Clock::now();
  write_json("request.json", {{"publisher_id", state_.at("publisher_id")}, {"request_id", request_}});
}

bool SimClient::try_frame(Sample & out)
{
  if (request_.empty()) return false;
  if (Clock::now() - requested_ > std::chrono::seconds(5))
    throw std::runtime_error("Frame timeout: use graphical mode, not --headless");
  auto frame = read_json("frame.json");
  if (frame.value("request_id", "") != request_) return false;
  if (!fresh() || frame.value("protocol", "") != "rm_aim_sim/2" ||
      frame.value("type", "") != "frame" || context(frame) != context(state_)) {
    request_.clear();
    return false;
  }
  const double age = state_.at("sim_time").get<double>() - frame.at("sim_time").get<double>();
  if (!std::isfinite(age) || age > 0.2 || age < -0.1) {
    request_.clear();
    return false;
  }
  const int width = frame.at("camera").at("width").get<int>();
  const int height = frame.at("camera").at("height").get<int>();
  const auto encoding = frame.at("encoding").get<std::string>();
  const int channels = encoding == "rgb8" ? 3 : encoding == "rgba8" ? 4 : 0;
  const std::string filename = "frame-" + request_ + ".raw";
  if (!channels || width < 160 || width > 3840 || height < 120 || height > 2160 ||
      frame.at("frame_file") != filename)
    throw std::runtime_error("Invalid frame layout");
  const size_t bytes = static_cast<size_t>(width) * height * channels;
  if (frame.at("image_bytes").get<size_t>() != bytes)
    throw std::runtime_error("Invalid image_bytes");
  int fd = ::openat(directory_, filename.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) throw std::runtime_error("Frame disappeared; reconnect after server restart");
  struct stat info{};
  cv::Mat rgb;
  try {
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size != static_cast<off_t>(bytes))
      throw std::runtime_error("Frame file size/type mismatch");
    rgb.create(height, width, CV_MAKETYPE(CV_8U, channels));
    read_all(fd, rgb.data, bytes);
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
  cv::cvtColor(rgb, out.bgr, channels == 3 ? cv::COLOR_RGB2BGR : cv::COLOR_RGBA2BGR);
  out.metadata = std::move(frame);
  out.received = Clock::now();
  request_.clear();  // 自有 BGR 存储已就绪，下一请求即确认本帧已复制。
  return true;
}

Json SimClient::read_json(const char * filename) const
{
  int fd = ::openat(directory_, filename, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return Json::object();
  struct stat info{};
  std::string data;
  try {
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
        info.st_size > 65536)
      throw std::runtime_error("Invalid JSON file size/type");
    data.resize(static_cast<size_t>(info.st_size));
    read_all(fd, data.data(), data.size());
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
  auto value = Json::parse(data, nullptr, false);
  return value.is_object() ? value : Json::object();
}

void SimClient::write_json(const char * filename, const Json & value) const
{
  const std::string temporary = std::string(filename) + ".client.tmp";
  int fd =
    ::openat(directory_, temporary.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) throw std::runtime_error("Cannot create command/request temporary file");
  try {
    const auto data = value.dump();
    size_t written = 0;
    while (written < data.size()) {
      const auto count = ::write(fd, data.data() + written, data.size() - written);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) throw std::runtime_error("Incomplete file write");
      written += static_cast<size_t>(count);
    }
  } catch (...) {
    ::close(fd);
    ::unlinkat(directory_, temporary.c_str(), 0);
    throw;
  }
  ::close(fd);
  if (::renameat(directory_, temporary.c_str(), directory_, filename) != 0)
    throw std::runtime_error("Cannot atomically publish command/request");
}

}  // namespace sim


