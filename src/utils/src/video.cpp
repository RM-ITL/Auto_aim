#include "video.hpp"

#include <stdexcept>
#include <utility>

#include "logger.hpp"

namespace utils
{
namespace
{
std::size_t clamp_capacity(std::size_t capacity)
{
  return capacity == 0 ? 1 : capacity;
}

std::function<void(void)> ensure_handler(std::function<void(void)> handler)
{
  if (handler) {
    return handler;
  }
  return []() {};
}
}  // namespace

Video::Video(
  const std::string & config_path,
  std::size_t queue_capacity, std::function<void(void)> queue_full_handler)
: Video(queue_capacity, std::move(queue_full_handler))
{

  auto yaml = YAML::LoadFile(config_path);
  video_path_ = yaml["video_path"].as<std::string>();
  utils::logger()->info("[Video] 尝试打开视频: {}", video_path_);
  if (!video_capture_.open(video_path_)) {
    utils::logger()->error("[Video] 打开视频失败: {}", video_path_);
    throw std::runtime_error("Failed to open video: " + video_path_);
  }
  utils::logger()->info(
    "[Video] 视频打开成功, 分辨率={}x{}, FPS={}",
    video_capture_.get(cv::CAP_PROP_FRAME_WIDTH),
    video_capture_.get(cv::CAP_PROP_FRAME_HEIGHT),
    video_capture_.get(cv::CAP_PROP_FPS));

  auto_reader_enabled_ = true;
  start_reader();
}

Video::Video(
  std::size_t queue_capacity, std::function<void(void)> queue_full_handler)
: queue_capacity_(clamp_capacity(queue_capacity)),
  frame_queue_(queue_capacity_, ensure_handler(std::move(queue_full_handler)))
{
}

Video::~Video()
{
  reader_quit_.store(true);

  if (auto_reader_enabled_) {
    if (reader_thread_.joinable()) {
      reader_thread_.join();
    }
    video_capture_.release();
    frame_queue_.push(VideoFrame{});
  }
}

void Video::submit(const cv::Mat & frame, Clock::time_point timestamp)
{
  frame_queue_.push(VideoFrame{frame.clone(), timestamp});
}

void Video::read(cv::Mat & frame, Clock::time_point & timestamp)
{
  utils::logger()->debug("[Video] 等待获取下一帧");
  VideoFrame frame_data;
  frame_queue_.pop(frame_data);

  frame = std::move(frame_data.frame);
  timestamp = frame_data.timestamp;

  if (frame.empty()) {
    utils::logger()->info("[Video] 获取到终止标记帧");
  }
}

void Video::start_reader()
{
  reader_quit_.store(false);
  reader_thread_ = std::thread(&Video::reader_loop, this);
}

void Video::reader_loop()
{
  utils::logger()->info("[Video] 读取线程启动");
  std::size_t frame_counter = 0;
  while (!reader_quit_.load()) {
    cv::Mat frame;
    if (!video_capture_.read(frame)) {
      utils::logger()->warn("[Video] 读取到空帧或视频结束, frame_counter={}", frame_counter);
      break;
    }

    ++frame_counter;
    if (frame_counter <= 5 || frame_counter % 100 == 0) {
      utils::logger()->debug("[Video] 读取到第{}帧, size={}x{}", frame_counter, frame.cols, frame.rows);
    }

    submit(frame, Clock::now());
  }

  utils::logger()->info("[Video] 读取线程结束, 推送终止标记");
  frame_queue_.push(VideoFrame{});
}

}  // namespace utils
