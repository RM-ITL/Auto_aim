#include "video.hpp"

#include <stdexcept>
#include <utility>

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
  const std::string & video_path,
  std::size_t queue_capacity, std::function<void(void)> queue_full_handler)
: Video(queue_capacity, std::move(queue_full_handler))
{
  if (video_path.empty()) {
    throw std::invalid_argument("Video path must not be empty");
  }

  video_path_ = video_path;
  if (!video_capture_.open(video_path_)) {
    throw std::runtime_error("Failed to open video: " + video_path_);
  }

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
  VideoFrame frame_data;
  frame_queue_.pop(frame_data);

  frame = std::move(frame_data.frame);
  timestamp = frame_data.timestamp;
}

void Video::start_reader()
{
  reader_quit_.store(false);
  reader_thread_ = std::thread(&Video::reader_loop, this);
}

void Video::reader_loop()
{
  while (!reader_quit_.load()) {
    cv::Mat frame;
    if (!video_capture_.read(frame)) {
      break;
    }

    submit(frame, Clock::now());
  }

  frame_queue_.push(VideoFrame{});
}

}  // namespace utils
