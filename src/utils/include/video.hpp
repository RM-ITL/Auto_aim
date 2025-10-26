#ifndef UTILS__VIDEO_HPP
#define UTILS__VIDEO_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "thread_safe_queue.hpp"

namespace utils
{

struct VideoFrame
{
  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;
};

class Video
{
public:
  using Clock = std::chrono::steady_clock;

  explicit Video(
    const std::string & video_path,
    std::size_t queue_capacity = 5,
    std::function<void(void)> queue_full_handler = nullptr);

  explicit Video(
    std::size_t queue_capacity = 5,
    std::function<void(void)> queue_full_handler = nullptr);

  ~Video();

  void submit(const cv::Mat & frame, Clock::time_point timestamp = Clock::now());

  void read(cv::Mat & frame, Clock::time_point & timestamp);

  std::size_t capacity() const noexcept { return queue_capacity_; }

private:
  std::size_t queue_capacity_;
  ThreadSafeQueue<VideoFrame> frame_queue_;
  std::string video_path_;
  cv::VideoCapture video_capture_;
  std::thread reader_thread_;
  std::atomic<bool> reader_quit_{false};
  bool auto_reader_enabled_{false};

  void start_reader();
  void reader_loop();
};

}  // namespace utils

#endif  // UTILS__VIDEO_PROCESSOR_HPP
