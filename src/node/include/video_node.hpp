#ifndef VIDEO_NODE_HPP_
#define VIDEO_NODE_HPP_

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "armor.hpp"
#include "buff_detector.hpp"
#include "video.hpp"

namespace Application
{

struct VideoDebugPacket
{
  cv::Mat rgb_image;
  // std::vector<armor_auto_aim::Visualization> reprojected_armors;
  bool valid{false};
};

class VideoApp
{
public:
  explicit VideoApp(const std::string & config_path);
  ~VideoApp();

  int run();
  void request_stop();

private:
  void visualize(const VideoDebugPacket & packet);

  std::string config_path_;

  std::unique_ptr<utils::Video> video_reader_;
  std::unique_ptr<auto_buff::Buff_Detector> detector_;
  // std::unique_ptr<io::DmImu> dm_imu_;
  // std::unique_ptr<solver::Solver> solver_;
  // solver::YawOptimizer * yaw_optimizer_{nullptr};
  // std::unique_ptr<tracker::Tracker> tracker_;

  bool enable_visualization_{true};
  std::string visualization_window_name_{"video_pipeline"};
  cv::Point visualization_center_point_{640, 384};
  std::atomic<int> visualization_frame_counter_{0};

  std::atomic<bool> quit_{false};
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace Application

#endif  // VIDEO_NODE_HPP_
