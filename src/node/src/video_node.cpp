#include "video_node.hpp"

#include <algorithm>
#include <csignal>
#include <exception>
#include <list>
#include <stdexcept>
#include <utility>

#include <Eigen/Geometry>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hpp"
#include "math_tools.hpp"
#include "target.hpp"

constexpr char kDefaultVideoPath[] = "/home/guo/ITL_sentry_auto_new/test2.mp4";
constexpr char kDefaultConfigPath[] =
  "/home/guo/ITL_sentry_auto_new/src/config/coord_converter.yaml";

namespace Application
{
namespace
{
std::atomic<bool> g_stop_requested{false};

void handle_signal(int)
{
  g_stop_requested.store(true);
}
}  // namespace

VideoApp::VideoApp(const std::string & config_path, const std::string & video_path)
: config_path_(config_path),
  video_path_(video_path),
  start_time_(std::chrono::steady_clock::now())
{
  if (video_path_.empty()) {
    throw std::invalid_argument("视频路径不能为空");
  }

  video_reader_ = std::make_unique<utils::Video>(video_path_);
  dm_imu_ = std::make_unique<io::DmImu>(config_path_);
  detector_ = std::make_unique<armor_auto_aim::Detector>(config_path_);
  solver_ = std::make_unique<solver::Solver>(config_path_);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(config_path_, *solver_);

  enable_visualization_ = detector_->config().enable_visualization;
  visualization_center_point_ = detector_->config().center_point;
  utils::logger()->info(
    "[VideoPipeline] 初始化完成, config: {}, video: {}", config_path_, video_path_);
}

VideoApp::~VideoApp()
{
  request_stop();
}

int VideoApp::run()
{
  if (enable_visualization_) {
    try {
      cv::namedWindow(visualization_window_name_, cv::WINDOW_NORMAL);
    } catch (const cv::Exception & e) {
      utils::logger()->warn("[VideoPipeline] 创建窗口失败: {}", e.what());
      enable_visualization_ = false;
    }
  }

  while (!quit_.load()) {
    if (g_stop_requested.load()) {
      break;
    }

    cv::Mat frame;
    utils::Video::Clock::time_point timestamp;
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
    video_reader_->read(frame, timestamp);

    orientation = dm_imu_->imu_at(timestamp);

    if (frame.empty()) {
      utils::logger()->info("[VideoPipeline] 视频读取结束或无可用帧");
      break;
    }

    VideoDebugPacket packet;
    packet.rgb_image = cv::Mat();
    cv::cvtColor(frame, packet.rgb_image, cv::COLOR_BGR2RGB);
    packet.valid = true;

    double timestamp_sec = utils::delta_time(timestamp, start_time_);
    solver_->updateIMU(orientation, timestamp_sec);

    auto armors = detector_->detect(packet.rgb_image);
    std::list<armor_auto_aim::Armor> armor_list(armors.begin(), armors.end());
    auto targets = tracker_->track(armor_list, timestamp);

    if (enable_visualization_) {
      packet.reprojected_armors.reserve(targets.size() * 4);
      for (const auto & target : targets) {
        const auto armor_xyza_list = target.armor_xyza_list();
        for (const Eigen::Vector4d & xyza : armor_xyza_list) {
          Eigen::Vector3d world_point(xyza.x(), xyza.y(), xyza.z());
          auto image_points = yaw_optimizer_->reproject_armor_out(
            world_point, xyza[3], target.armor_type, target.name);
          if (image_points.size() == 4) {
            armor_auto_aim::Visualization vis_armor;
            std::copy(image_points.begin(), image_points.end(), vis_armor.corners.begin());
            vis_armor.name = target.name;
            vis_armor.type = target.armor_type;
            packet.reprojected_armors.push_back(vis_armor);
          }
        }
      }
    }

    if (enable_visualization_) {
      visualize(packet);
    }
  }

  if (enable_visualization_) {
    cv::destroyWindow(visualization_window_name_);
  }

  return 0;
}

void VideoApp::request_stop()
{
  quit_.store(true);
}

void VideoApp::visualize(const VideoDebugPacket & packet)
{
  if (!packet.valid) {
    return;
  }

  try {
    cv::Mat canvas = packet.rgb_image.clone();
    const int frame_index = visualization_frame_counter_.fetch_add(1) + 1;
    detector_->visualize_results(
      canvas, packet.reprojected_armors, visualization_center_point_, frame_index);
    cv::cvtColor(canvas, canvas, cv::COLOR_RGB2BGR);
    cv::imshow(visualization_window_name_, canvas);
    cv::waitKey(1);
  } catch (const std::exception & e) {
    utils::logger()->warn("[VideoPipeline] 可视化失败: {}", e.what());
  }
}

}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys = std::string("{help h usage ? | | 输出命令行参数说明}")
                             + "{video v |" + std::string(kDefaultVideoPath) + "| 视频文件路径}"
                             + "{config c|" + std::string(kDefaultConfigPath) + "| YAML配置文件路径}"
                             + "{@video-path    | | 视频文件路径(可选)}"
                             + "{@config-path   | | YAML配置文件路径(可选)}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  if (!cli.check()) {
    utils::logger()->error("[VideoPipeline] 命令行参数解析失败");
    cli.printErrors();
    return 1;
  }

  std::string video_path = cli.has("@video-path") ? cli.get<std::string>("@video-path") : "";
  if (video_path.empty()) {
    video_path = cli.get<std::string>("video");
  }

  std::string config_path = cli.has("@config-path") ? cli.get<std::string>("@config-path") : "";
  if (config_path.empty()) {
    config_path = cli.get<std::string>("config");
  }

  if (video_path == "video-path" && argc >= 3) {
    video_path = argv[2];
  }

  if (!cli.has("@video-path") && !cli.has("video")) {
    utils::logger()->info("[VideoPipeline] 未提供视频路径, 使用默认路径: {}", video_path);
  }

  std::signal(SIGINT, Application::handle_signal);

  try {
    Application::VideoApp app(config_path, video_path);
    return app.run();
  } catch (const std::exception & e) {
    utils::logger()->error("[VideoPipeline] 程序异常终止: {}", e.what());
  }

  return 1;
}
