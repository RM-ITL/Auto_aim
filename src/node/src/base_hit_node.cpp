#include "base_hit_node.hpp"

#include <csignal>
#include <exception>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hpp"

namespace base_hit
{

namespace
{
std::atomic<bool> g_stop_requested{false};

void handle_signal(int)
{
  g_stop_requested.store(true);
}
}  // namespace

BaseHitNode::BaseHitNode(const std::string & config_path)
: config_path_(config_path),
  start_time_(std::chrono::steady_clock::now())
{
  // 初始化相机
  camera_ = std::make_unique<camera::HikCamera>(config_path_);

  // 初始化检测器
  detector_ = std::make_unique<Detector>(config_path_);

  // 初始化性能监控
  utils::PerformanceMonitor::Config perf_config;
  perf_config.enable_logging = true;
  perf_config.print_interval_sec = 5.0;
  perf_config.logger_name = "base_hit";
  perf_monitor_.set_config(perf_config);
  perf_monitor_.register_metric("detect");
  perf_monitor_.register_metric("total");
  perf_monitor_.reset_all();

  utils::logger()->info("[BaseHitNode] 初始化完成, config: {}", config_path_);
}

BaseHitNode::~BaseHitNode()
{
  request_stop();
}

int BaseHitNode::run()
{
  // 设置信号处理
  std::signal(SIGINT, handle_signal);

  // 创建可视化窗口
  if (enable_visualization_) {
    try {
      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
      cv::resizeWindow(window_name_, 1280, 720);
    } catch (const cv::Exception & e) {
      utils::logger()->warn("[BaseHitNode] 创建窗口失败: {}", e.what());
      enable_visualization_ = false;
    }
  }

  utils::logger()->info("[BaseHitNode] 开始运行主循环");

  int frame_count = 0;

  while (!quit_.load()) {
    if (g_stop_requested.load()) {
      utils::logger()->info("[BaseHitNode] 收到停止信号");
      break;
    }

    auto total_timer = perf_monitor_.create_timer("total");

    // 读取图像
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    camera_->read(img, timestamp);

    if (img.empty()) {
      utils::logger()->warn("[BaseHitNode] 读取到空图像");
      continue;
    }

    // 检测
    std::vector<Detector::Detection> detections;
    {
      auto detect_timer = perf_monitor_.create_timer("detect");
      detections = detector_->detect(img);
      detect_timer.set_success(true);
    }

    frame_count++;
    if (frame_count <= 5 || frame_count % 100 == 0) {
      utils::logger()->debug(
        "[BaseHitNode] 帧{}: 检测到{}个目标",
        frame_count, detections.size());
    }

    // 可视化
    if (enable_visualization_) {
      visualize(img, detections);
    }

    total_timer.set_success(true);
  }

  // 清理
  if (enable_visualization_) {
    cv::destroyWindow(window_name_);
  }

  utils::logger()->info("[BaseHitNode] 运行结束, 共处理{}帧", frame_count);
  return 0;
}

void BaseHitNode::request_stop()
{
  quit_.store(true);
}

void BaseHitNode::visualize(
  const cv::Mat & img,
  const std::vector<Detector::Detection> & detections)
{
  try {
    cv::Mat canvas;
    // 转换为 RGB 用于显示（与 test_node 保持一致）
    cv::cvtColor(img, canvas, cv::COLOR_BGR2RGB);

    // 绘制检测结果
    detector_->visualize(canvas, detections);

    cv::imshow(window_name_, canvas);

    int key = cv::waitKey(1);
    if (key == 27 || key == 'q') {  // ESC 或 q 退出
      quit_.store(true);
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[BaseHitNode] 可视化失败: {}", e.what());
  }
}

}  // namespace base_hit

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | YAML配置文件路径}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  std::string config_path = "/home/guo/ITL_Auto_aim/src/config/config.yaml";
  if (cli.has("@config-path")) {
    config_path = cli.get<std::string>("@config-path");
  }

  try {
    base_hit::BaseHitNode app(config_path);
    return app.run();
  } catch (const std::exception & e) {
    utils::logger()->error("[BaseHitNode] 程序异常: {}", e.what());
    return 1;
  }
}
