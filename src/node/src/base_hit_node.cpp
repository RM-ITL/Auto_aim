#include "base_hit_node.hpp"

#include <csignal>
#include <exception>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hpp"

namespace auto_base
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
  ros_node_ = std::make_shared<rclcpp::Node>("base_hit_node");
  hit_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Basehit>(
    "center", rclcpp::QoS(10)
  );

  camera_ = std::make_unique<camera::Camera>(config_path_);

  detector_ = std::make_unique<Detector>(config_path_);

  tracker_ = std::make_unique<LightTracker>(config_path_);

  aimer_ = std::make_unique<LightAimer>(config_path_);

  // 初始化下位机模拟器
  dart_sim_ = std::make_unique<io::DartSimulator>();
  utils::logger()->info("[BaseHitNode] 使用模拟下位机");

  // 初始化性能监控
  utils::PerformanceMonitor::Config perf_config;
  perf_config.enable_logging = true;
  perf_config.print_interval_sec = 5.0;
  perf_config.logger_name = "base_hit";
  perf_monitor_.set_config(perf_config);
  perf_monitor_.register_metric("detect");
  perf_monitor_.register_metric("track");
  perf_monitor_.register_metric("aim");
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

    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    camera_->read(img, timestamp);

    if (img.empty()) {
      utils::logger()->warn("[BaseHitNode] 读取到空图像");
      continue;
    }

    // [2] 获取下位机数据（时间戳对齐）
    io::DartToVision dart_data = dart_sim_->get_nearest_state(timestamp);

    std::vector<Detector::GreenLight> detections;
    {
      auto detect_timer = perf_monitor_.create_timer("detect");
      detections = detector_->detect(img);
      detect_timer.set_success(true);
    }
    std::list<LightTarget*> targets;
    {
      auto track_timer = perf_monitor_.create_timer("track");
      targets = tracker_->track(detections, timestamp);
      track_timer.set_success(true);
    }

    float yaw_error = 0.0f;
    int target_status = 0;  // 0: Lost, 1: Found

    if (!targets.empty()) {
      auto aim_timer = perf_monitor_.create_timer("aim");

      // 使用第一个目标进行瞄准
      yaw_error = static_cast<float>(aimer_->aim(targets.front(), dart_data));
      target_status = 1;  // Found

      aim_timer.set_success(true);
    }

    // [6] 发送控制命令到下位机模拟器
    dart_sim_->send(yaw_error, target_status);

    // [7] 发布ROS消息（可选）
    if (hit_pub_ && !detections.empty()) {
      auto msg = autoaim_msgs::msg::Basehit{};
      msg.center_x = static_cast<float>(detections[0].center.x);
      msg.center_y = static_cast<float>(detections[0].center.y);
      msg.yaw_error = yaw_error;
      hit_pub_->publish(msg);
    }

    frame_count++;
    if (frame_count <= 5 || frame_count % 100 == 0) {
      utils::logger()->debug(
        "[BaseHitNode] 帧{}: 检测{}个, 追踪{}个, 状态:{}, yaw_error:{:.2f}",
        frame_count, detections.size(), targets.size(),
        tracker_->state(), yaw_error);
    }

    // [8] 可视化
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
  const std::vector<Detector::GreenLight> & detections)
{
  try {
    cv::Mat canvas;
    // 转换为 RGB 用于显示
    cv::cvtColor(img, canvas, cv::COLOR_BGR2RGB);

    // 绘制检测结果
    detector_->visualize(canvas, detections);

    // 显示追踪器状态
    std::string state_text = "State: " + tracker_->state();
    cv::putText(
      canvas, state_text,
      cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(255, 255, 0), 2);

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

  std::string config_path = "/home/guo/ITL_Auto_aim/src/config/dart.yaml";
  if (cli.has("@config-path")) {
    config_path = cli.get<std::string>("@config-path");
  }


  rclcpp::init(argc, argv);
  std::signal(SIGINT, auto_base::handle_signal);

  try {
    auto_base::BaseHitNode app(config_path);
    int ret = app.run();
    rclcpp::shutdown();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[BaseHitNode] 程序异常: {}", e.what());
    return 1;
  }
}
