// USB 副相机引导测试节点
//
// 目的：验证"副相机发现目标 -> 粗 yaw 引导云台 -> 主相机接管自瞄"整条链路。
//   主链路与 deep_node 一致：主相机 -> Detector -> Tracker -> Planner -> Gimbal。
//   另起一个 USB 线程：USB 相机 -> 独立 Detector -> 选最高优先级敌方装甲板
//     -> alpha = atan((u - cx) / fx) -> 引导 yaw = yaw_mount + yaw_gain * alpha。
//   仲裁：主链路有可规划目标时由主链路控制；主链路无目标超过 main_hold_s，
//         且 USB 连续检测到目标时，由 USB 引导云台（mode=1，不开火）。
//
// 用法：
//   ros2 run test_pipeline usb_detect_node [usb-camera-yaml] [--no-show] [--detect-only]
//   例：ros2 run test_pipeline usb_detect_node src/config/usb_camera.yaml
//
//   --detect-only  只跑 USB 相机 + 检测（原 usb_detect_node 功能），不连主相机/下位机
//   --no-show      不弹窗（此时无法用按键标定）
//
// 窗口按键（任一窗口获得焦点时）：
//   g      开/关 USB 引导（默认见 guide.enable）
//   c      标定：USB 画面与主相机都对准同一目标时按下，记录一个样本并更新 yaw_mount
//          (样本角度分散时同时拟合 yaw_gain)，终端打印可写回 yaml 的值
//   r      清空标定样本，恢复 yaml 中的 yaw_mount / yaw_gain
//   q/ESC  退出
//
// 画框颜色：
//   USB 窗口 / detect-only：绿=被选中的引导目标(敌方色+最高优先级/最靠光轴)；
//     完整模式下 黄=敌方色但未选中，白=非敌方色；detect-only 下非选中一律白。
//   主相机窗口：绿=重投影的跟踪目标(detector 画)。
//
// 颜色相关注意：
//   1. 敌方颜色取自 usb_camera.yaml 的 enemy_color 覆盖项；为空才用 pipeline_config
//      (默认 standard3.yaml) 的 Tracker.enemy_color。改 sentry.yaml 对本节点无效。
//   2. 检测器输入要求 BGR(yolov5 预处理声明输入为 BGR)。V4L2 帧本身就是 BGR，
//      所以 swap_rb 默认 false；若真红装甲板标签显示 blue，检查 swap_rb 是否被设成 true。
//      主相机 Bayer 解码后实际为 RGB 顺序，主链路的 BGR2RGB 恰好把它转回 BGR。
//   3. 图像显示：两个窗口都直接显示 BGR 数据，颜色为真实色。
//
// 标定 yaw_mount / yaw_gain 步骤：
//   把装甲板放进 USB 视野，手动转云台让主相机准星也对准它，按 c 记一个样本；
//   换 3~4 个不同水平位置各按一次(拉开 alpha 跨度)，终端打印可写回 yaml 的 yaw_mount_deg 与 yaw_gain。

#include "usb_detect_node.hpp"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "draw_tools.hpp"
#include "logger.hpp"
#include "math_tools.hpp"

namespace Application
{
namespace
{
std::atomic<bool> g_stop_requested{false};
UsbGuideApp * g_app_instance{nullptr};

constexpr double kDeg = CV_PI / 180.0;

void handle_signal(int)
{
  g_stop_requested.store(true);
  if (g_app_instance) {
    g_app_instance->request_stop();
  }
}

// 若 yaml 中存在 key，则用 v4l2-ctl 设置对应的 V4L2 控制项
void apply_v4l2_ctrl(
  const std::string & device, const YAML::Node & cam, const std::string & yaml_key,
  const std::string & ctrl_name)
{
  if (!cam[yaml_key]) {
    return;
  }
  const int value = cam[yaml_key].as<int>();
  std::ostringstream cmd;
  cmd << "v4l2-ctl -d '" << device << "' -c " << ctrl_name << "=" << value;
  const int ret = std::system(cmd.str().c_str());
  if (ret != 0) {
    utils::logger()->warn("[USBCamera] 设置 {} 失败 (v4l2-ctl 未安装?): {}", ctrl_name, cmd.str());
  } else {
    utils::logger()->info("[USBCamera] {} = {}", ctrl_name, value);
  }
}

int fourcc_from_string(const std::string & s)
{
  if (s.size() == 4) {
    return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
  }
  return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
}

// device: 整数 N -> /dev/videoN；字符串 -> 原样（/dev/v4l/by-path/... 或视频文件路径）
std::string resolve_device(const YAML::Node & cam)
{
  const YAML::Node dev = cam["device"];
  if (!dev) {
    return "/dev/video2";
  }
  try {
    return "/dev/video" + std::to_string(dev.as<int>());
  } catch (const YAML::Exception &) {
    return dev.as<std::string>();
  }
}

// 选引导目标：敌方颜色 + 有效编号，先按优先级(rank 越小越优先)，再按离光心水平距离
int select_target(const std::vector<Armors> & armors, armor_auto_aim::Color enemy, double cx)
{
  int best = -1;
  for (int i = 0; i < static_cast<int>(armors.size()); ++i) {
    const auto & a = armors[i];
    if (a.color != enemy || a.name == armor_auto_aim::ArmorName::not_armor) {
      continue;
    }
    if (best < 0) {
      best = i;
      continue;
    }
    const auto & b = armors[best];
    if (a.rank < b.rank ||
        (a.rank == b.rank && std::abs(a.center.x - cx) < std::abs(b.center.x - cx))) {
      best = i;
    }
  }
  return best;
}

// 粗引导焦距：优先用标定 fx，否则按水平视场角估算
double resolve_fx(const UsbGuideConfig & cfg, int width)
{
  if (cfg.fx > 0.0) {
    return cfg.fx;
  }
  return (width / 2.0) / std::tan(cfg.hfov_deg * kDeg / 2.0);
}

// 标签：编号 大小 颜色 置信度，例 "three small red 92%"
std::string armor_label(const Armors & a)
{
  std::ostringstream label;
  label << armor_auto_aim::armor_name_to_string(a.name) << " "
        << (a.type == armor_auto_aim::ArmorType::small ? "small" : "big") << " "
        << (static_cast<size_t>(a.color) < armor_auto_aim::COLORS.size() ?
              armor_auto_aim::COLORS[a.color] : std::string("unknown"))
        << " " << static_cast<int>(a.confidence * 100) << "%";
  return label.str();
}

void draw_vertical_line(cv::Mat & img, double x, const cv::Scalar & color)
{
  const int xi = static_cast<int>(std::lround(x));
  cv::line(img, {xi, 0}, {xi, img.rows - 1}, color, 1, cv::LINE_AA);
}

}  // namespace

UsbGuideConfig UsbGuideConfig::load(const std::string & path)
{
  const YAML::Node root = YAML::LoadFile(path);
  UsbGuideConfig cfg;

  const std::string cwd = std::filesystem::current_path().string();
  cfg.pipeline_config = root["pipeline_config"] ?
    root["pipeline_config"].as<std::string>() : cwd + "/src/config/standard3.yaml";
  cfg.detector_config = root["detector_config"] ?
    root["detector_config"].as<std::string>() : std::string{};

  cfg.camera = root["usb_camera"];
  if (!cfg.camera) {
    throw std::runtime_error("配置缺少 usb_camera 段: " + path);
  }
  cfg.hfov_deg = cfg.camera["hfov_deg"] ? cfg.camera["hfov_deg"].as<double>() : 90.0;
  cfg.fx = cfg.camera["fx"] ? cfg.camera["fx"].as<double>() : 0.0;
  cfg.cx = cfg.camera["cx"] ? cfg.camera["cx"].as<double>() : -1.0;
  cfg.detector_device = cfg.camera["detector_device"] ?
    cfg.camera["detector_device"].as<std::string>() : std::string{};
  cfg.swap_rb = cfg.camera["swap_rb"] ? cfg.camera["swap_rb"].as<bool>() : false;
  cfg.enemy_color = cfg.camera["enemy_color"] ? cfg.camera["enemy_color"].as<std::string>() :
    std::string{};
  if (!cfg.enemy_color.empty() && cfg.enemy_color != "red" && cfg.enemy_color != "blue") {
    throw std::runtime_error("usb_camera.enemy_color 只能是 red 或 blue");
  }
  if (cfg.fx <= 0.0 && (cfg.hfov_deg <= 1.0 || cfg.hfov_deg >= 179.0)) {
    throw std::runtime_error("usb_camera.hfov_deg 必须在 (1, 179) 度之间，或直接给出 fx");
  }

  const YAML::Node guide = root["guide"];
  if (guide) {
    auto read_d = [&guide](const char * key, double def) {
      return guide[key] ? guide[key].as<double>() : def;
    };
    cfg.guide_enable = guide["enable"] ? guide["enable"].as<bool>() : false;
    cfg.yaw_mount = read_d("yaw_mount_deg", 0.0) * kDeg;
    cfg.yaw_mount_relative =
      guide["yaw_mount_relative"] ? guide["yaw_mount_relative"].as<bool>() : true;
    cfg.yaw_gain = read_d("yaw_gain", -1.0);
    if (guide["pitch_deg"]) {
      cfg.pitch = guide["pitch_deg"].as<double>() * kDeg;
    }
    cfg.max_yaw_error = read_d("max_yaw_error_deg", 0.0) * kDeg;
    cfg.guide_timeout_s = read_d("timeout_s", 0.2);
    cfg.main_hold_s = read_d("main_hold_s", 0.3);
    cfg.min_detect_frames = guide["min_detect_frames"] ? guide["min_detect_frames"].as<int>() : 3;
    cfg.yaw_continuous = guide["yaw_continuous"] ? guide["yaw_continuous"].as<bool>() : true;
    cfg.allow_fire = guide["allow_fire"] ? guide["allow_fire"].as<bool>() : false;
  }
  if (std::abs(cfg.yaw_gain) < 0.1 || std::abs(cfg.yaw_gain) > 10.0) {
    throw std::runtime_error("guide.yaw_gain 绝对值应在 [0.1, 10]，通常为 +1 或 -1");
  }
  if (cfg.min_detect_frames < 1) {
    cfg.min_detect_frames = 1;
  }
  return cfg;
}

// ============================ UsbCamera ============================

UsbCamera::UsbCamera(const YAML::Node & cam) : cam_(cam), device_(resolve_device(cam))
{
  is_device_ = device_.rfind("/dev/", 0) == 0;
  if (!open_device()) {
    throw std::runtime_error("打开 USB 相机失败: " + device_);
  }
  thread_ = std::thread(&UsbCamera::capture_loop, this);
}

UsbCamera::~UsbCamera()
{
  stop();
  if (thread_.joinable()) {
    thread_.join();
  }
  capture_.release();
}

bool UsbCamera::open_device()
{
  if (!is_device_) {
    // 视频文件：离线验证检测与引导角计算
    capture_.open(device_);
    if (!capture_.isOpened()) {
      return false;
    }
    const double fps = capture_.get(cv::CAP_PROP_FPS);
    file_period_ = std::chrono::microseconds(
      static_cast<int64_t>(1e6 / (fps > 1.0 ? fps : 30.0)));
    utils::logger()->info("[USBCamera] 视频文件: {}, FPS={:.1f}", device_, fps);
    return true;
  }

  // 先用 v4l2-ctl 应用曝光/增益等控制项；auto_* 开关要先于其控制的目标项
  apply_v4l2_ctrl(device_, cam_, "auto_exposure", "auto_exposure");
  apply_v4l2_ctrl(device_, cam_, "exposure", "exposure_time_absolute");
  apply_v4l2_ctrl(device_, cam_, "gain", "gain");
  apply_v4l2_ctrl(device_, cam_, "gamma", "gamma");
  apply_v4l2_ctrl(device_, cam_, "brightness", "brightness");
  apply_v4l2_ctrl(device_, cam_, "auto_white_balance", "white_balance_automatic");
  apply_v4l2_ctrl(device_, cam_, "white_balance_temperature", "white_balance_temperature");

  capture_.open(device_, cv::CAP_V4L2);
  if (!capture_.isOpened()) {
    return false;
  }
  if (cam_["fourcc"]) {
    capture_.set(cv::CAP_PROP_FOURCC, fourcc_from_string(cam_["fourcc"].as<std::string>()));
  }
  if (cam_["width"]) capture_.set(cv::CAP_PROP_FRAME_WIDTH, cam_["width"].as<double>());
  if (cam_["height"]) capture_.set(cv::CAP_PROP_FRAME_HEIGHT, cam_["height"].as<double>());
  if (cam_["fps"]) capture_.set(cv::CAP_PROP_FPS, cam_["fps"].as<double>());
  // 驱动缓冲越少，读到的帧越新；取图线程持续读帧，再兜一层
  capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);

  utils::logger()->info(
    "[USBCamera] 已打开: {}, 实际分辨率={}x{}, FPS={}", device_,
    capture_.get(cv::CAP_PROP_FRAME_WIDTH), capture_.get(cv::CAP_PROP_FRAME_HEIGHT),
    capture_.get(cv::CAP_PROP_FPS));
  return true;
}

void UsbCamera::capture_loop()
{
  int fail_count = 0;
  while (!quit_.load()) {
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
      if (!is_device_) {
        capture_.set(cv::CAP_PROP_POS_FRAMES, 0);  // 视频文件循环播放
        std::this_thread::sleep_for(file_period_);
        continue;
      }
      // USB 掉线：连续失败后释放重开，避免整条引导链路卡死
      if (++fail_count % 20 == 0) {
        utils::logger()->warn("[USBCamera] 连续 {} 次读帧失败，尝试重开 {}", fail_count, device_);
        capture_.release();
        open_device();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    fail_count = 0;
    const auto now = std::chrono::steady_clock::now();
    // clone：部分后端返回的 Mat 引用驱动内部缓冲，下一帧会被覆盖
    cv::Mat owned = frame.clone();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_ = owned;
      latest_time_ = now;  // V4L2 无曝光时间戳，用到达时刻近似；粗引导足够
      ++seq_;
    }
    frame_cv_.notify_all();
    if (!is_device_) {
      std::this_thread::sleep_for(file_period_);
    }
  }
}

bool UsbCamera::wait_new_frame(
  cv::Mat & img, std::chrono::steady_clock::time_point & timestamp, uint64_t & last_seq,
  std::chrono::milliseconds timeout)
{
  std::unique_lock<std::mutex> lock(mutex_);
  if (!frame_cv_.wait_for(
        lock, timeout, [&] { return quit_.load() || seq_ > last_seq; })) {
    return false;
  }
  if (quit_.load()) {
    return false;
  }
  img = latest_;  // 取图线程每帧新建 Mat，这里共享引用不会被覆盖
  timestamp = latest_time_;
  last_seq = seq_;
  return true;
}

void UsbCamera::stop()
{
  quit_.store(true);
  frame_cv_.notify_all();
}

// ============================ UsbGuideApp ============================

UsbGuideApp::UsbGuideApp(
  const app_config::AppConfig & app_config, const UsbGuideConfig & guide_config, bool show)
: guide_config_(guide_config),
  enable_visualization_(show),
  start_time_(std::chrono::steady_clock::now()),
  last_delay_log_time_(std::chrono::steady_clock::now())
{
  ros_node_ = std::make_shared<rclcpp::Node>("usb_guide_node");
  debug_pub_ = ros_node_->create_publisher<autoaim_msgs::msg::Debug>("debug", rclcpp::QoS(10));
  utils::logger()->info("[UsbGuide] 主链路配置: {}", app_config.source_path);

  // 主链路：与 deep_node 相同
  camera_ = std::make_unique<camera::Camera>(app_config.camera);
  detector_ = std::make_unique<armor_auto_aim::Detector>(app_config.detector);
  solver_ = std::make_unique<solver::Solver>(app_config.solver);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(app_config.tracker, *solver_);
  planner_ = std::make_unique<plan::Planner>(app_config.planner);
  gimbal_ = std::make_unique<io::Gimbal>(app_config.gimbal);
  shooter_ = std::make_unique<shooter::Shooter>(app_config.shooter);

  // USB 引导：独立检测器实例，检测参数默认与主链路相同
  auto usb_detector_config = guide_config_.detector_config.empty() ?
    app_config.detector : app_config::AppConfig::load(guide_config_.detector_config).detector;
  if (!guide_config_.detector_device.empty()) {
    usb_detector_config.yolov5.device = guide_config_.detector_device;
    usb_detector_config.yolo11.device = guide_config_.detector_device;
  }
  usb_detector_ = std::make_unique<armor_auto_aim::Detector>(usb_detector_config);
  usb_camera_ = std::make_unique<UsbCamera>(guide_config_.camera);

  // 敌方颜色：usb_camera.yaml 的 enemy_color 覆盖优先，否则用 pipeline 配置的 Tracker.enemy_color
  const std::string enemy_str = guide_config_.enemy_color.empty() ?
    app_config.tracker.enemy_color : guide_config_.enemy_color;
  enemy_color_ = (enemy_str == "red") ? armor_auto_aim::Color::red : armor_auto_aim::Color::blue;
  guide_enabled_.store(guide_config_.guide_enable);
  yaw_gain_.store(guide_config_.yaw_gain);
  yaw_mount_.store(guide_config_.yaw_mount);  // 相对模式在收到首个云台状态后再加上基准 yaw
  mount_ready_.store(!guide_config_.yaw_mount_relative);

  utils::logger()->info(
    "[UsbGuide] USB 设备={} 敌方={} 引导={} yaw_mount={:.1f}°({}) yaw_gain={:.2f} "
    "yaw_continuous={} allow_fire={}",
    usb_camera_->device(), enemy_str,
    guide_config_.guide_enable ? "开" : "关(按 g 开启)", guide_config_.yaw_mount / kDeg,
    guide_config_.yaw_mount_relative ? "相对启动时云台" : "绝对", guide_config_.yaw_gain,
    guide_config_.yaw_continuous, guide_config_.allow_fire);
  g_app_instance = this;
}

UsbGuideApp::~UsbGuideApp()
{
  g_app_instance = nullptr;
  request_stop();
  join_threads();
}

void UsbGuideApp::request_stop()
{
  if (!quit_.exchange(true)) {
    // 关闭队列唤醒阻塞线程；关闭相机让 read() 不再阻塞
    visualization_queue.shutdown();
    target_queue.shutdown();
    if (gimbal_) gimbal_->stop();
    if (camera_) camera_->stop();
    if (usb_camera_) usb_camera_->stop();
    utils::logger()->info("[UsbGuide] 正在停止...");
  }
}

void UsbGuideApp::start_threads()
{
  quit_.store(false);
  planner_thread_ = std::thread(&UsbGuideApp::planner_loop, this);
  sender_thread_ = std::thread(&UsbGuideApp::sender_loop, this);
  usb_detect_thread_ = std::thread(&UsbGuideApp::usb_detect_loop, this);
  if (enable_visualization_) {
    visualization_thread_ = std::thread(&UsbGuideApp::visualization_loop, this);
  }
}

void UsbGuideApp::join_threads()
{
  for (auto * t : {&planner_thread_, &sender_thread_, &usb_detect_thread_, &visualization_thread_}) {
    if (t->joinable()) t->join();
  }
}

// 主相机检测线程（与 deep_node::run 相同，去掉 dm_imu 与偏移标定）
int UsbGuideApp::run()
{
  start_threads();
  std::string last_state = tracker_->state();

  while (!quit_.load() && !g_stop_requested.load()) {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    camera_->read(img, timestamp);
    if (quit_.load() || g_stop_requested.load() || img.empty()) {
      break;
    }

    DebugPacket debug_packet;
    cv::cvtColor(img, debug_packet.rgb_image, cv::COLOR_BGR2RGB);

    auto orientation_opt = gimbal_->q_for_image(timestamp);
    if (!orientation_opt) {
      continue;  // 无有效 IMU 样本：丢帧，不污染 EKF
    }
    solver_->updateIMU(*orientation_opt, utils::delta_time(timestamp, start_time_));

    // 首次拿到云台数据：确定 yaw_mount 基准，之后才允许 USB 引导
    if (!mount_ready_.load()) {
      const double base = guide_config_.yaw_mount_relative ? gimbal_->state().yaw : 0.0;
      base_yaw_.store(base);
      yaw_mount_.store(utils::limit_rad(guide_config_.yaw_mount + base));
      mount_ready_.store(true);
      utils::logger()->info(
        "[UsbGuide] 已收到云台数据，基准 yaw={:.1f}°，生效 yaw_mount={:.1f}°",
        base / kDeg, yaw_mount_.load() / kDeg);
    }

    auto armor = detector_->detect(debug_packet.rgb_image);
    std::list<armor_auto_aim::Armor> armor_list(armor.begin(), armor.end());
    auto targets = tracker_->track(armor_list, timestamp);

    // 本帧主相机原始方位角(未经 EKF)：只有 solver 本帧解算成功时才更新，供标定参考
    {
      const double frame_sec = std::chrono::duration<double>(timestamp.time_since_epoch()).count();
      const auto pose = solver_->getLastpose();
      if (solver_->getLastPnPResult().success && std::abs(pose.timestamp - frame_sec) < 1e-6) {
        std::lock_guard<std::mutex> lk(main_ref_mutex_);
        main_ref_yaw_ = pose.world_spherical.yaw;
        main_ref_time_ = timestamp;
      }
    }

    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    // 主链路延迟：USB 检测器与主检测器抢 CPU 时这里会变大
    const auto now = std::chrono::steady_clock::now();
    delay_window_.push_back(std::chrono::duration<double, std::milli>(now - timestamp).count());
    if (delay_window_.size() > delay_window_size_) delay_window_.pop_front();
    if (now - last_delay_log_time_ > std::chrono::seconds(5) && delay_window_.size() >= 10) {
      std::vector<double> sorted(delay_window_.begin(), delay_window_.end());
      std::sort(sorted.begin(), sorted.end());
      const double avg = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
      utils::logger()->info(
        "[UsbGuide] 主链路延迟 - 平均: {:.2f}ms, 95分位: {:.2f}ms, 最大: {:.2f}ms | USB {:.1f}fps",
        avg, sorted[static_cast<size_t>(sorted.size() * 0.95)], sorted.back(), usb_fps_.load());
      last_delay_log_time_ = now;
    }

    if (enable_visualization_) {
      for (const auto & target : targets) {
        const auto xyza_list = std::visit([](const auto & t) { return t.armor_xyza_list(); }, target);
        const auto armor_type = std::visit([](const auto & t) { return t.armor_type; }, target);
        const auto target_name = std::visit([](const auto & t) { return t.name; }, target);
        for (const Eigen::Vector4d & xyza : xyza_list) {
          auto image_points = yaw_optimizer_->reproject_armor_out(
            xyza.head<3>(), xyza[3], armor_type, target_name);
          if (image_points.size() == 4) {
            Visualization vis_armor;
            std::copy(image_points.begin(), image_points.end(), vis_armor.corners.begin());
            vis_armor.name = target_name;
            vis_armor.type = armor_type;
            debug_packet.reprojected_armors.push_back(vis_armor);
          }
        }
      }
    }

    debug_packet.tracker_state = tracker_->state();
    debug_packet.valid = true;
    if (enable_visualization_) {
      visualization_queue.push(debug_packet);
    }
    if (debug_packet.tracker_state != last_state) {
      utils::logger()->info(
        "[UsbGuide] Tracker状态切换: {} -> {}", last_state, debug_packet.tracker_state);
      last_state = debug_packet.tracker_state;
    }
  }

  request_stop();
  join_threads();
  return 0;
}

double UsbGuideApp::guide_yaw(double alpha) const
{
  return utils::limit_rad(yaw_mount_.load() + yaw_gain_.load() * alpha);
}

// USB 检测线程：取最新帧 -> 检测 -> 选目标 -> 计算引导 yaw
void UsbGuideApp::usb_detect_loop()
{
  utils::logger()->info("[UsbGuide] USB 检测线程启动");
  uint64_t last_seq = 0;
  int detect_count = 0;
  int fps_frames = 0;
  auto fps_start = std::chrono::steady_clock::now();

  while (!quit_.load() && !g_stop_requested.load()) {
    cv::Mat bgr;
    std::chrono::steady_clock::time_point frame_time;
    if (!usb_camera_->wait_new_frame(bgr, frame_time, last_seq, std::chrono::milliseconds(200)) ||
        bgr.empty()) {
      continue;
    }

    cv::Mat input = bgr;
    if (guide_config_.swap_rb) {
      cv::cvtColor(bgr, input, cv::COLOR_BGR2RGB);
    }
    std::vector<Armors> armors;
    try {
      armors = usb_detector_->detect(input);
    } catch (const std::exception & e) {
      utils::logger()->warn("[UsbGuide] USB 检测失败: {}", e.what());
      continue;
    }

    const double cx = guide_config_.cx >= 0.0 ? guide_config_.cx : bgr.cols / 2.0;
    const int selected = select_target(armors, enemy_color_, cx);

    UsbGuideSample sample;
    sample.time = frame_time;
    sample.fx = resolve_fx(guide_config_, bgr.cols);
    if (selected >= 0) {
      ++detect_count;
      sample.detected = true;
      sample.detect_count = detect_count;
      sample.alpha = std::atan((armors[selected].center.x - cx) / sample.fx);
      sample.yaw = guide_yaw(sample.alpha);
      sample.name = armors[selected].name;
      sample.valid = detect_count >= guide_config_.min_detect_frames && mount_ready_.load();
    } else {
      detect_count = 0;
    }
    {
      std::lock_guard<std::mutex> lk(guide_mutex_);
      guide_sample_ = sample;
    }

    if (enable_visualization_) {
      std::lock_guard<std::mutex> lk(usb_debug_mutex_);
      usb_debug_.bgr = bgr;
      usb_debug_.armors = std::move(armors);
      usb_debug_.selected = selected;
      usb_debug_.cx = cx;
      usb_debug_.sample = sample;
      usb_debug_.valid = true;
    }

    ++fps_frames;
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = utils::delta_time(now, fps_start);
    if (elapsed >= 1.0) {
      usb_fps_.store(fps_frames / elapsed);
      fps_frames = 0;
      fps_start = now;
    }
  }
  utils::logger()->info("[UsbGuide] USB 检测线程退出");
}

plan::Plan UsbGuideApp::make_guide_plan(
  const UsbGuideSample & sample, const io::GimbalState & gs, float pitch_hold) const
{
  double err = utils::limit_rad(sample.yaw - gs.yaw);  // 最短角差
  if (guide_config_.max_yaw_error > 0.0) {
    err = std::clamp(err, -guide_config_.max_yaw_error, guide_config_.max_yaw_error);
  }
  plan::Plan plan;
  plan.control = true;
  plan.fire = false;  // 引导阶段不开火
  plan.target_yaw = static_cast<float>(sample.yaw);
  plan.yaw = static_cast<float>(gs.yaw + err);  // 下发前统一按 yaw_continuous 处理
  plan.pitch = pitch_hold;
  plan.target_pitch = pitch_hold;
  return plan;
}

void UsbGuideApp::planner_loop()
{
  utils::logger()->info("[UsbGuide] 规划线程启动");
  auto last_main_time = std::chrono::steady_clock::now() - std::chrono::hours(1);
  ControlSource last_source = ControlSource::None;
  float pitch_hold = 0.0f;
  const auto hold = std::chrono::duration<double>(guide_config_.main_hold_s);
  const auto timeout = std::chrono::duration<double>(guide_config_.guide_timeout_s);

  while (!quit_.load() && !g_stop_requested.load()) {
    std::optional<tracker::TargetVariant> target;
    if (!target_queue.empty()) {
      target = target_queue.front();  // 与 deep_node 相同：front 不出队，始终取最新目标
    }
    const auto gs = gimbal_->state();
    const auto now = std::chrono::steady_clock::now();

    // 1. 主链路：有可规划目标时始终优先
    plan::Plan plan = planner_->plan(target, gs.bullet_speed);
    ControlSource source = ControlSource::None;
    if (target.has_value()) {
      last_main_time = now;  // tracker 有目标(含 detecting/temp_lost)时 USB 不抢控制权
    }
    if (target.has_value() && plan.control) {
      source = ControlSource::Main;
      plan.fire = guide_config_.allow_fire && plan.fire &&
                  shooter_->checkfire(plan.yaw, plan.pitch, gs, target.value());
    } else {
      plan = plan::Plan{};
      // 2. 主链路无目标超过 main_hold_s，且 USB 样本新鲜有效 -> USB 引导
      UsbGuideSample sample;
      {
        std::lock_guard<std::mutex> lk(guide_mutex_);
        sample = guide_sample_;
      }
      const bool fresh = sample.valid && now - sample.time < timeout;
      if (guide_enabled_.load() && fresh && now - last_main_time > hold) {
        if (last_source != ControlSource::Usb) {
          pitch_hold = static_cast<float>(guide_config_.pitch.value_or(gs.pitch));
        }
        plan = make_guide_plan(sample, gs, pitch_hold);
        source = ControlSource::Usb;
        if (last_source != ControlSource::Usb) {
          utils::logger()->info(
            "[UsbGuide] USB 接管: 目标={} 偏角={:.1f}° 引导yaw={:.1f}° 当前yaw={:.1f}°",
            armor_auto_aim::armor_name_to_string(sample.name), sample.alpha / kDeg,
            sample.yaw / kDeg, gs.yaw / kDeg);
        }
      }
    }

    if (source != last_source) {
      static const char * kNames[] = {"none", "main", "usb"};
      utils::logger()->info(
        "[UsbGuide] 控制权: {} -> {}", kNames[static_cast<int>(last_source)],
        kNames[static_cast<int>(source)]);
      last_source = source;
    }
    control_source_.store(source);

    // 下发 yaw 统一处理：连续模式发 gs.yaw + 最短角差，避免下位机多圈 yaw 时绕远路
    if (plan.control) {
      plan.yaw = guide_config_.yaw_continuous ?
        static_cast<float>(gs.yaw + utils::limit_rad(plan.yaw - gs.yaw)) :
        static_cast<float>(utils::limit_rad(plan.yaw));
    }

    update_latest_plan(plan);
    publish_debug(plan, gs, source == ControlSource::Main);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  utils::logger()->info("[UsbGuide] 规划线程退出");
}

void UsbGuideApp::publish_debug(
  const plan::Plan & plan, const io::GimbalState & gs, bool target_valid)
{
  if (!debug_pub_) return;
  auto msg = autoaim_msgs::msg::Debug{};
  msg.enable_control = plan.control;
  msg.fire = plan.fire;
  msg.yaw_offest = plan.control ? static_cast<float>(utils::limit_rad(plan.yaw - gs.yaw)) : 0.0f;
  msg.pitch_offset = plan.control ? plan.pitch - gs.pitch : 0.0f;
  msg.yaw = plan.yaw;
  msg.pitch = plan.pitch;
  msg.yaw_gimbal = gs.yaw;
  msg.pitch_gimbal = gs.pitch;
  msg.bullet_speed = gs.bullet_speed;
  msg.yaw_vel = plan.yaw_vel;
  msg.pitch_vel = plan.pitch_vel;
  msg.yaw_acc = plan.yaw_acc;
  msg.pitch_acc = plan.pitch_acc;
  msg.yaw_vel_gimbal = gs.yaw_vel;
  msg.pitch_vel_gimbal = gs.pitch_vel;
  msg.target_valid = target_valid;
  msg.aim_plate_index = plan.aim_plate_index;
  msg.target_yaw = plan.target_yaw;
  debug_pub_->publish(msg);
}

void UsbGuideApp::update_latest_plan(const plan::Plan & plan)
{
  std::lock_guard<std::mutex> lock(latest_plan_mutex_);
  latest_plan_ = plan;
  latest_plan_valid_ = true;
  latest_plan_time_ = std::chrono::steady_clock::now();
}

UsbGuideApp::LatestPlanSnapshot UsbGuideApp::latest_plan_snapshot()
{
  std::lock_guard<std::mutex> lock(latest_plan_mutex_);
  return {latest_plan_, latest_plan_valid_, latest_plan_time_};
}

// 定频下发线程（与 deep_node::sender_loop 相同的 100Hz + 100ms 过期保护）
void UsbGuideApp::sender_loop()
{
  utils::logger()->info("[UsbGuide] 云台发送线程启动");
  constexpr auto send_period = std::chrono::milliseconds(10);
  constexpr auto latest_plan_timeout = std::chrono::milliseconds(100);

  while (!quit_.load() && !g_stop_requested.load()) {
    const auto cycle_start = std::chrono::steady_clock::now();
    const auto snapshot = latest_plan_snapshot();
    const bool expired = !snapshot.valid || cycle_start - snapshot.time > latest_plan_timeout;

    if (!expired && snapshot.plan.control) {
      const auto & p = snapshot.plan;
      gimbal_->send(
        p.control, p.fire, p.yaw, p.yaw_vel, p.yaw_acc, p.pitch, p.pitch_vel, p.pitch_acc);
    } else if (!expired) {
      const auto gs = gimbal_->state();
      gimbal_->send(false, false, gs.yaw, 0, 0, gs.pitch, 0, 0);
    } else {
      gimbal_->send(false, false, 0, 0, 0, 0, 0, 0);
    }

    const auto elapsed = std::chrono::steady_clock::now() - cycle_start;
    if (elapsed < send_period) {
      std::this_thread::sleep_for(send_period - elapsed);
    }
  }
  utils::logger()->info("[UsbGuide] 云台发送线程退出");
}

// 标定：USB 与主相机看到同一目标时调用。参考 yaw 优先用主相机原始解算方位角，
// 没有主相机解算时用当前云台 yaw（要求已手动把主相机准星对准目标）。
void UsbGuideApp::calibrate_yaw_mount()
{
  const auto now = std::chrono::steady_clock::now();
  UsbGuideSample sample;
  {
    std::lock_guard<std::mutex> lk(guide_mutex_);
    sample = guide_sample_;
  }
  if (!sample.detected || now - sample.time > std::chrono::milliseconds(300)) {
    utils::logger()->warn("[Calib] USB 画面当前没有敌方装甲板，样本未记录");
    return;
  }

  double ref_yaw = 0.0;
  const char * ref_name = "云台 yaw";
  {
    std::lock_guard<std::mutex> lk(main_ref_mutex_);
    if (now - main_ref_time_ < std::chrono::milliseconds(300)) {
      ref_yaw = main_ref_yaw_;
      ref_name = "主相机解算方位角";
    } else {
      ref_yaw = gimbal_->state().yaw;
    }
  }
  // 参考 yaw 展开到第一个样本附近，避免跨 ±pi 时拟合出错
  if (!calib_samples_.empty()) {
    ref_yaw = calib_samples_.front().first + utils::limit_rad(ref_yaw - calib_samples_.front().first);
  }
  calib_samples_.emplace_back(ref_yaw, sample.alpha);

  double gain = yaw_gain_.load();
  double mount = ref_yaw - gain * sample.alpha;
  const auto minmax = std::minmax_element(
    calib_samples_.begin(), calib_samples_.end(),
    [](const auto & a, const auto & b) { return a.second < b.second; });
  const double spread = minmax.second->second - minmax.first->second;
  if (calib_samples_.size() >= 2 && spread > 10.0 * kDeg) {
    // 最小二乘拟合 ref = mount + gain * alpha
    const double n = static_cast<double>(calib_samples_.size());
    double sa = 0, sr = 0, saa = 0, sar = 0;
    for (const auto & [r, a] : calib_samples_) {
      sa += a; sr += r; saa += a * a; sar += a * r;
    }
    gain = (n * sar - sa * sr) / (n * saa - sa * sa);
    mount = (sr - gain * sa) / n;
  } else {
    // 样本角度太集中：只更新 yaw_mount，gain 保持；多个样本取平均
    double sum = 0.0;
    for (const auto & [r, a] : calib_samples_) sum += r - gain * a;
    mount = sum / static_cast<double>(calib_samples_.size());
  }
  yaw_gain_.store(gain);
  yaw_mount_.store(utils::limit_rad(mount));

  const double mount_cfg = guide_config_.yaw_mount_relative ?
    utils::limit_rad(mount - base_yaw_.load()) : utils::limit_rad(mount);
  utils::logger()->info(
    "[Calib] 样本#{} 参考={}({:.2f}°) USB偏角={:.2f}° -> yaw_mount={:.2f}° yaw_gain={:.3f}",
    calib_samples_.size(), ref_name, ref_yaw / kDeg, sample.alpha / kDeg, mount / kDeg, gain);
  utils::logger()->info(
    "[Calib] 写回 usb_camera.yaml: guide.yaw_mount_deg: {:.2f}  guide.yaw_gain: {:.3f}",
    mount_cfg / kDeg, gain);
  if (std::abs(std::abs(gain) - 1.0) > 0.15) {
    // |gain| 偏离 1 说明 fx 不准：把 gain 吸收进 fx，yaw_gain 写回 ±1
    utils::logger()->info(
      "[Calib] |yaw_gain| 偏离 1 较多，也可改为 usb_camera.fx: {:.0f} 且 guide.yaw_gain: {:.0f}",
      sample.fx / std::abs(gain), gain > 0 ? 1.0 : -1.0);
  }
}

void UsbGuideApp::reset_calibration()
{
  calib_samples_.clear();
  yaw_gain_.store(guide_config_.yaw_gain);
  yaw_mount_.store(utils::limit_rad(guide_config_.yaw_mount + base_yaw_.load()));
  utils::logger()->info(
    "[Calib] 已清空样本，恢复 yaw_mount={:.2f}° yaw_gain={:.3f}", yaw_mount_.load() / kDeg,
    guide_config_.yaw_gain);
}

// 可视化线程：主相机窗口 + USB 窗口，按键处理也在这里（OpenCV GUI 需单线程）
void UsbGuideApp::visualization_loop()
{
  utils::logger()->info("[UsbGuide] 可视化线程启动  按键: g=引导开关 c=标定样本 r=清空标定 q=退出");
  try {
    cv::namedWindow(main_window_name_, cv::WINDOW_NORMAL);
    cv::namedWindow(usb_window_name_, cv::WINDOW_NORMAL);
  } catch (const cv::Exception & e) {
    utils::logger()->error("[UsbGuide] 创建可视化窗口失败: {}", e.what());
    return;
  }
  static const char * kSource[] = {"NONE", "MAIN", "USB"};

  while (!quit_.load() && !g_stop_requested.load()) {
    DebugPacket packet;
    if (!visualization_queue.try_pop(packet, std::chrono::milliseconds(50))) {
      // 主相机暂时无帧时也要刷新 USB 窗口和响应按键
      packet.valid = false;
    }

    try {
      if (packet.valid) {
        cv::Mat canvas = packet.rgb_image.clone();
        const int frame_index = visualization_frame_counter_.fetch_add(1) + 1;
        detector_->visualize_results(
          canvas, packet.reprojected_armors, {canvas.cols / 2, canvas.rows / 2}, frame_index);
        utils::draw_text(
          canvas, std::string("tracker: ") + packet.tracker_state + "  ctrl: " +
          kSource[static_cast<int>(control_source_.load())], {10, 60}, utils::colors::YELLOW,
          0.8, 2);
        // 主相机经 Bayer 解码 + BGR2RGB 后，rgb_image 实际是 BGR 顺序（与 deep_node 一致），直接显示
        cv::imshow(main_window_name_, canvas);
      }

      UsbDebugPacket usb;
      {
        std::lock_guard<std::mutex> lk(usb_debug_mutex_);
        if (usb_debug_.valid) {
          usb = usb_debug_;
          usb_debug_.valid = false;  // 只画新帧
        }
      }
      if (usb.valid) {
        cv::Mat canvas = usb.bgr.clone();
        draw_vertical_line(canvas, usb.cx, utils::colors::YELLOW);  // 光轴(alpha=0)
        for (int i = 0; i < static_cast<int>(usb.armors.size()); ++i) {
          const auto & a = usb.armors[i];
          const bool enemy = a.color == enemy_color_;
          const cv::Scalar color = i == usb.selected ? utils::colors::GREEN :
            (enemy ? utils::colors::YELLOW : utils::colors::WHITE);
          if (a.points.size() == 4) {
            utils::draw_quadrangle_with_corners(canvas, a.points, color, 2, 3);
          }
          utils::draw_detection_label(canvas, armor_label(a), a.center, color, 0.6, 2);
        }
        std::ostringstream info;
        info.setf(std::ios::fixed);
        info.precision(1);
        if (usb.sample.detected) {
          info << "alpha=" << usb.sample.alpha / kDeg << "deg  guide_yaw="
               << usb.sample.yaw / kDeg << "deg  n=" << usb.sample.detect_count
               << (usb.sample.valid ? " [valid]" : " [wait]");
        } else {
          info << "no enemy";
        }
        utils::draw_text(canvas, info.str(), {10, 30}, utils::colors::GREEN, 0.7, 2);
        std::ostringstream state;
        state.setf(std::ios::fixed);
        state.precision(1);
        state << "enemy=" << (enemy_color_ == armor_auto_aim::Color::red ? "red" : "blue")
              << "  guide=" << (guide_enabled_.load() ? "ON" : "OFF")
              << "  ctrl=" << kSource[static_cast<int>(control_source_.load())]
              << "  mount=" << yaw_mount_.load() / kDeg << "deg  gain=" << yaw_gain_.load()
              << "  calib=" << calib_samples_.size() << "  " << usb_fps_.load() << "fps";
        utils::draw_text(canvas, state.str(), {10, 60}, utils::colors::CYAN, 0.6, 2);
        cv::imshow(usb_window_name_, canvas);
      }

      const int key = cv::waitKey(1);
      if (key == 'q' || key == 27) {
        g_stop_requested.store(true);
        request_stop();
        break;
      } else if (key == 'g') {
        const bool on = !guide_enabled_.load();
        guide_enabled_.store(on);
        utils::logger()->info("[UsbGuide] USB 引导 {}", on ? "开启" : "关闭");
      } else if (key == 'c') {
        calibrate_yaw_mount();
      } else if (key == 'r') {
        reset_calibration();
      }
    } catch (const std::exception & e) {
      utils::logger()->warn("[UsbGuide] 可视化帧处理失败: {}", e.what());
    }
  }

  cv::destroyAllWindows();
  utils::logger()->info("[UsbGuide] 可视化线程退出");
}

// 只测 USB 识别 + 引导角计算（原 usb_detect_node 行为），不连主相机/下位机
int run_detect_only(const UsbGuideConfig & guide_config, bool show)
{
  const std::string detector_cfg = guide_config.detector_config.empty() ?
    guide_config.pipeline_config : guide_config.detector_config;
  const auto app_config = app_config::AppConfig::load(detector_cfg);
  auto detector_config = app_config.detector;
  if (!guide_config.detector_device.empty()) {
    detector_config.yolov5.device = guide_config.detector_device;
    detector_config.yolo11.device = guide_config.detector_device;
  }
  armor_auto_aim::Detector detector(detector_config);
  const std::string enemy_str = guide_config.enemy_color.empty() ?
    app_config.tracker.enemy_color : guide_config.enemy_color;
  const auto enemy = (enemy_str == "red") ?
    armor_auto_aim::Color::red : armor_auto_aim::Color::blue;
  utils::logger()->info(
    "[USBDetect] 检测器配置: {}, 敌方: {}", detector_cfg, enemy_str);

  UsbCamera camera(guide_config.camera);
  const std::string window_name = "usb_detect";
  if (show) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  }

  uint64_t last_seq = 0;
  int frame_index = 0;
  while (!g_stop_requested.load()) {
    cv::Mat bgr;
    std::chrono::steady_clock::time_point t;
    if (!camera.wait_new_frame(bgr, t, last_seq, std::chrono::milliseconds(500)) || bgr.empty()) {
      continue;
    }
    ++frame_index;

    cv::Mat input = bgr;
    if (guide_config.swap_rb) {
      cv::cvtColor(bgr, input, cv::COLOR_BGR2RGB);
    }
    const auto armors = detector.detect(input);
    const double cx = guide_config.cx >= 0.0 ? guide_config.cx : bgr.cols / 2.0;
    const double fx = resolve_fx(guide_config, bgr.cols);
    const int selected = select_target(armors, enemy, cx);
    const double alpha =
      selected >= 0 ? std::atan((armors[selected].center.x - cx) / fx) : 0.0;

    std::ostringstream oss;
    oss << "[USBDetect] frame " << frame_index << ": " << armors.size() << " armor(s)";
    for (const auto & a : armors) {
      oss << " | " << armor_auto_aim::armor_name_to_string(a.name) << " "
          << (a.color == armor_auto_aim::Color::red ? "red" : "blue") << " "
          << (a.type == armor_auto_aim::ArmorType::small ? "small" : "big")
          << " conf=" << a.confidence;
    }
    if (selected >= 0) {
      oss << " | 引导目标偏角 alpha=" << alpha / kDeg << "°";
    }
    if (!armors.empty() || frame_index % 30 == 0) {
      utils::logger()->info(oss.str());
    }

    if (show) {
      cv::Mat canvas = bgr.clone();
      draw_vertical_line(canvas, cx, utils::colors::YELLOW);
      for (int i = 0; i < static_cast<int>(armors.size()); ++i) {
        const auto & a = armors[i];
        const cv::Scalar color = i == selected ? utils::colors::GREEN : utils::colors::WHITE;
        if (a.points.size() == 4) {
          utils::draw_quadrangle_with_corners(canvas, a.points, color, 2, 3);
        }
        utils::draw_detection_label(canvas, armor_label(a), a.center, color, 0.6, 2);
      }
      utils::draw_frame_number(canvas, frame_index);
      cv::imshow(window_name, canvas);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') {
        break;
      }
    }
  }

  if (show) {
    cv::destroyWindow(window_name);
  }
  utils::logger()->info("[USBDetect] 已退出");
  return 0;
}

}  // namespace Application

int main(int argc, char ** argv)
{
  std::string cfg_path = std::filesystem::current_path().string() + "/src/config/usb_camera.yaml";
  bool show = true;
  bool detect_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--ros-args") {
      break;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "用法: ros2 run test_pipeline usb_detect_node [usb-camera-yaml] "
                   "[--no-show] [--detect-only]\n";
      return 0;
    } else if (arg == "--no-show") {
      show = false;
    } else if (arg == "--detect-only") {
      detect_only = true;
    } else if (!arg.empty() && arg[0] != '-') {
      cfg_path = arg;
    }
  }

  std::signal(SIGINT, Application::handle_signal);

  try {
    const auto guide_config = Application::UsbGuideConfig::load(cfg_path);
    utils::logger()->info("[UsbGuide] 配置: {}", cfg_path);
    if (detect_only) {
      return Application::run_detect_only(guide_config, show);
    }

    rclcpp::init(argc, argv);
    std::signal(SIGINT, Application::handle_signal);  // rclcpp::init 会覆盖 SIGINT 处理
    int ret = 1;
    {
      Application::UsbGuideApp app(
        app_config::AppConfig::load(guide_config.pipeline_config), guide_config, show);
      ret = app.run();
    }
    rclcpp::shutdown();
    return ret;
  } catch (const std::exception & e) {
    utils::logger()->error("[UsbGuide] 程序异常终止: {}", e.what());
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 1;
}
