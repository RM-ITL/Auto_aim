#include "video_node.hpp"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <list>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "app_config/app_config.hpp"
#include "draw_tools.hpp"
#include "logger.hpp"
#include "math_tools.hpp"
#include "target.hpp"

namespace Application
{
namespace
{
std::atomic<bool> g_stop_requested{false};

void handle_signal(int)
{
  g_stop_requested.store(true);
}

// 探测视频帧率：优先命令行，其次视频自带，最后兜底 30。
double resolve_fps(const std::string & path, double override_fps)
{
  if (override_fps > 0.0) return override_fps;
  cv::VideoCapture probe(path);
  double fps = probe.isOpened() ? probe.get(cv::CAP_PROP_FPS) : 0.0;
  probe.release();
  if (!(fps > 1.0) || fps > 1000.0) {
    utils::logger()->warn("[VideoAutoAim] 视频 FPS 异常({:.3f})，回退为 30", fps);
    fps = 30.0;
  }
  return fps;
}

}  // namespace

VideoApp::VideoApp(const app_config::AppConfig & app_config, const Options & options)
: app_config_(app_config), options_(options)
{
  node_ = std::make_shared<rclcpp::Node>("video_autoaim_node");
  debug_pub_ = node_->create_publisher<autoaim_msgs::msg::Debug>("debug", rclcpp::QoS(10));
  target_pub_ = node_->create_publisher<autoaim_msgs::msg::Target>("target", rclcpp::QoS(10));
  orientation_pub_ =
    node_->create_publisher<autoaim_msgs::msg::Orienta>("orientation", rclcpp::QoS(10));

  const std::string video_path =
    options_.video_path.empty() ? app_config_.video.video_path : options_.video_path;
  utils::logger()->info("[VideoAutoAim] 视频源: {}", video_path);
  video_ = std::make_unique<utils::Video>(video_path);

  // 敌方颜色处理：any=>关闭过滤；red/blue=>覆盖配置后仍过滤；空=>用配置原值。
  if (options_.enemy_color == "any" || options_.enemy_color == "none") {
    use_enemy_color_ = false;
    utils::logger()->info("[VideoAutoAim] 关闭敌方颜色过滤(接受任意颜色装甲板)");
  } else if (options_.enemy_color == "red" || options_.enemy_color == "blue") {
    app_config_.tracker.enemy_color = options_.enemy_color;
    utils::logger()->info("[VideoAutoAim] 敌方颜色覆盖为: {}", options_.enemy_color);
  }

  detector_ = std::make_unique<armor_auto_aim::Detector>(app_config_.detector);
  solver_ = std::make_unique<solver::Solver>(app_config_.solver);
  yaw_optimizer_ = solver_->getYawOptimizer();
  tracker_ = std::make_unique<tracker::Tracker>(app_config_.tracker, *solver_);
  planner_ = std::make_unique<plan::Planner>(app_config_.planner);

  // 复刻 Planner::plan(optional) 的系统延迟前瞻参数（离线在合成时间轴上用）。
  decision_speed_ = app_config_.planner.decision_speed;
  high_delay_ = app_config_.planner.high_speed_delay_time;
  low_delay_ = app_config_.planner.low_speed_delay_time;
  shoot_delay_ = app_config_.planner.shoot_delay;

  utils::logger()->info(
    "[VideoAutoAim] 初始化完成 | config={} | bullet_speed={:.1f} | 无 IMU(静止云台假设)",
    app_config_.source_path, options_.bullet_speed);
}

VideoApp::~VideoApp()
{
  request_stop();
}

void VideoApp::request_stop() { quit_.store(true); }

void VideoApp::publish_orientation_identity()
{
  // 无 IMU：发单位四元数，PlotJuggler 侧始终有一条姿态基线可对照。
  autoaim_msgs::msg::Orienta msg;
  msg.w = 1.0F;
  msg.x = 0.0F;
  msg.y = 0.0F;
  msg.z = 0.0F;
  orientation_pub_->publish(msg);
}

void VideoApp::publish_idle()
{
  autoaim_msgs::msg::Debug msg;  // 全部默认 0/false：无目标时也保证曲线连续
  debug_pub_->publish(msg);
}

void VideoApp::publish_target(const TargetVariant & target_est)
{
  autoaim_msgs::msg::Target msg;
  std::visit(
    [&msg](const auto & t) {
      const auto & ekf = t.ekf();
      msg.x = static_cast<float>(ekf.x[0]);
      msg.vx = static_cast<float>(ekf.x[1]);
      msg.y = static_cast<float>(ekf.x[2]);
      msg.vy = static_cast<float>(ekf.x[3]);
      msg.z = static_cast<float>(ekf.x[4]);
      msg.vz = static_cast<float>(ekf.x[5]);
    },
    target_est);
  target_pub_->publish(msg);
}

void VideoApp::publish_debug(
  const std::optional<TargetVariant> & target_est, const plan::Plan & plan, bool has_target)
{
  // 静止云台：当前云台角取解算器的世界姿态(≈0)，offset 即等于规划角本身。
  const auto gimbal = solver_->getCurrentAngles();
  const float gimbal_yaw = static_cast<float>(gimbal.yaw);
  const float gimbal_pitch = static_cast<float>(gimbal.pitch);

  autoaim_msgs::msg::Debug msg;
  msg.enable_control = plan.control;
  msg.fire = plan.fire;
  msg.yaw = plan.yaw;
  msg.pitch = plan.pitch;
  msg.yaw_gimbal = gimbal_yaw;
  msg.pitch_gimbal = gimbal_pitch;
  msg.yaw_offest = plan.yaw - gimbal_yaw;
  msg.pitch_offset = plan.pitch - gimbal_pitch;
  msg.yaw_vel = plan.yaw_vel;
  msg.pitch_vel = plan.pitch_vel;
  msg.yaw_acc = plan.yaw_acc;
  msg.pitch_acc = plan.pitch_acc;
  msg.bullet_speed = static_cast<float>(options_.bullet_speed);
  msg.target_valid = has_target;
  msg.aim_plate_index = plan.aim_plate_index;
  msg.target_yaw = plan.target_yaw;

  if (target_est.has_value()) {
    std::visit(
      [&msg](const auto & t) {
        const auto x = t.ekf_x();
        msg.theta = static_cast<float>(x[6]);
        msg.omega = static_cast<float>(x[7]);
        msg.center_yaw = static_cast<float>(std::atan2(x[2], x[0]));
        msg.radius = static_cast<float>(x[8]);
      },
      target_est.value());
  }

  // 原始(未经 EKF)视觉观测，便于与滤波后的 target_yaw 对拍。
  const auto raw_pose = solver_->getLastpose();
  msg.raw_armor_yaw = static_cast<float>(raw_pose.world_spherical.yaw);
  msg.raw_gimbal_yaw = gimbal_yaw;
  msg.raw_armor_valid = solver_->getLastPnPResult().success;

  debug_pub_->publish(msg);
}

void VideoApp::draw(
  cv::Mat & bgr, const std::vector<armor_auto_aim::Visualization> & reprojected, int frame_index,
  double fps_est, const std::string & state, const plan::Plan & plan, bool has_target)
{
  // detector 的可视化按 RGB 口径画框，这里统一在 RGB 上绘制再转回 BGR 显示。
  cv::Mat canvas;
  cv::cvtColor(bgr, canvas, cv::COLOR_BGR2RGB);
  detector_->visualize_results(canvas, reprojected, center_point_, frame_index);
  cv::cvtColor(canvas, bgr, cv::COLOR_RGB2BGR);

  const auto deg = [](float rad) { return rad * 180.0F / static_cast<float>(CV_PI); };
  utils::draw_text(
    bgr, cv::format("fps=%.1f  state=%s", fps_est, state.c_str()), {10, 60},
    utils::colors::CYAN, 0.6, 2);
  if (has_target && plan.control) {
    utils::draw_text(
      bgr,
      cv::format(
        "aim yaw=%.2f deg pitch=%.2f deg  fire=%d", deg(plan.yaw), deg(plan.pitch),
        static_cast<int>(plan.fire)),
      {10, 90}, utils::colors::YELLOW, 0.6, 2);
  } else {
    utils::draw_text(bgr, "no target", {10, 90}, utils::colors::RED, 0.6, 2);
  }
}

int VideoApp::run()
{
  const std::string video_path =
    options_.video_path.empty() ? app_config_.video.video_path : options_.video_path;
  const double fps = resolve_fps(video_path, options_.fps);
  const double frame_dt = 1.0 / fps;
  utils::logger()->info(
    "[VideoAutoAim] 回放 fps={:.3f} (帧间隔={:.1f}ms) realtime={} show={}", fps, frame_dt * 1000.0,
    options_.realtime, options_.show);

  if (options_.show) {
    try {
      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    } catch (const cv::Exception & e) {
      utils::logger()->warn("[VideoAutoAim] 创建窗口失败，转为无窗口模式: {}", e.what());
      options_.show = false;
    }
  }

  int frame_index = 0;
  bool center_set = false;
  auto play_start = std::chrono::steady_clock::now();
  auto last_fps_log = play_start;
  int fps_count = 0;
  int det_count_window = 0;
  double fps_est = fps;

  while (!quit_.load() && !g_stop_requested.load()) {
    cv::Mat frame;
    utils::Video::Clock::time_point stamp;
    video_->read(frame, stamp);
    if (frame.empty()) {
      utils::logger()->info("[VideoAutoAim] 视频读取结束");
      break;
    }

    if (!center_set) {
      center_point_ = cv::Point(frame.cols / 2, frame.rows / 2);
      center_set = true;
    }

    // 合成时间轴：帧号/fps。EKF 的 dt 严格等于视频帧间隔，不随处理耗时漂移。
    const double t_sec = frame_index * frame_dt;
    const auto t_point = std::chrono::steady_clock::time_point(
      std::chrono::nanoseconds(static_cast<std::int64_t>(t_sec * 1e9)));

    // 无 IMU：单位四元数当作静止云台姿态。
    solver_->updateIMU(Eigen::Quaterniond::Identity(), t_sec);
    publish_orientation_identity();

    // 检测器输入与实车一致：BGR -> RGB。
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    auto armors = detector_->detect(rgb);

    det_count_window += static_cast<int>(armors.size());

    std::list<armor_auto_aim::Armor> armor_list(armors.begin(), armors.end());
    auto targets = tracker_->track(armor_list, t_point, use_enemy_color_);

    std::vector<armor_auto_aim::Visualization> reprojected;
    plan::Plan plan;
    bool has_target = !targets.empty();

    if (has_target) {
      // tv_est：跟踪器刚更新到本帧(t_point)的滤波估计，用于发布 EKF 状态。
      TargetVariant tv_est = targets.front();
      publish_target(tv_est);

      // tv_plan：拷贝一份做「系统延迟前瞻」后再交规划器(其内部再补弹丸飞行时间)。
      TargetVariant tv_plan = tv_est;
      const double omega = std::visit(
        [](auto & t) { return std::abs(t.ekf_x()[7]); }, tv_plan);
      const double delay = omega > decision_speed_ ? high_delay_ : low_delay_;
      const double lead = delay + shoot_delay_;
      const auto future = t_point + std::chrono::nanoseconds(static_cast<std::int64_t>(lead * 1e9));
      std::visit([&future](auto & t) { t.predict(future); }, tv_plan);

      plan = planner_->plan(tv_plan, options_.bullet_speed);
      publish_debug(tv_est, plan, true);

      if (options_.show) {
        // 用滤波估计的整车装甲板世界坐标重投影回像素，画预测框。
        const auto xyza_list = std::visit([](auto & t) { return t.armor_xyza_list(); }, tv_est);
        const auto name = std::visit([](auto & t) { return t.name; }, tv_est);
        const auto type = std::visit([](auto & t) { return t.armor_type; }, tv_est);
        reprojected.reserve(xyza_list.size());
        for (const Eigen::Vector4d & xyza : xyza_list) {
          auto pts = yaw_optimizer_->reproject_armor_out(xyza.head<3>(), xyza[3], type, name);
          if (pts.size() == 4) {
            armor_auto_aim::Visualization vis;
            std::copy(pts.begin(), pts.end(), vis.corners.begin());
            vis.name = name;
            vis.type = type;
            reprojected.push_back(vis);
          }
        }
      }
    } else {
      publish_idle();
    }

    // fps 估计（处理速率）
    ++fps_count;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_fps_log >= std::chrono::seconds(1)) {
      fps_est = fps_count / std::chrono::duration<double>(now - last_fps_log).count();
      last_fps_log = now;
      utils::logger()->info(
        "[VideoAutoAim] 帧#{} 处理{:.1f}fps 近1s检测{}个 状态={} 目标={}", frame_index, fps_est,
        det_count_window, tracker_->state(), has_target);
      fps_count = 0;
      det_count_window = 0;
    }

    if (options_.show) {
      draw(frame, reprojected, frame_index + 1, fps_est, tracker_->state(), plan, has_target);
      cv::imshow(window_name_, frame);
      if ((cv::waitKey(1) & 0xFF) == 27) {  // ESC 退出
        break;
      }
    }

    // 实时节流：让墙钟帧间隔逼近视频帧间隔；处理跟不上时不加速、任其落后。
    if (options_.realtime) {
      const auto target_wall = play_start + std::chrono::nanoseconds(
        static_cast<std::int64_t>((frame_index + 1) * frame_dt * 1e9));
      std::this_thread::sleep_until(target_wall);
    }

    ++frame_index;
  }

  if (options_.show) {
    cv::destroyWindow(window_name_);
  }
  utils::logger()->info("[VideoAutoAim] 共处理 {} 帧", frame_index);
  return 0;
}

}  // namespace Application

namespace
{
std::string resolve_config_path(const std::string & cli_value)
{
  if (!cli_value.empty()) return cli_value;
  return std::filesystem::current_path().string() + "/src/config/standard3.yaml";
}
}  // namespace

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? |                           | 输出命令行参数说明}"
    "{@config-path   | src/config/standard3.yaml | YAML 配置文件路径(内参/检测/跟踪/规划参数)}"
    "{video          |                           | 视频文件路径(覆盖配置里的 Video.video_path)}"
    "{fps            | 0                         | 回放/解算帧率(<=0 用视频自带)}"
    "{bullet-speed   | 22.0                      | 规划用弹速 m/s}"
    "{enemy-color    |                           | red/blue 覆盖敌方色; any 关闭颜色过滤; 空=用配置}"
    "{no-show        | false                     | 不弹可视化窗口(仅发 Topic)}"
    "{no-realtime    | false                     | 不按 fps 节流(尽快处理)}";

  cv::CommandLineParser cli(argc, argv, keys);
  cli.about("video_autoaim: 视频离线自瞄(检测+解算+跟踪+规划)，数据发 Topic 供 PlotJuggler 订阅");
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  Application::VideoApp::Options options;
  options.config_path = resolve_config_path(cli.get<std::string>("@config-path"));
  options.video_path = cli.get<std::string>("video");
  options.fps = cli.get<double>("fps");
  options.bullet_speed = cli.get<double>("bullet-speed");
  options.enemy_color = cli.get<std::string>("enemy-color");
  options.show = !cli.get<bool>("no-show");
  options.realtime = !cli.get<bool>("no-realtime");

  rclcpp::init(argc, argv);
  std::signal(SIGINT, Application::handle_signal);

  int ret = 1;
  try {
    Application::VideoApp app(app_config::AppConfig::load(options.config_path), options);
    ret = app.run();
  } catch (const std::exception & e) {
    utils::logger()->error("[VideoAutoAim] 程序异常终止: {}", e.what());
  }

  rclcpp::shutdown();
  return ret;
}
