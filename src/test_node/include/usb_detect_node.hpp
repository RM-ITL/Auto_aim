#ifndef USB_DETECT_NODE_HPP_
#define USB_DETECT_NODE_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

#include <rclcpp/rclcpp.hpp>

#include "app_config/app_config.hpp"
#include "armor.hpp"
#include "autoaim_msgs/msg/debug.hpp"
#include "camera.hpp"
#include "detect_node.hpp"
#include "gimbal.hpp"
#include "planner.hpp"
#include "shooter.hpp"
#include "solver_node.hpp"
#include "thread_safe_queue.hpp"
#include "tracker.hpp"

namespace Application
{

using Armors = armor_auto_aim::Armor;
using Visualization = armor_auto_aim::Visualization;

// usb_camera.yaml 解析结果
struct UsbGuideConfig
{
  std::string pipeline_config;   // 主链路 yaml（相机/解算/跟踪/规划/下位机）
  std::string detector_config;   // USB 检测器参数来源，空 = 与 pipeline_config 相同
  YAML::Node camera;             // usb_camera 段：采集参数 + V4L2 控制项
  double hfov_deg{0.0};          // USB 相机水平视场角(度)，fx 未给时用它估算 fx
  double fx{0.0};                // USB 相机焦距(px)，>0 时优先于 hfov_deg
  double cx{-1.0};               // 光心 u(px)，<0 表示取图像宽度一半
  std::string detector_device;   // USB 检测器推理设备，空 = 与 detector_config 相同
  bool swap_rb{false};           // 送检测器前是否 BGR->RGB；检测器要求 BGR，V4L2 帧已是 BGR
  std::string enemy_color;       // 敌方颜色覆盖 "red"/"blue"；空 = 用配置里的 Tracker.enemy_color

  bool guide_enable{false};      // 启动时是否允许 USB 引导（运行中可按 g 切换）
  double yaw_mount{0.0};         // USB 光轴方向对应的云台 yaw(rad)
  bool yaw_mount_relative{true}; // true: yaw_mount 是相对"启动时云台 yaw"的偏移
  double yaw_gain{-1.0};         // 带符号增益：yaw = yaw_mount + yaw_gain * alpha，通常 ±1
  std::optional<double> pitch;   // 引导时的 pitch(rad)，未配置则保持接管时的 pitch
  double max_yaw_error{0.0};     // 引导时下发 yaw 与 gs.yaw 的最大差(rad)，限制转速；0 = 不限
  double guide_timeout_s{0.2};   // USB 引导样本有效期
  double main_hold_s{0.3};       // 主链路丢失目标后，等待这段时间才让 USB 接管
  int min_detect_frames{3};      // USB 连续检测到目标的帧数达到该值才引导

  // 作用于整条下发链路（主链路 + USB 引导）
  bool yaw_continuous{true};     // true: 下发 gs.yaw + 最短角差；false: 限幅到 [-pi, pi]（同 deep_node）
  bool allow_fire{false};        // false 时强制不开火

  static UsbGuideConfig load(const std::string & path);
};

// USB 相机：独立取图线程只保留最新一帧，避免 V4L2 缓冲把旧帧当新帧用
class UsbCamera
{
public:
  explicit UsbCamera(const YAML::Node & cam);
  ~UsbCamera();

  UsbCamera(const UsbCamera &) = delete;
  UsbCamera & operator=(const UsbCamera &) = delete;

  // 等待序号大于 last_seq 的新帧；超时或已停止返回 false
  bool wait_new_frame(
    cv::Mat & img, std::chrono::steady_clock::time_point & timestamp, uint64_t & last_seq,
    std::chrono::milliseconds timeout);
  // 只发停止信号并唤醒等待者，取图线程在析构时 join
  void stop();
  const std::string & device() const { return device_; }

private:
  bool open_device();
  void capture_loop();

  YAML::Node cam_;
  std::string device_;
  bool is_device_{true};                      // false = 视频文件（离线测试，循环播放）
  std::chrono::microseconds file_period_{0};  // 视频文件按原帧率放
  cv::VideoCapture capture_;
  std::mutex mutex_;
  std::condition_variable frame_cv_;
  cv::Mat latest_;
  std::chrono::steady_clock::time_point latest_time_{};
  uint64_t seq_{0};
  std::atomic<bool> quit_{false};
  std::thread thread_;
};

// USB 相机对当前最高优先级敌方目标给出的引导量
struct UsbGuideSample
{
  bool detected{false};   // 本帧有敌方装甲板
  bool valid{false};      // 连续检测帧数达到 min_detect_frames 才为 true，planner 只用 valid 样本
  int detect_count{0};    // 连续检测帧数
  double alpha{0.0};      // 目标相对 USB 光轴的水平偏角(rad)，画面右侧为正
  double yaw{0.0};        // 引导目标 yaw(rad) = yaw_mount + gain * alpha，与 gs.yaw 同一参考
  double fx{0.0};         // 计算 alpha 用的焦距(px)，标定时用来给出 fx 修正建议
  armor_auto_aim::ArmorName name{armor_auto_aim::ArmorName::not_armor};
  std::chrono::steady_clock::time_point time{};
};

struct UsbDebugPacket
{
  cv::Mat bgr;
  std::vector<Armors> armors;
  int selected{-1};       // armors 中被选为引导目标的下标
  double cx{0.0};
  UsbGuideSample sample;
  bool valid{false};
};

struct DebugPacket
{
  cv::Mat rgb_image;      // 与 deep_node 相同：主相机帧 BGR->RGB 后送检测、直接显示
  std::vector<Visualization> reprojected_armors;
  std::string tracker_state;
  bool valid{false};
};

enum class ControlSource { None, Main, Usb };

class UsbGuideApp
{
public:
  UsbGuideApp(
    const app_config::AppConfig & app_config, const UsbGuideConfig & guide_config, bool show);
  ~UsbGuideApp();

  int run();
  void request_stop();

private:
  void start_threads();
  void join_threads();
  void visualization_loop();
  void planner_loop();
  void sender_loop();
  void usb_detect_loop();
  void calibrate_yaw_mount();
  void reset_calibration();
  double guide_yaw(double alpha) const;  // yaw_mount + gain * alpha，限幅到 [-pi, pi]
  plan::Plan make_guide_plan(
    const UsbGuideSample & sample, const io::GimbalState & gs, float pitch_hold) const;
  void publish_debug(const plan::Plan & plan, const io::GimbalState & gs, bool target_valid);
  void update_latest_plan(const plan::Plan & plan);

  struct LatestPlanSnapshot
  {
    plan::Plan plan;
    bool valid{false};
    std::chrono::steady_clock::time_point time{};
  };
  LatestPlanSnapshot latest_plan_snapshot();

  // 主链路组件（与 deep_node 一致）
  std::unique_ptr<camera::Camera> camera_;
  std::unique_ptr<armor_auto_aim::Detector> detector_;
  std::unique_ptr<solver::Solver> solver_;
  solver::YawOptimizer * yaw_optimizer_{nullptr};
  std::unique_ptr<tracker::Tracker> tracker_;
  std::unique_ptr<plan::Planner> planner_;
  std::unique_ptr<io::Gimbal> gimbal_;
  std::unique_ptr<shooter::Shooter> shooter_;
  rclcpp::Node::SharedPtr ros_node_;
  rclcpp::Publisher<autoaim_msgs::msg::Debug>::SharedPtr debug_pub_;

  // USB 引导组件
  UsbGuideConfig guide_config_;
  std::unique_ptr<UsbCamera> usb_camera_;
  std::unique_ptr<armor_auto_aim::Detector> usb_detector_;
  armor_auto_aim::Color enemy_color_{armor_auto_aim::Color::red};
  std::atomic<bool> guide_enabled_{false};
  std::atomic<bool> mount_ready_{false};   // 相对模式下，收到首个云台状态后才可引导
  std::atomic<double> base_yaw_{0.0};      // 相对模式：启动时云台 yaw；绝对模式：0
  std::atomic<double> yaw_mount_{0.0};     // 当前生效的 yaw_mount（含 base_yaw_）
  std::atomic<double> yaw_gain_{-1.0};     // 带符号增益：yaw = yaw_mount + gain * alpha
  std::vector<std::pair<double, double>> calib_samples_;  // (参考 yaw, USB 偏角)，仅可视化线程访问

  // 主链路最近一次有效规划的 target_yaw，标定 yaw_mount 时作参考
  std::mutex main_ref_mutex_;
  double main_ref_yaw_{0.0};
  std::chrono::steady_clock::time_point main_ref_time_{};

  std::mutex guide_mutex_;
  UsbGuideSample guide_sample_;
  std::mutex usb_debug_mutex_;
  UsbDebugPacket usb_debug_;
  std::atomic<ControlSource> control_source_{ControlSource::None};
  std::atomic<double> usb_fps_{0.0};

  tools::ThreadSafeQueue<DebugPacket, true> visualization_queue{2};
  tools::ThreadSafeQueue<std::optional<tracker::TargetVariant>, true> target_queue{1};
  std::mutex latest_plan_mutex_;
  plan::Plan latest_plan_;
  bool latest_plan_valid_{false};
  std::chrono::steady_clock::time_point latest_plan_time_;

  std::atomic<bool> quit_{false};
  std::thread visualization_thread_;
  std::thread planner_thread_;
  std::thread sender_thread_;
  std::thread usb_detect_thread_;

  bool enable_visualization_{true};
  std::string main_window_name_{"main_camera"};
  std::string usb_window_name_{"usb_camera"};
  std::atomic<int> visualization_frame_counter_{0};

  std::chrono::steady_clock::time_point start_time_;

  // 主链路延迟统计（曝光 -> 送入 target_queue），用来观察 USB 检测器抢 CPU 的影响
  std::deque<double> delay_window_;
  const size_t delay_window_size_{100};
  std::chrono::steady_clock::time_point last_delay_log_time_;
};

// 只跑 USB 相机 + 检测器，不连主相机/下位机（原 usb_detect_node 功能）
int run_detect_only(const UsbGuideConfig & guide_config, bool show);

}  // namespace Application

#endif  // USB_DETECT_NODE_HPP_
