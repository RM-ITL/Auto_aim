#ifndef VIDEO_NODE_HPP_
#define VIDEO_NODE_HPP_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

#include "app_config/app_config.hpp"
#include "armor.hpp"
#include "detect_node.hpp"
#include "solver_node.hpp"
#include "tracker.hpp"
#include "planner.hpp"
#include "video.hpp"

#include "autoaim_msgs/msg/debug.hpp"
#include "autoaim_msgs/msg/orienta.hpp"
#include "autoaim_msgs/msg/target.hpp"

namespace Application
{

// 视频离线自瞄节点：从视频文件读帧，跑「检测 → 解算 → 跟踪(EKF) → 规划」整链，
// 把云台转动角度等数据发到 ROS2 Topic，供 PlotJuggler 曲线订阅。
//
// 与实车链路(deep_node)的差异（都在本文件内部消化，便于对照）：
//   - 无 IMU：相机当作静止云台，姿态恒为单位四元数（world 系 == gimbal 系）。
//   - 无相机/下位机/串口：图像来自 utils::Video，弹速用命令行常量。
//   - 时间轴用「帧号 / fps」合成（而非 steady_clock::now()），保证 EKF 的 dt 等于
//     视频真实帧间隔，速度/角速度估计不被处理耗时污染；规划的系统延迟前瞻也在该
//     合成时间轴上做，整条链路离线可复现、与播放速度无关。
class VideoApp
{
public:
  struct Options
  {
    std::string config_path;       // YAML 配置（内参/检测器/跟踪器/规划器参数来源）
    std::string video_path;        // 空 => 用 app_config.video.video_path
    double fps = 0.0;              // <=0 => 用视频自带 FPS（再兜底 30）
    double bullet_speed = 22.0;    // 规划用弹速(m/s)，Planner 内部会裁剪到 [10,25]
    // 敌方颜色：""=用配置原值并开启颜色过滤；"red"/"blue"=覆盖配置再过滤；
    // "any"=关闭颜色过滤(离线视频色标常不可靠时用)。
    std::string enemy_color;
    bool show = true;              // 是否弹可视化窗口
    bool realtime = true;          // 是否按 fps 节流播放（利于 PlotJuggler 实时看曲线）
  };

  VideoApp(const app_config::AppConfig & app_config, const Options & options);
  ~VideoApp();

  int run();
  void request_stop();

private:
  using TargetVariant = plan::TargetVariant;

  void publish_orientation_identity();
  void publish_idle();
  void publish_target(const TargetVariant & target_est);
  void publish_debug(
    const std::optional<TargetVariant> & target_est, const plan::Plan & plan, bool has_target);

  void draw(
    cv::Mat & bgr, const std::vector<armor_auto_aim::Visualization> & reprojected,
    int frame_index, double fps_est, const std::string & state, const plan::Plan & plan,
    bool has_target);

  app_config::AppConfig app_config_;
  Options options_;

  std::unique_ptr<utils::Video> video_;
  std::unique_ptr<armor_auto_aim::Detector> detector_;
  std::unique_ptr<solver::Solver> solver_;
  solver::YawOptimizer * yaw_optimizer_{nullptr};
  std::unique_ptr<tracker::Tracker> tracker_;
  std::unique_ptr<plan::Planner> planner_;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<autoaim_msgs::msg::Debug>::SharedPtr debug_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::Target>::SharedPtr target_pub_;
  rclcpp::Publisher<autoaim_msgs::msg::Orienta>::SharedPtr orientation_pub_;

  // 规划的系统延迟前瞻参数：从 app_config.planner 读出，在合成时间轴上复刻
  // Planner::plan(optional) 的「now()+lead」逻辑（离线不用墙钟）。
  double decision_speed_{0.0};
  double high_delay_{0.0};
  double low_delay_{0.0};
  double shoot_delay_{0.0};

  bool use_enemy_color_{true};  // 由 options_.enemy_color 推导；"any" => false

  std::string window_name_{"video_autoaim"};
  cv::Point center_point_{640, 384};

  std::atomic<bool> quit_{false};
};

}  // namespace Application

#endif  // VIDEO_NODE_HPP_
