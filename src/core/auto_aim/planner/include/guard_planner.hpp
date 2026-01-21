#ifndef AUTO_AIM__GUARD_PLANNER_HPP
#define AUTO_AIM__GUARD_PLANNER_HPP

#include <Eigen/Dense>
#include <optional>
#include <variant>

#include "planner.hpp"  // 复用 Plan 结构体和 TargetVariant

namespace guard
{

// 装甲板射击窗口信息
struct ArmorWindow
{
  int id;                    // 装甲板ID
  double facing_angle;       // 朝向角（装甲板法线与枪口连线的夹角）
  double distance;           // 到枪口的距离
  Eigen::Vector3d position;  // 世界坐标位置
  bool in_window;            // 是否在射击窗口内
  double predicted_angle;    // 预测子弹到达时的朝向角
};

// 守株待兔规划器
// 策略：云台指向目标旋转中心，等待装甲板转入射击窗口时开火
class GuardPlanner
{
public:
  // 调试信息
  Eigen::Vector4d debug_xyza;
  ArmorWindow debug_best_armor;

  GuardPlanner(const std::string & config_path);

  // 主接口：与Planner保持一致
  plan::Plan plan(predict::Target target, double bullet_speed);
  plan::Plan plan(predict::OutpostTarget target, double bullet_speed);
  plan::Plan plan(plan::TargetVariant target, double bullet_speed);
  plan::Plan plan(std::optional<plan::TargetVariant> target, double bullet_speed);

private:
  // 配置参数
  double yaw_offset_;           // 相机-枪管yaw偏差
  double pitch_offset_;         // pitch补偿
  double window_angle_;         // 射击窗口半角 (rad)
  double spin_threshold_;       // 判断高速旋转的角速度阈值 (rad/s)
  double fire_angle_thresh_;    // 射击时的最大朝向角阈值
  bool require_approaching_;    // 是否要求装甲板正在转入

  // 延迟补偿参数
  double decision_speed_;           // rad/s，高低速判断阈值
  double high_speed_delay_time_;    // s，高速目标系统延迟
  double low_speed_delay_time_;     // s，低速目标系统延迟

  // 迭代收敛参数
  int max_fly_time_iterations_;         // 飞行时间迭代最大次数
  double fly_time_convergence_thresh_;  // s，收敛阈值

  // 计算装甲板的射击窗口状态
  template <typename TargetType>
  std::vector<ArmorWindow> compute_armor_windows(
    const TargetType & target,
    double bullet_speed,
    double delay_time) const;

  // 获取系统延迟时间（根据目标角速度）
  template <typename TargetType>
  double get_delay_time(const TargetType & target) const;

  // 找到最佳射击目标（窗口内最近且朝向最正的）
  std::optional<ArmorWindow> find_best_armor(
    const std::vector<ArmorWindow> & windows) const;

  // 计算瞄准角度
  Eigen::Vector2d compute_aim_angles(
    const Eigen::Vector3d & target_pos,
    double bullet_speed) const;

  // 计算旋转中心的瞄准角度
  template <typename TargetType>
  Eigen::Vector2d compute_center_aim(
    const TargetType & target,
    double bullet_speed) const;

  // 判断是否应该使用守株待兔模式
  template <typename TargetType>
  bool should_use_guard_mode(const TargetType & target) const;

  // 核心规划逻辑
  template <typename TargetType>
  plan::Plan plan_impl(TargetType target, double bullet_speed);
};

}  // namespace guard

#endif  // AUTO_AIM__GUARD_PLANNER_HPP
