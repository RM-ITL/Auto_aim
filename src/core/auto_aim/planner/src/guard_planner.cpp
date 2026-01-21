#include "guard_planner.hpp"

#include <cmath>
#include <algorithm>

#include "math_tools.hpp"
#include "trajectory.hpp"
#include "yaml.hpp"

namespace guard
{

GuardPlanner::GuardPlanner(const std::string & config_path)
{
  auto yaml = utils::load(config_path);

  // 复用planner的参数
  auto planner_yaml = yaml["Planner"];
  yaw_offset_ = utils::read<double>(planner_yaml, "yaw_offset") / 57.3;
  pitch_offset_ = utils::read<double>(planner_yaml, "pitch_offset") / 57.3;

  // GuardPlanner专用参数
  auto guard_yaml = yaml["GuardPlanner"];
  window_angle_ = utils::read<double>(guard_yaml, "guard_window_angle") / 57.3;  // 转为弧度
  spin_threshold_ = utils::read<double>(guard_yaml, "guard_spin_threshold");
  fire_angle_thresh_ = utils::read<double>(guard_yaml, "guard_fire_angle_thresh") / 57.3;
  require_approaching_ = utils::read<bool>(guard_yaml, "guard_require_approaching");

  // 延迟补偿参数
  decision_speed_ = utils::read<double>(guard_yaml, "decision_speed");
  high_speed_delay_time_ = utils::read<double>(guard_yaml, "high_speed_delay_time");
  low_speed_delay_time_ = utils::read<double>(guard_yaml, "low_speed_delay_time");

  // 迭代收敛参数
  max_fly_time_iterations_ = utils::read<int>(guard_yaml, "max_fly_time_iterations");
  fly_time_convergence_thresh_ = utils::read<double>(guard_yaml, "fly_time_convergence_thresh");

  utils::logger()->info(
    "[GuardPlanner] 初始化完成: 射击窗口={:.1f}度, 高速阈值={:.2f}rad/s, 射击角度阈值={:.1f}度",
    window_angle_ * 57.3, spin_threshold_, fire_angle_thresh_ * 57.3);
  utils::logger()->info(
    "[GuardPlanner] 延迟补偿: 判断阈值={:.1f}rad/s, 高速延迟={:.3f}s, 低速延迟={:.3f}s",
    decision_speed_, high_speed_delay_time_, low_speed_delay_time_);
}

// ============ 公共接口 ============

plan::Plan GuardPlanner::plan(predict::Target target, double bullet_speed)
{
  return plan_impl(target, bullet_speed);
}

plan::Plan GuardPlanner::plan(predict::OutpostTarget target, double bullet_speed)
{
  return plan_impl(target, bullet_speed);
}

plan::Plan GuardPlanner::plan(plan::TargetVariant target, double bullet_speed)
{
  return std::visit([this, bullet_speed](auto & t) {
    return this->plan(t, bullet_speed);
  }, target);
}

plan::Plan GuardPlanner::plan(std::optional<plan::TargetVariant> target, double bullet_speed)
{
  if (!target.has_value()) return {false};
  return plan(*target, bullet_speed);
}

// ============ 核心实现 ============

template <typename TargetType>
plan::Plan GuardPlanner::plan_impl(TargetType target, double bullet_speed)
{
  // 0. 检查弹速
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. 获取系统延迟时间
  double delay_time = get_delay_time(target);

  // 2. 系统延迟补偿：预测目标状态
  target.predict(delay_time);

  // 3. 判断是否使用守株待兔模式
  if (!should_use_guard_mode(target)) {
    // 低速目标：简单跟踪最近装甲板
    auto xyza_list = target.armor_xyza_list();
    Eigen::Vector3d nearest_xyz;
    double min_dist = 1e10;

    for (const auto & xyza : xyza_list) {
      double dist = xyza.template head<2>().norm();
      if (dist < min_dist) {
        min_dist = dist;
        nearest_xyz = xyza.template head<3>();
      }
    }

    auto aim = compute_aim_angles(nearest_xyz, bullet_speed);

    plan::Plan result;
    result.control = true;
    result.fire = true;  // 低速目标直接射击
    result.yaw = aim(0);
    result.pitch = aim(1);
    result.target_yaw = aim(0);
    result.target_pitch = aim(1);
    result.yaw_vel = 0;
    result.pitch_vel = 0;
    result.yaw_acc = 0;
    result.pitch_acc = 0;

    return result;
  }

  // 4. 高速旋转目标：守株待兔模式
  auto windows = compute_armor_windows(target, bullet_speed, delay_time);
  auto best_armor = find_best_armor(windows);

  // 5. 计算云台指向
  plan::Plan result;
  result.control = true;

  if (best_armor.has_value()) {
    // 有装甲板在射击窗口内：瞄准该装甲板
    auto aim = compute_aim_angles(best_armor->position, bullet_speed);

    result.yaw = aim(0);
    result.pitch = aim(1);
    result.target_yaw = aim(0);
    result.target_pitch = aim(1);

    // 射击判断：预测角度也在阈值内
    bool angle_ok = std::abs(best_armor->predicted_angle) < fire_angle_thresh_;

    // 可选：要求装甲板正在转入（facing_angle在减小）
    bool approaching_ok = true;
    if (require_approaching_) {
      double omega = target.ekf_x()[7];
      // 如果omega和facing_angle同号，说明在转入
      approaching_ok = (best_armor->facing_angle * omega) <= 0;
    }

    // 过滤低处装甲板（ID=0和ID=2不开火，ID=1和ID=3是高处装甲板可以开火）
    bool height_ok = (best_armor->id == 1 || best_armor->id == 3);

    result.fire = angle_ok && approaching_ok && height_ok;

    // 更新调试信息
    debug_best_armor = *best_armor;
    debug_xyza = Eigen::Vector4d(
      best_armor->position.x(),
      best_armor->position.y(),
      best_armor->position.z(),
      best_armor->facing_angle);

    if (result.fire) {
      utils::logger()->debug(
        "[GuardPlanner] 射击! 装甲板{}: 朝向角={:.1f}度, 预测角={:.1f}度, 距离={:.2f}m",
        best_armor->id,
        best_armor->facing_angle * 57.3,
        best_armor->predicted_angle * 57.3,
        best_armor->distance);
    }
  } else {
    // 没有装甲板在窗口内：指向旋转中心，等待
    auto center_aim = compute_center_aim(target, bullet_speed);

    result.yaw = center_aim(0);
    result.pitch = center_aim(1);
    result.target_yaw = center_aim(0);
    result.target_pitch = center_aim(1);
    result.fire = false;

    // 找一个最接近窗口的装甲板用于调试
    double min_angle = 1e10;
    for (const auto & w : windows) {
      if (std::abs(w.facing_angle) < std::abs(min_angle)) {
        min_angle = w.facing_angle;
        debug_best_armor = w;
      }
    }
    debug_xyza = Eigen::Vector4d(0, 0, 0, min_angle);
  }

  // 速度和加速度设为0（守株待兔模式不需要复杂的轨迹规划）
  result.yaw_vel = 0;
  result.pitch_vel = 0;
  result.yaw_acc = 0;
  result.pitch_acc = 0;

  return result;
}

template <typename TargetType>
std::vector<ArmorWindow> GuardPlanner::compute_armor_windows(
  const TargetType & target,
  double bullet_speed,
  double delay_time) const
{
  std::vector<ArmorWindow> windows;
  auto xyza_list = target.armor_xyza_list();
  double omega = target.ekf_x()[7];  // 角速度

  for (size_t i = 0; i < xyza_list.size(); ++i) {
    const auto & xyza = xyza_list[i];
    ArmorWindow w;
    w.id = static_cast<int>(i);
    w.position = xyza.template head<3>();
    w.distance = w.position.template head<2>().norm();

    // 装甲板朝向角（法线方向）
    double armor_angle = xyza[3];  // 这是装甲板在世界坐标系的角度
    double armor_normal = utils::limit_rad(armor_angle + M_PI);  // 法线指向外

    // 枪口到装甲板的方位角
    double azim = std::atan2(w.position.y(), w.position.x());

    // 朝向角：法线方向与枪口连线的夹角
    // 如果接近0，说明装甲板正对枪口
    w.facing_angle = utils::limit_rad(armor_normal - azim - M_PI);

    // 判断是否在射击窗口内
    w.in_window = std::abs(w.facing_angle) < window_angle_;

    // 迭代收敛计算飞行时间
    utils::Trajectory bullet_traj(bullet_speed, w.distance, w.position.z());
    if (!bullet_traj.unsolvable) {
      double fly_time = bullet_traj.fly_time;

      // 飞行时间迭代收敛
      for (int iter = 0; iter < max_fly_time_iterations_; ++iter) {
        double total_delay = delay_time + fly_time;

        // 预测装甲板转过的角度后的新位置
        double predicted_angle_rad = omega * total_delay;
        // 基于旋转中心计算预测位置（简化：仅影响距离估计）
        double cx = target.ekf_x()[0];
        double cy = target.ekf_x()[2];
        double cz = target.ekf_x()[4];
        double r = target.ekf_x()[6];  // 旋转半径

        // 预测装甲板的新位置
        double new_angle = armor_angle + predicted_angle_rad;
        double new_x = cx + r * std::cos(new_angle);
        double new_y = cy + r * std::sin(new_angle);
        double new_z = cz;  // z坐标不变

        double new_dist = std::sqrt(new_x * new_x + new_y * new_y);

        // 重新计算弹道
        utils::Trajectory new_traj(bullet_speed, new_dist, new_z);
        if (new_traj.unsolvable) break;

        double new_fly_time = new_traj.fly_time;

        // 检查收敛
        if (std::abs(new_fly_time - fly_time) < fly_time_convergence_thresh_) {
          fly_time = new_fly_time;
          break;
        }
        fly_time = new_fly_time;
      }

      // 使用总延迟计算预测角度
      double total_delay = delay_time + fly_time;
      double delta_angle = omega * total_delay;
      w.predicted_angle = utils::limit_rad(w.facing_angle - delta_angle);
      // 注意：facing_angle减小说明在转向正对
    } else {
      w.predicted_angle = w.facing_angle;  // 无法计算时使用当前值
    }

    windows.push_back(w);
  }

  return windows;
}

std::optional<ArmorWindow> GuardPlanner::find_best_armor(
  const std::vector<ArmorWindow> & windows) const
{
  std::optional<ArmorWindow> best;
  double best_score = -1e10;

  for (const auto & w : windows) {
    if (!w.in_window) continue;

    // 评分：朝向角越小越好，距离越近越好
    // 归一化处理
    double angle_score = 1.0 - std::abs(w.facing_angle) / window_angle_;  // 0~1
    double dist_score = 1.0 / (1.0 + w.distance);  // 距离越近越好

    double score = angle_score * 0.7 + dist_score * 0.3;

    if (score > best_score) {
      best_score = score;
      best = w;
    }
  }

  return best;
}

Eigen::Vector2d GuardPlanner::compute_aim_angles(
  const Eigen::Vector3d & target_pos,
  double bullet_speed) const
{
  double dist = target_pos.head<2>().norm();
  double azim = std::atan2(target_pos.y(), target_pos.x());

  utils::Trajectory bullet_traj(bullet_speed, dist, target_pos.z());

  double yaw = utils::limit_rad(azim + yaw_offset_);
  double pitch = -bullet_traj.pitch - pitch_offset_;

  return Eigen::Vector2d(yaw, pitch);
}

template <typename TargetType>
Eigen::Vector2d GuardPlanner::compute_center_aim(
  const TargetType & target,
  double bullet_speed) const
{
  // 旋转中心坐标
  auto x = target.ekf_x();
  double cx = x[0];
  double cy = x[2];
  double cz = x[4];

  return compute_aim_angles(Eigen::Vector3d(cx, cy, cz), bullet_speed);
}

template <typename TargetType>
bool GuardPlanner::should_use_guard_mode(const TargetType & target) const
{
  double omega = std::abs(target.ekf_x()[7]);
  return omega > spin_threshold_;
}

template <typename TargetType>
double GuardPlanner::get_delay_time(const TargetType & target) const
{
  double omega = std::abs(target.ekf_x()[7]);
  return omega > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
}

// 显式实例化模板
template plan::Plan GuardPlanner::plan_impl(predict::Target, double);
template plan::Plan GuardPlanner::plan_impl(predict::OutpostTarget, double);
template std::vector<ArmorWindow> GuardPlanner::compute_armor_windows(const predict::Target &, double, double) const;
template std::vector<ArmorWindow> GuardPlanner::compute_armor_windows(const predict::OutpostTarget &, double, double) const;
template Eigen::Vector2d GuardPlanner::compute_center_aim(const predict::Target &, double) const;
template Eigen::Vector2d GuardPlanner::compute_center_aim(const predict::OutpostTarget &, double) const;
template bool GuardPlanner::should_use_guard_mode(const predict::Target &) const;
template bool GuardPlanner::should_use_guard_mode(const predict::OutpostTarget &) const;
template double GuardPlanner::get_delay_time(const predict::Target &) const;
template double GuardPlanner::get_delay_time(const predict::OutpostTarget &) const;

}  // namespace guard
