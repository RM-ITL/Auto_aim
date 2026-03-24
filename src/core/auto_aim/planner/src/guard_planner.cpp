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
  window_angle_ = utils::read<double>(guard_yaml, "guard_window_angle") / 57.3;
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
    "[GuardPlanner] 初始化完成: 射击窗口={:.1f}度, 射击角度阈值={:.1f}度",
    window_angle_ * 57.3, fire_angle_thresh_ * 57.3);
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

  // 3. 计算Yaw等待角（指向旋转中心 + 平移速度前馈，保持基本不动）
  double yaw_wait = compute_wait_yaw(target, delay_time);

  // 4. 计算所有装甲板的窗口状态
  auto windows = compute_armor_windows(target, bullet_speed, delay_time);
  auto best_armor = find_best_armor(windows);

  // 5. 构建输出
  plan::Plan result;
  result.control = true;

  // Yaw始终指向旋转中心方位角（不追踪装甲板）
  result.yaw = static_cast<float>(yaw_wait);
  result.target_yaw = result.yaw;
  result.yaw_vel = 0;
  result.yaw_acc = 0;

  if (best_armor.has_value()) {
    // 有装甲板在窗口内：Pitch追踪该装甲板高度
    double dist = best_armor->position.template head<2>().norm();
    double pitch = compute_pitch(dist, best_armor->position.z(), bullet_speed);
    result.pitch = static_cast<float>(pitch);
    result.target_pitch = result.pitch;

    // 开火判断：基于装甲板朝向
    bool angle_ok = std::abs(best_armor->predicted_angle) < fire_angle_thresh_;

    bool approaching_ok = true;
    if (require_approaching_) {
      double omega = target.ekf_x()[7];
      approaching_ok = (best_armor->facing_angle * omega) <= 0;
    }

    result.fire = angle_ok && approaching_ok;

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
    // 没有装甲板在窗口内：Pitch指向旋转中心高度，等待
    auto x = target.ekf_x();
    double cx = x[0], cy = x[2], cz = x[4];
    double center_dist = std::sqrt(cx * cx + cy * cy);
    double pitch = compute_pitch(center_dist, cz, bullet_speed);
    result.pitch = static_cast<float>(pitch);
    result.target_pitch = result.pitch;
    result.fire = false;

    // 找最接近窗口的装甲板用于调试
    double min_angle = 1e10;
    for (const auto & w : windows) {
      if (std::abs(w.facing_angle) < std::abs(min_angle)) {
        min_angle = w.facing_angle;
        debug_best_armor = w;
      }
    }
    debug_xyza = Eigen::Vector4d(0, 0, 0, min_angle);
  }

  result.pitch_vel = 0;
  result.pitch_acc = 0;

  return result;
}

template <typename TargetType>
double GuardPlanner::compute_wait_yaw(const TargetType & target, double delay_time) const
{
  auto x = target.ekf_x();
  double cx = x[0], vx = x[1];
  double cy = x[2], vy = x[3];

  // 平移速度前馈：预测旋转中心位置
  double cx_pred = cx + vx * delay_time;
  double cy_pred = cy + vy * delay_time;

  double azim = std::atan2(cy_pred, cx_pred);
  return utils::limit_rad(azim + yaw_offset_);
}

template <typename TargetType>
std::vector<ArmorWindow> GuardPlanner::compute_armor_windows(
  const TargetType & target,
  double bullet_speed,
  double delay_time) const
{
  std::vector<ArmorWindow> windows;
  auto xyza_list = target.armor_xyza_list();
  double omega = target.ekf_x()[7];

  for (size_t i = 0; i < xyza_list.size(); ++i) {
    const auto & xyza = xyza_list[i];
    ArmorWindow w;
    w.id = static_cast<int>(i);
    w.position = xyza.template head<3>();
    w.distance = w.position.template head<2>().norm();

    // 装甲板朝向角（法线方向）
    double armor_angle = xyza[3];
    double armor_normal = utils::limit_rad(armor_angle + M_PI);

    // 枪口到装甲板的方位角
    double azim = std::atan2(w.position.y(), w.position.x());

    // 朝向角：法线方向与枪口连线的夹角
    w.facing_angle = utils::limit_rad(armor_normal - azim - M_PI);

    // 判断是否在射击窗口内
    w.in_window = std::abs(w.facing_angle) < window_angle_;

    // 迭代收敛计算飞行时间
    utils::Trajectory bullet_traj(bullet_speed, w.distance, w.position.z());
    if (!bullet_traj.unsolvable) {
      double fly_time = bullet_traj.fly_time;

      for (int iter = 0; iter < max_fly_time_iterations_; ++iter) {
        double total_delay = delay_time + fly_time;

        double predicted_angle_rad = omega * total_delay;
        double cx = target.ekf_x()[0];
        double cy = target.ekf_x()[2];
        double cz = target.ekf_x()[4];
        double r = target.ekf_x()[8];

        double new_angle = armor_angle + predicted_angle_rad;
        double new_x = cx - r * std::cos(new_angle);
        double new_y = cy - r * std::sin(new_angle);

        double new_dist = std::sqrt(new_x * new_x + new_y * new_y);

        utils::Trajectory new_traj(bullet_speed, new_dist, cz);
        if (new_traj.unsolvable) break;

        double new_fly_time = new_traj.fly_time;
        if (std::abs(new_fly_time - fly_time) < fly_time_convergence_thresh_) {
          fly_time = new_fly_time;
          break;
        }
        fly_time = new_fly_time;
      }

      double total_delay = delay_time + fly_time;
      double delta_angle = omega * total_delay;
      w.predicted_angle = utils::limit_rad(w.facing_angle - delta_angle);
    } else {
      w.predicted_angle = w.facing_angle;
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

    double angle_score = 1.0 - std::abs(w.facing_angle) / window_angle_;
    double dist_score = 1.0 / (1.0 + w.distance);

    double score = angle_score * 0.7 + dist_score * 0.3;

    if (score > best_score) {
      best_score = score;
      best = w;
    }
  }

  return best;
}

double GuardPlanner::compute_pitch(double dist, double z, double bullet_speed) const
{
  utils::Trajectory traj(bullet_speed, dist, z);
  return -traj.pitch - pitch_offset_;
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
template double GuardPlanner::compute_wait_yaw(const predict::Target &, double) const;
template double GuardPlanner::compute_wait_yaw(const predict::OutpostTarget &, double) const;
template double GuardPlanner::get_delay_time(const predict::Target &) const;
template double GuardPlanner::get_delay_time(const predict::OutpostTarget &) const;

}  // namespace guard
