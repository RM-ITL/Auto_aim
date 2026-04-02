#include "aim_planner.hpp"

#include <cmath>
#include <vector>

#include "math_tools.hpp"
#include "trajectory.hpp"
#include "yaml.hpp"

using namespace std::chrono_literals;

namespace plan
{
AimPlanner::AimPlanner(const std::string & config_path)
{
  auto yaml = utils::load(config_path);
  auto planner_yaml = yaml["Planner"];
  yaw_offset_ = utils::read<double>(planner_yaml, "yaw_offset") / 57.3;
  pitch_offset_ = utils::read<double>(planner_yaml, "pitch_offset") / 57.3;
  fire_thresh_ = utils::read<double>(planner_yaml, "fire_thresh");
  decision_speed_ = utils::read<double>(planner_yaml, "decision_speed");
  high_speed_delay_time_ = utils::read<double>(planner_yaml, "high_speed_delay_time");
  low_speed_delay_time_ = utils::read<double>(planner_yaml, "low_speed_delay_time");
  armor_hysteresis_ = utils::read<double>(planner_yaml, "armor_hysteresis");

  // 高速模式参数
  omega_threshold_ = utils::read<double>(planner_yaml, "omega_threshold");
  window_angle_ = utils::read<double>(planner_yaml, "window_angle") / 57.3;  // 转换为弧度

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

// 子弹飞行时间补偿
Plan AimPlanner::plan(predict::Target target, double bullet_speed)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 判断工作模式
  double omega = std::abs(target.ekf_x()[7]);
  ArmorMode mode = get_armor_mode(omega);

  // ========== 高速模式：守株待兔 ==========
  if (mode == ArmorMode::HIGH_SPEED) {
    // 注意：系统延迟补偿已在plan(optional<Target>)中完成，此处不再重复

    // 1. 选择最正对的装甲板
    int armor_id = select_armor_high_speed(target);

    // 2. 计算弹道飞行时间补偿
    auto xyza = target.armor_xyza_list()[armor_id];
    auto dist = xyza.head<2>().norm();
    auto bullet_traj = utils::Trajectory(bullet_speed, dist, xyza.z());
    target.predict(bullet_traj.unsolvable ? 0.0 : bullet_traj.fly_time);

    // 3. 重新选择（预测后位置可能变化）
    armor_id = select_armor_high_speed(target);
    xyza = target.armor_xyza_list()[armor_id];

    debug_xyza = Eigen::Vector4d(xyza.x(), xyza.y(), xyza.z(), xyza[3]);

    // 4. 构建输出
    Plan plan;
    plan.control = true;

    // Yaw：指向当前旋转中心
    plan.yaw = static_cast<float>(compute_yaw_high_speed(target));
    plan.target_yaw = plan.yaw;
    plan.yaw_vel = 0;
    plan.yaw_acc = 0;

    // Pitch：追踪选中装甲板的高度
    plan.pitch = static_cast<float>(compute_pitch_high_speed(target, armor_id, bullet_speed));
    plan.target_pitch = plan.pitch;
    plan.pitch_vel = 0;
    plan.pitch_acc = 0;

    // Fire：基于facing_angle宽松判断，精细控制交给Shooter
    plan.fire = should_fire_high_speed(target, armor_id);

    return plan;
  }

  // ========== 低速模式：TinyMPC轨迹跟踪（原有逻辑） ==========

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  if (target.single_plate_mode) {
    // 单板模式：只用当前观测板估算飞行时间
    auto xyza = target.armor_xyza_list()[target.last_id];
    min_dist = xyza.head<2>().norm();
    xyz = xyza.head<3>();
  } else {
    for (auto & xyza : target.armor_xyza_list()) {
      auto dist = xyza.head<2>().norm();
      if (dist < min_dist) {
        min_dist = dist;
        xyz = xyza.head<3>();
      }
    }
  }
  auto bullet_traj = utils::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    utils::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = utils::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = utils::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}

// 系统延迟时间的补偿
Plan AimPlanner::plan(std::optional<predict::Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed);
}

// variant 重载：从 variant 中提取 Target，OutpostTarget 不处理
Plan AimPlanner::plan(std::optional<std::variant<predict::Target, predict::OutpostTarget>> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  auto* p = std::get_if<predict::Target>(&target.value());
  if (!p) return {false};

  return plan(std::optional<predict::Target>(*p), bullet_speed);
}

void AimPlanner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = utils::load(config_path);
  auto planner_yaml = yaml["Planner"];
  auto max_yaw_acc = utils::read<double>(planner_yaml, "max_yaw_acc");
  auto Q_yaw = utils::read<std::vector<double>>(planner_yaml, "Q_yaw");
  auto R_yaw = utils::read<std::vector<double>>(planner_yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void AimPlanner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = utils::load(config_path);
  auto planner_yaml = yaml["Planner"];
  auto max_pitch_acc = utils::read<double>(planner_yaml, "max_pitch_acc");
  auto Q_pitch = utils::read<std::vector<double>>(planner_yaml, "Q_pitch");
  auto R_pitch = utils::read<std::vector<double>>(planner_yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> AimPlanner::aim(const predict::Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;
  int selected_id = 0;

  if (target.single_plate_mode) {
    // 单板模式：只用当前观测板
    auto xyza = target.armor_xyza_list()[target.last_id];
    xyz = xyza.head<3>();
    yaw = xyza[3];
  } else {
    // 多板模式：找最小距离的板
    for (int i = 0; i < 4; i++) {
      auto xyza = target.armor_xyza_list()[i];
      auto dist = xyza.head<2>().norm();
      if (dist < min_dist) {
        min_dist = dist;
        xyz = xyza.head<3>();
        yaw = xyza[3];
        selected_id = i;
      }
    }

    // 滞后机制：只有新板比旧板好armor_hysteresis_以上，才切换
    if (selected_id != target.last_id) {
      auto old_xyza = target.armor_xyza_list()[target.last_id];
      auto old_dist = old_xyza.head<2>().norm();

      if (min_dist > old_dist * armor_hysteresis_) {
        // 新板不够优，保持旧板
        xyz = old_xyza.head<3>();
        yaw = old_xyza[3];
      }
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = utils::Trajectory(bullet_speed, xyz.head<2>().norm(), xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {utils::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

// 轨迹生成中的连续预测
Trajectory AimPlanner::get_trajectory(predict::Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = utils::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << utils::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

// ============ 高速模式方法实现 ============

AimPlanner::ArmorMode AimPlanner::get_armor_mode(double omega) const
{
  return omega >= omega_threshold_ ? ArmorMode::HIGH_SPEED : ArmorMode::LOW_SPEED;
}

double AimPlanner::compute_facing_angle(const Eigen::Vector4d & xyza, const Eigen::Vector3d & xyz) const
{
  // 装甲板法向量方向（板的朝向角 + π 得到法线方向）
  double armor_normal = utils::limit_rad(xyza[3] + M_PI);
  // 枪口到装甲板的方位角
  double azim = std::atan2(xyz.y(), xyz.x());
  // 朝向角：法线方向与枪口连线的夹角
  return utils::limit_rad(armor_normal - azim - M_PI);
}

int AimPlanner::select_armor_high_speed(const predict::Target & target) const
{
  auto xyza_list = target.armor_xyza_list();
  int best_id = 0;
  double min_facing = 1e10;

  for (int i = 0; i < static_cast<int>(xyza_list.size()); i++) {
    const auto & xyza = xyza_list[i];
    Eigen::Vector3d xyz = xyza.head<3>();
    double facing = std::abs(compute_facing_angle(xyza, xyz));

    if (facing < min_facing) {
      min_facing = facing;
      best_id = i;
    }
  }

  return best_id;
}

double AimPlanner::compute_yaw_high_speed(const predict::Target & target) const
{
  // 直接指向当前旋转中心，不做速度前馈
  auto x = target.ekf_x();
  double cx = x[0], cy = x[2];
  double azim = std::atan2(cy, cx);
  return utils::limit_rad(azim + yaw_offset_);
}

double AimPlanner::compute_pitch_high_speed(
  const predict::Target & target, int armor_id, double bullet_speed) const
{
  auto xyza = target.armor_xyza_list()[armor_id];
  double dist = xyza.head<2>().norm();
  double z = xyza.z();

  utils::Trajectory traj(bullet_speed, dist, z);
  if (traj.unsolvable) {
    // 降级：用旋转中心高度
    auto x = target.ekf_x();
    double center_dist = std::sqrt(x[0] * x[0] + x[2] * x[2]);
    utils::Trajectory center_traj(bullet_speed, center_dist, x[4]);
    return -center_traj.pitch - pitch_offset_;
  }
  return -traj.pitch - pitch_offset_;
}

bool AimPlanner::should_fire_high_speed(const predict::Target & target, int armor_id) const
{
  auto xyza = target.armor_xyza_list()[armor_id];
  Eigen::Vector3d xyz = xyza.head<3>();
  double facing = std::abs(compute_facing_angle(xyza, xyz));

  // 宽松判断：在窗口内就允许开火，精细控制交给Shooter
  return facing < window_angle_;
}

}  // namespace plan