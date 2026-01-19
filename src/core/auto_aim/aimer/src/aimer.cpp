#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "logger.hpp"
#include "math_tools.hpp"
#include "trajectory.hpp"

namespace aimer
{
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["Aimer"]["yaw_offset"].as<double>() / 57.3;        // degree to rad
  pitch_offset_ = yaml["Aimer"]["pitch_offset"].as<double>() / 57.3;    // degree to rad
  comming_angle_ = yaml["Aimer"]["comming_angle"].as<double>() / 57.3;  // degree to rad
  leaving_angle_ = yaml["Aimer"]["leaving_angle"].as<double>() / 57.3;  // degree to rad
  high_speed_delay_time_ = yaml["Aimer"]["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["Aimer"]["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["Aimer"]["decision_speed"].as<double>();
  if (yaml["Aimer"]["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["Aimer"]["left_yaw_offset"].as<double>() / 57.3;    // degree to rad
    right_yaw_offset_ = yaml["Aimer"]["right_yaw_offset"].as<double>() / 57.3;  // degree to rad
    utils::logger()->info("[Aimer] successfully loading shootmode");
  }
}

io::GimbalCommand Aimer::aim(
  std::list<predict::Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  if (targets.empty()) return {false, false, 0, 0};
  auto target = targets.front();

  auto ekf = target.ekf();
  double delay_time =
    target.ekf_x()[7] > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  if (bullet_speed < 14) bullet_speed = 23;

  // 考虑detecor和tracker所消耗的时间，此外假设aimer的用时可忽略不计
  auto future = timestamp;
  if (to_now) {
    double dt;
    dt = utils::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  else {
    auto dt = 0.005 + delay_time;  //detector-aimer耗时0.005+发弹延时0.1
    // utils::logger()->info("dt is {:.4f} second", dt);
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  auto aim_point0 = choose_aim_point(target);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    // utils::logger()->debug("Invalid aim_point0.");
    return {false, false, 0, 0};
  }

  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  utils::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
  if (trajectory0.unsolvable) {
    utils::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  // 迭代求解飞行时间 (最多10次，收敛条件：相邻两次fly_time差 <0.001)
  [[maybe_unused]] bool converged = false;
  double prev_fly_time = trajectory0.fly_time;
  utils::Trajectory current_traj = trajectory0;
  std::vector<predict::Target> iteration_target(10, target);  // 创建10个目标副本用于迭代预测

  for (int iter = 0; iter < 10; ++iter) {
    // 预测目标在 future + prev_fly_time 时刻的位置
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    // 计算瞄准点
    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    // 计算新弹道
    Eigen::Vector3d xyz = aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = utils::Trajectory(bullet_speed, d, xyz.z());

    // 检查弹道是否可解
    if (current_traj.unsolvable) {
      utils::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    // 检查收敛条件
    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      converged = true;
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  // 计算最终角度
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);  //世界坐标系下pitch向上为负
  return {true, false, static_cast<float>(yaw), static_cast<float>(pitch)};
}

// io::Command Aimer::aim(
//   std::list<predict::Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
//   io::ShootMode shoot_mode, bool to_now)
// {
//   double yaw_offset;
//   if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
//     yaw_offset = left_yaw_offset_.value();
//   } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
//     yaw_offset = right_yaw_offset_.value();
//   } else {
//     yaw_offset = yaw_offset_;
//   }

//   auto command = aim(targets, timestamp, bullet_speed, to_now);
//   command.yaw = command.yaw - yaw_offset_ + yaw_offset;

//   return command;
// }

AimPoint Aimer::choose_aim_point(const predict::Target & target)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  const auto armor_num = static_cast<int>(armor_xyza_list.size());
  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  if (!target.jumped) return {true, armor_xyza_list[0]};

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 如果delta_angle为0，则该装甲板中心和整车中心的连线在世界坐标系的xy平面过原点
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = utils::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // 不考虑小陀螺
  if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost) {
    // 选择在可射击范围内的装甲板
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }
    // 绝无可能
    if (id_list.empty()) {
      utils::logger()->warn("Empty id list!");
      return {false, armor_xyza_list[0]};
    }

    // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
    if (id_list.size() > 1) {
      int id0 = id_list[0], id1 = id_list[1];

      // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
      if (lock_id_ != id0 && lock_id_ != id1)
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

      return {true, armor_xyza_list[lock_id_]};
    }

    // 只有一个装甲板在可射击范围内时，退出锁定模式
    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  double coming_angle, leaving_angle;
  if (target.name == ArmorName::outpost) {
    coming_angle = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  } else {
    coming_angle = comming_angle_;
    leaving_angle = leaving_angle_;
  }

  // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

// ============ 新增：TargetVariant 接口实现 ============

io::GimbalCommand Aimer::aim(
  std::optional<TargetVariant> target, std::chrono::steady_clock::time_point timestamp,
  double bullet_speed, bool to_now)
{
  if (!target.has_value()) return {0, 0, 0};

  // 使用 std::visit 分发到对应的实现
  return std::visit(
    [this, timestamp, bullet_speed, to_now](auto & t) {
      return this->aim_single_target(t, timestamp, bullet_speed, to_now);
    },
    target.value());
}

io::GimbalCommand Aimer::aim_single_target(
  const predict::Target & target, std::chrono::steady_clock::time_point timestamp,
  double bullet_speed, bool to_now)
{
  // 复用原有的实现逻辑
  std::list<predict::Target> targets;
  targets.push_back(target);
  return aim(targets, timestamp, bullet_speed, to_now);
}

io::GimbalCommand Aimer::aim_single_target(
  const predict::OutpostTarget & target, std::chrono::steady_clock::time_point timestamp,
  double bullet_speed, bool to_now)
{
  // OutpostTarget 的实现逻辑（类似 Target 的实现）
  auto ekf = target.ekf();
  double delay_time =
    target.ekf_x()[7] > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  if (bullet_speed < 14) bullet_speed = 23;

  // 创建可变副本用于预测
  predict::OutpostTarget mutable_target = target;

  // 考虑detector和tracker所消耗的时间
  auto future = timestamp;
  if (to_now) {
    double dt;
    dt = utils::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    mutable_target.predict(future);
  } else {
    auto dt = 0.005 + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    mutable_target.predict(future);
  }

  auto aim_point0 = choose_aim_point(mutable_target);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    return {false, false, 0, 0};
  }

  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  utils::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
  if (trajectory0.unsolvable) {
    utils::logger()->debug(
      "[Aimer] Unsolvable trajectory0 for outpost: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  // 迭代求解飞行时间
  [[maybe_unused]] bool converged = false;
  double prev_fly_time = trajectory0.fly_time;
  utils::Trajectory current_traj = trajectory0;
  std::vector<predict::OutpostTarget> iteration_target(10, mutable_target);

  for (int iter = 0; iter < 10; ++iter) {
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    Eigen::Vector3d xyz = aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = utils::Trajectory(bullet_speed, d, xyz.z());

    if (current_traj.unsolvable) {
      utils::logger()->debug(
        "[Aimer] Unsolvable trajectory for outpost in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}",
        iter + 1, bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      converged = true;
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  // 计算最终角度
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);
  return {true, false, static_cast<float>(yaw), static_cast<float>(pitch)};
}

AimPoint Aimer::choose_aim_point(const predict::OutpostTarget & target)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  const auto armor_num = static_cast<int>(armor_xyza_list.size());

  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  if (!target.jumped) return {true, armor_xyza_list[0]};

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = utils::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // 前哨站特殊处理
  double coming_angle = 70 / 57.3;
  double leaving_angle = 30 / 57.3;

  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

}  // namespace aimer
