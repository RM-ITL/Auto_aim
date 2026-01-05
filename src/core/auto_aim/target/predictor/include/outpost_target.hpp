#ifndef AUTO_AIM__OUTPOST_TARGET_HPP
#define AUTO_AIM__OUTPOST_TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <set>
#include <vector>

#include "module/solver.hpp"
#include "solver_node.hpp"
#include "full_model.hpp"
#include "armor.hpp"
#include "logger.hpp"

namespace predict
{

// 前哨站专用目标估计器
// 状态向量 (11维): x = [cx, vx, cy, vy, cz, vz, θ, ω, r, h1, h2]
// ID=0: 角度θ, 高度cz (基准)
// ID=1: 角度θ+120°, 高度cz+h1
// ID=2: 角度θ+240°, 高度cz+h2
class OutpostTarget
{
public:
  armor_auto_aim::ArmorName name = armor_auto_aim::ArmorName::outpost;
  armor_auto_aim::ArmorType armor_type = armor_auto_aim::ArmorType::small;
  armor_auto_aim::ArmorPriority priority = armor_auto_aim::ArmorPriority::fifth;

  bool jumped = false;
  int last_id = 0;

  OutpostTarget() = default;

  OutpostTarget(
    const solver::Armor_pose & armor_pose,
    std::chrono::steady_clock::time_point t,
    double radius,
    Eigen::VectorXd P0_dig);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const solver::Armor_pose & armor_pose);

  Eigen::VectorXd ekf_x() const;
  const motion_model::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;
  bool convergened();
  bool h_converged() const;  // h1/h2是否收敛
  bool isinit = false;
  bool checkinit();

  const std::set<int> & observed_ids() const { return observed_ids_; }

private:
  static constexpr int ARMOR_NUM = 3;

  // 收敛阈值
  static constexpr double H_CONVERGENCE_THRESHOLD = 0.02;  // h1/h2协方差阈值
  static constexpr double P_DIVERGENCE_THRESHOLD = 100.0;  // 发散协方差阈值
  static constexpr double H_MAX_REASONABLE = 0.25;          // h最大合理值(米)

  int switch_count_ = 0;
  int update_count_ = 0;

  bool is_switch_ = false;
  bool is_converged_ = false;
  bool is_h_converged_ = false;  // h1/h2收敛标志

  motion_model::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  std::set<int> observed_ids_;  // 已观测到的ID

  // 用于角度跳变检测
  double last_obs_yaw_ = 0.0;   // 上一次观测的yaw角
  int current_id_ = 0;          // 当前跟踪的装甲板ID

  // ID确定逻辑（分阶段）
  int determine_armor_id(double obs_z, double obs_yaw);
  int match_by_height(double obs_z) const;
  int match_by_yaw_jump(double yaw_jump) const;
  double get_predicted_z(int id) const;

  // 5维观测更新 [yaw, pitch, distance, angle, z_armor]
  void update_ypdaz(const solver::Armor_pose & armor_pose, int id);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace predict

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
