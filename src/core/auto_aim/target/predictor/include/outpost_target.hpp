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
  static constexpr double H_MAX_REASONABLE = 0.25;          // h最大合理值(米)
  static constexpr double MATCH_GATE = 10.0;                // 匹配门控阈值（参照wust_vision）

  int switch_count_ = 0;
  int update_count_ = 0;

  bool is_switch_ = false;
  bool is_converged_ = false;
  bool is_h_converged_ = false;  // h1/h2收敛标志

  motion_model::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  std::set<int> observed_ids_;  // 已观测到的ID

  // 当前跟踪的装甲板ID
  int current_id_ = 0;

  // 马氏距离匹配
  int match_by_mahalanobis(const solver::Armor_pose & armor_pose);
  Eigen::VectorXd predict_observation(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd compute_R(const solver::Armor_pose & armor_pose) const;
  Eigen::MatrixXd compute_R_from_prediction(const Eigen::VectorXd & z_pred) const;  // 参照wust_vision
  double get_predicted_z(int id) const;  // 保留用于调试

  // 4维观测更新 [yaw, pitch, distance, angle]（间接耦合方式）
  void update_ypda(const solver::Armor_pose & armor_pose, int id);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace predict

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP