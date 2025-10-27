#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "module/solver.hpp"
#include "solver_node.hpp"
#include "full_model.hpp"
#include "common/armor.hpp"
#include "logger.hpp"
namespace predict
{

class Target
{
public:
  // 使用完整的命名空间限定符
  armor_auto_aim::ArmorName name;
  armor_auto_aim::ArmorType armor_type;
  armor_auto_aim::ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  
  // 使用 solver 命名空间下的 Armor_pose
  Target(
    const solver::Armor_pose & armor_pose, 
    std::chrono::steady_clock::time_point t, 
    double radius, 
    int armor_num,
    Eigen::VectorXd P0_dig); 
    
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const solver::Armor_pose & armor_pose);

  Eigen::VectorXd ekf_x() const;
  const motion_model::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;
  bool convergened();
  bool isinit = false;
  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  motion_model::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const solver::Armor_pose & armor_pose, int id);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace predict

#endif  // AUTO_AIM__TARGET_HPP