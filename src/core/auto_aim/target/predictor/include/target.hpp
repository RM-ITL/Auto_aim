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
#include "armor.hpp"
#include "logger.hpp"
namespace predict
{

class Target
{
public:
  // 使用完整的命名空间限定符
  armor_auto_aim::ArmorName name;
  armor_auto_aim::ArmorType armor_type;
  armor_auto_aim::ArmorPriority priority = armor_auto_aim::ArmorPriority::fifth;
  bool jumped = false;
  int last_id = 0;  // debug only
  bool single_plate_mode = false;   // 单板观测模式标志
  int single_plate_threshold = 50;  // 进入单板模式的无切换帧数阈值
  double omega_threshold = 0.5;     // 角速度阈值(rad/s)，低于此值才允许进入单板模式

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
  // 每帧序贯更新：一帧内把所有可见装甲板逐块喂进同一个 EKF（predict 已在外部调过一次）。
  void update(const std::vector<solver::Armor_pose> & armor_poses);

  // uv 观测上下文注入：tracker 在每次 update 前调用，交出重投影能力(solver)与开关/噪声/clamp 参数。
  // 关闭(use_uv=false)时不影响任何原 YPDA 行为。
  void set_uv_context(
    solver::Solver * solver, bool use_uv, double pixel_std, double radius_min, double radius_max,
    bool jacobian_analytic = true);

  // §9-E 初始化 π 歧义消解：用可见板数最大化在 yaw 与 yaw+π 间选朝向（仅 uv 模式，构造后调用一次）。
  void disambiguate_pi();

  Eigen::VectorXd ekf_x() const;
  const motion_model::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;
  bool convergened();
  bool isinit = false;
  // bool checkinit();

  // // h可观测性诊断指标
  // double id_switch_rate() const;  // ID切换率 (0~1)
  // double omega() const;           // 角速度绝对值 (rad/s)
  // double h_observability() const; // h可观测性评分 (0~1)

private:
  int armor_num_ = 0;
  int switch_count_ = 0;
  int update_count_ = 0;
  int no_switch_count_ = 0;  // 连续无板切换帧计数

  bool is_switch_ = false, is_converged_ = false;

  motion_model::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  bool is_parameter_reliable(int param_idx, double threshold) const;  // 置信度检查，来进行l和h状态变量的可用性判断

  void update_ypda(const solver::Armor_pose & armor_pose, int id);  // yaw pitch distance angle
  void update_uv(const solver::Armor_pose & armor_pose, int id);    // 灯条端点像素 uv（单灯条 4 维）
  void clamp_state();                                               // 状态 clamp：防 r 穿负 / 速度爆炸(§9-F)

  // uv 观测辅助
  bool uv_slot_faces_camera(int id) const;                          // §9-B 可见性：槽位是否朝相机
  Eigen::Vector3d uv_armor_corner_world(const Eigen::VectorXd & x, int id, int k) const;  // 世界角点
  // §9-C 混合解析雅可比：单角点 ∂世界角点/∂状态 (3×11)，θ 数值中心差分、其余解析。
  Eigen::Matrix<double, 3, 11> uv_corner_state_jacobian(int id, int k) const;
  // §9-D 索引递推关联：观测→槽位（锚点 + 像素横坐标排序 + ±1 mod armor_num_）。返回每个 pose 的 slot。
  std::vector<int> uv_associate(const std::vector<solver::Armor_pose> & poses, int & primary_id) const;
  double uv_reproj_error(const solver::Armor_pose & pose, int id) const;  // 四角重投影误差(px)

  // uv 观测上下文（由 set_uv_context 注入）
  solver::Solver * solver_ = nullptr;  // 重投影 h(x) 与 world→camera 变换来源；Solver 生命周期覆盖 Target
  bool use_uv_ = false;
  double uv_pixel_std_ = 6.0;
  double radius_min_ = 0.10;
  double radius_max_ = 0.40;
  double uv_gate_cos_ = 0.0;    // 可见性门控阈值：normal·view < 此值(默认 cos90°=0，背向)则跳过
  double uv_nis_reject_ = 50.0;  // 观测级卡方门：单灯条 NIS 超此值判 mis-association/坏帧，丢弃该 update
  bool uv_jacobian_analytic_ = true;  // true=§9-C 混合解析雅可比；false=数值中心差分

  // 纯关联：把一个观测匹配到模型的某个装甲板槽位 id（无副作用）。
  int match_armor_id(const solver::Armor_pose & armor_pose) const;
  // 每帧记账：切换/单板模式/计数等，一帧只调一次（primary_id 决定 is_switch_ 语义）。
  void update_frame_bookkeeping(int primary_id);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace predict

#endif  // AUTO_AIM__TARGET_HPP
