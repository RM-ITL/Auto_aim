#include "outpost_target.hpp"

#include <cmath>
#include <algorithm>

#include "math_tools.hpp"

namespace predict
{

OutpostTarget::OutpostTarget(
  const solver::Armor_pose & armor_pose,
  std::chrono::steady_clock::time_point t,
  double radius,
  Eigen::VectorXd P0_dig)
: armor_type(armor_pose.type),
  t_(t),
  current_id_(0)
{
  auto r = radius;
  const Eigen::Vector3d & xyz = armor_pose.world_position;
  double yaw = armor_pose.world_orientation.yaw;

  // 旋转中心坐标，以第一个观测为基准
  auto center_x = xyz[0] + r * std::cos(yaw);
  auto center_y = xyz[1] + r * std::sin(yaw);
  auto center_z = xyz[2];  // 直接使用观测值作为基准

  utils::logger()->debug(
    "[OutpostTarget] 初始化 center_z={:.3f}, 基准角度={:.3f}",
    center_z, yaw
  );

  // 状态向量: cx vx cy vy cz vz θ ω r h1 h2
  Eigen::VectorXd x0(11);
  x0 << center_x, 0, center_y, 0, center_z, 0, yaw, 0, r, 0, 0;

  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = utils::limit_rad(c[6]);
    return c;
  };

  ekf_ = motion_model::ExtendedKalmanFilter(x0, P0, x_add);

  // 标记ID=0已观测
  observed_ids_.insert(0);
}

void OutpostTarget::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = utils::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void OutpostTarget::predict(double dt)
{
  // 状态转移矩阵
  Eigen::MatrixXd F(11, 11);
  F << 1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1;

  // 前哨站过程噪声
  double v1 = 10;   // 位置加速度方差
  double v2 = 0.1;  // 角加速度方差

  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  // h1和h2是常量，过程噪声设小
  double v_h = 0.01;

  Eigen::MatrixXd Q(11, 11);
  Q << a * v1, b * v1,      0,      0,      0,      0,      0,      0,   0,   0,   0,
       b * v1, c * v1,      0,      0,      0,      0,      0,      0,   0,   0,   0,
            0,      0, a * v1, b * v1,      0,      0,      0,      0,   0,   0,   0,
            0,      0, b * v1, c * v1,      0,      0,      0,      0,   0,   0,   0,
            0,      0,      0,      0, a * v1, b * v1,      0,      0,   0,   0,   0,
            0,      0,      0,      0, b * v1, c * v1,      0,      0,   0,   0,   0,
            0,      0,      0,      0,      0,      0, a * v2, b * v2,   0,   0,   0,
            0,      0,      0,      0,      0,      0, b * v2, c * v2,   0,   0,   0,
            0,      0,      0,      0,      0,      0,      0,      0,   0,   0,   0,
            0,      0,      0,      0,      0,      0,      0,      0,   0, v_h,   0,
            0,      0,      0,      0,      0,      0,      0,      0,   0,   0, v_h;

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = utils::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速限制
  if (this->convergened() && std::abs(this->ekf_.x[7]) > 2) {
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

// 根据ID获取预测的z坐标
double OutpostTarget::get_predicted_z(int id) const
{
  double center_z = ekf_.x[4];
  if (id == 0) return center_z;
  if (id == 1) return center_z + ekf_.x[9];   // h1
  if (id == 2) return center_z + ekf_.x[10];  // h2
  return center_z;
}

// 计算给定ID下的5维预测观测 [yaw, pitch, distance, angle, z_armor]
Eigen::VectorXd OutpostTarget::predict_observation(const Eigen::VectorXd & x, int id) const
{
  Eigen::Vector3d xyz = h_armor_xyz(x, id);
  Eigen::VectorXd ypd = utils::xyz2ypd(xyz);
  auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / ARMOR_NUM);
  double z_armor = xyz[2];

  Eigen::VectorXd result(5);
  result << ypd[0], ypd[1], ypd[2], angle, z_armor;
  return result;
}

// 计算观测噪声矩阵R
Eigen::MatrixXd OutpostTarget::compute_R(const solver::Armor_pose & armor_pose) const
{
  auto center_yaw = std::atan2(armor_pose.world_position[1], armor_pose.world_position[0]);
  auto delta_angle = utils::limit_rad(armor_pose.world_orientation.yaw - center_yaw);

  Eigen::VectorXd R_dig(5);
  R_dig << 4e-3,   // yaw噪声
           4e-3,   // pitch噪声
           std::log(std::abs(delta_angle) + 1) + 1,  // distance噪声
           std::log(std::abs(armor_pose.world_spherical.distance) + 1) / 200 + 9e-2,  // angle噪声
           1e-2;   // z_armor噪声

  return R_dig.asDiagonal();
}

// 马氏距离匹配
int OutpostTarget::match_by_mahalanobis(const solver::Armor_pose & armor_pose)
{
  // 构建5维观测向量
  Eigen::VectorXd z_obs(5);
  z_obs << armor_pose.world_spherical.yaw,
           armor_pose.world_spherical.pitch,
           armor_pose.world_spherical.distance,
           armor_pose.world_orientation.yaw,
           armor_pose.world_position[2];

  // 计算观测噪声
  Eigen::MatrixXd R = compute_R(armor_pose);

  double min_d2 = std::numeric_limits<double>::max();
  int best_id = current_id_;

  // 5维观测的χ²门限（99%置信度，自由度5）
  const double CHI2_GATE = 15.09;

  // 记录每个ID的马氏距离用于调试
  double d2_list[3] = {0, 0, 0};

  for (int id = 0; id < ARMOR_NUM; id++) {
    // 计算预测观测
    Eigen::VectorXd z_pred = predict_observation(ekf_.x, id);

    // 计算残差（角度归一化）
    Eigen::VectorXd nu = z_obs - z_pred;
    nu[0] = utils::limit_rad(nu[0]);  // yaw
    nu[1] = utils::limit_rad(nu[1]);  // pitch
    nu[3] = utils::limit_rad(nu[3]);  // angle

    // 计算雅可比和残差协方差
    Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
    Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;

    // 计算马氏距离
    double d2 = nu.transpose() * S.inverse() * nu;
    d2_list[id] = d2;

    if (d2 < min_d2) {
      min_d2 = d2;
      best_id = id;
    }
  }

  // 记录匹配结果（调试用）
  utils::logger()->debug(
    "[OutpostTarget] 马氏距离: d2=[{:.2f},{:.2f},{:.2f}], best_id={}, min_d2={:.2f}",
    d2_list[0], d2_list[1], d2_list[2], best_id, min_d2
  );

  if (min_d2 > CHI2_GATE) {
    utils::logger()->warn(
      "[OutpostTarget] 马氏距离超过门限: {:.2f} > {:.2f}, 保持ID={}",
      min_d2, CHI2_GATE, current_id_
    );
    // 超过门限时，仍使用最小距离的ID（而不是拒绝更新）
  }

  return best_id;
}

void OutpostTarget::update(const solver::Armor_pose & armor_pose)
{
  // 使用马氏距离匹配确定ID
  int id = match_by_mahalanobis(armor_pose);

  // 记录观测到的ID
  observed_ids_.insert(id);

  // 更新切换状态
  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
    switch_count_++;
  } else {
    is_switch_ = false;
  }

  last_id = id;
  current_id_ = id;
  update_count_++;

  // 使用5维观测更新
  update_ypdaz(armor_pose, id);
}

void OutpostTarget::update_ypdaz(const solver::Armor_pose & armor_pose, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

  auto center_yaw = std::atan2(armor_pose.world_position[1], armor_pose.world_position[0]);
  auto delta_angle = utils::limit_rad(armor_pose.world_orientation.yaw - center_yaw);

  // 5维观测噪声矩阵 R
  Eigen::VectorXd R_dig(5);
  R_dig << 4e-3,   // yaw噪声
           4e-3,   // pitch噪声
           log(std::abs(delta_angle) + 1) + 1,  // distance噪声
           log(std::abs(armor_pose.world_spherical.distance) + 1) / 200 + 9e-2,  // angle噪声
           1e-2;   // z_armor噪声（直接测量，噪声较小）

  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 观测函数 h(): 状态 -> [yaw, pitch, distance, angle, z_armor]
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::Vector3d xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = utils::xyz2ypd(xyz);
    auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / ARMOR_NUM);
    double z_armor = xyz[2];

    Eigen::VectorXd result(5);
    result << ypd[0], ypd[1], ypd[2], angle, z_armor;
    return result;
  };

  // 残差处理（角度归一化）
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = utils::limit_rad(c[0]);  // yaw
    c[1] = utils::limit_rad(c[1]);  // pitch
    c[3] = utils::limit_rad(c[3]);  // angle
    return c;
  };

  // 5维观测向量
  Eigen::VectorXd z(5);
  z << armor_pose.world_spherical.yaw,
       armor_pose.world_spherical.pitch,
       armor_pose.world_spherical.distance,
       armor_pose.world_orientation.yaw,
       armor_pose.world_position[2];  // z_armor

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd OutpostTarget::ekf_x() const { return ekf_.x; }

const motion_model::ExtendedKalmanFilter & OutpostTarget::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  for (int i = 0; i < ARMOR_NUM; i++) {
    auto angle = utils::limit_rad(ekf_.x[6] + i * 2 * CV_PI / ARMOR_NUM);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    list.push_back(Eigen::Vector4d(xyz[0], xyz[1], xyz[2], angle));
  }
  return list;
}

bool OutpostTarget::diverged() const
{
  // 检查协方差是否发散
  for (int i = 0; i < 11; i++) {
    if (ekf_.P(i, i) > P_DIVERGENCE_THRESHOLD) {
      utils::logger()->warn(
        "[OutpostTarget] 协方差发散: P({},{})={:.3f} > {:.3f}",
        i, i, ekf_.P(i, i), P_DIVERGENCE_THRESHOLD
      );
      return true;
    }
  }

  // 检查h1/h2是否超出合理范围
  if (std::abs(ekf_.x[9]) > H_MAX_REASONABLE || std::abs(ekf_.x[10]) > H_MAX_REASONABLE) {
    utils::logger()->warn(
      "[OutpostTarget] h超出合理范围: h1={:.3f}, h2={:.3f}, 阈值={:.3f}",
      ekf_.x[9], ekf_.x[10], H_MAX_REASONABLE
    );
    return true;
  }

  // 检查状态是否包含NaN
  if (ekf_.x.hasNaN() || ekf_.P.hasNaN()) {
    utils::logger()->warn("[OutpostTarget] 状态包含NaN，发散");
    return true;
  }

  return false;
}

bool OutpostTarget::h_converged() const
{
  // h1和h2的协方差都小于阈值，且观测到至少2个不同ID
  double P_h1 = ekf_.P(9, 9);
  double P_h2 = ekf_.P(10, 10);

  bool variance_ok = (P_h1 < H_CONVERGENCE_THRESHOLD) && (P_h2 < H_CONVERGENCE_THRESHOLD);
  bool enough_ids = observed_ids_.size() >= 2;

  return variance_ok && enough_ids;
}

bool OutpostTarget::convergened()
{
  // 整体收敛判断
  if (is_converged_) {
    return true;
  }

  // 检查是否发散
  if (diverged()) {
    return false;
  }

  // 条件1：更新次数足够
  if (update_count_ < 15) {
    return false;
  }

  // 条件2：h已收敛
  if (!h_converged()) {
    return false;
  }

  // 条件3：位置和速度的协方差足够小
  double P_pos_max = std::max({ekf_.P(0, 0), ekf_.P(2, 2), ekf_.P(4, 4)});
  double P_vel_max = std::max({ekf_.P(1, 1), ekf_.P(3, 3), ekf_.P(5, 5)});

  if (P_pos_max > 0.5 || P_vel_max > 10.0) {
    return false;
  }

  // 所有条件满足，标记为收敛
  is_converged_ = true;
  is_h_converged_ = true;

  utils::logger()->info(
    "[OutpostTarget] 收敛完成: update_count={}, observed_ids={}, "
    "P_h1={:.4f}, P_h2={:.4f}, h1={:.3f}, h2={:.3f}",
    update_count_, observed_ids_.size(),
    ekf_.P(9, 9), ekf_.P(10, 10),
    ekf_.x[9], ekf_.x[10]
  );

  return true;
}

Eigen::Vector3d OutpostTarget::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / ARMOR_NUM);

  double armor_z;
  if (id == 0) {
    armor_z = x[4];           // center_z (基准)
  } else if (id == 1) {
    armor_z = x[4] + x[9];    // center_z + h1
  } else {
    armor_z = x[4] + x[10];   // center_z + h2
  }

  auto armor_x = x[0] - x[8] * std::cos(angle);
  auto armor_y = x[2] - x[8] * std::sin(angle);

  return Eigen::Vector3d(armor_x, armor_y, armor_z);
}

Eigen::MatrixXd OutpostTarget::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / ARMOR_NUM);

  auto r = x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);

  // dz/dh1 和 dz/dh2
  double dz_dh1 = (id == 1) ? 1.0 : 0.0;
  double dz_dh2 = (id == 2) ? 1.0 : 0.0;

  // H_armor_xyza: 从状态向量到 (armor_x, armor_y, armor_z, angle)
  // 维度: 4 x 11
  Eigen::MatrixXd H_armor_xyza(4, 11);
  H_armor_xyza << 1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, 0,      0,
                  0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, 0,      0,
                  0, 0, 0, 0, 1, 0,     0, 0,     0, dz_dh1, dz_dh2,
                  0, 0, 0, 0, 0, 0,     1, 0,     0, 0,      0;

  Eigen::Vector3d armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = utils::xyz2ypd_jacobian(armor_xyz);

  // H_armor_ypdaz: 从 (x, y, z, angle) 到 (yaw, pitch, distance, angle, z_armor)
  // 维度: 5 x 4
  Eigen::MatrixXd H_armor_ypdaz(5, 4);
  H_armor_ypdaz << H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0,  // yaw
                   H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0,  // pitch
                   H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0,  // distance
                                   0,                 0,                 0, 1,  // angle
                                   0,                 0,                 1, 0;  // z_armor (直接等于z)

  // 最终雅可比: 5 x 11
  return H_armor_ypdaz * H_armor_xyza;
}

bool OutpostTarget::checkinit() { return isinit; }

}  // namespace predict
