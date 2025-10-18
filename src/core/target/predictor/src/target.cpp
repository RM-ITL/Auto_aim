#include "target.hpp"

#include <numeric>

#include "math_tools.hpp"

namespace predict
{
Target::Target(
  const solver::Armor_pose & armor_pose, 
  std::chrono::steady_clock::time_point t, 
  double radius, 
  int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor_pose.id),        // 注意: Armor_pose 中的字段是 id, 不是 name
  armor_type(armor_pose.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor_auto_aim::ArmorPriority::fifth;
  
  const Eigen::Vector3d & xyz = armor_pose.world_position;
  // 使用 world_orientation 的 yaw
  double yaw = armor_pose.world_orientation.yaw;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(yaw);
  auto center_y = xyz[1] + r * std::sin(yaw);
  auto center_z = xyz[2];


  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0(11);
  x0 << center_x, 0, center_y, 0, center_z, 0, yaw, 0, r, 0, 0;
  
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = utils::limit_rad(c[6]);
    return c;
  };

  ekf_ = motion_model::ExtendedKalmanFilter(x0, P0, x_add);  // 初始化滤波器（预测量、预测量协方差）
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0(11);
  x0 << x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h;
  
  Eigen::VectorXd P0_dig(11);
  P0_dig.setZero();
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = utils::limit_rad(c[6]);
    return c;
  };

  ekf_ = motion_model::ExtendedKalmanFilter(x0, P0, x_add);  // 初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = utils::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
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

  // Piecewise White Noise Model
  double v1, v2;
  if (name == armor_auto_aim::ArmorName::outpost) {
    v1 = 10;   // 前哨站加速度方差
    v2 = 0.1;  // 前哨站角加速度方差
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
  }
  
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  
  // 预测过程噪声协方差的方差
  Eigen::MatrixXd Q(11, 11);
  Q << a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0,
       b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0,
            0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0,
            0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0,
            0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0,
            0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0,
            0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0,
            0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0,
            0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0,
            0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0,
            0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0;

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = utils::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速特判
  if (this->convergened() && this->name == armor_auto_aim::ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
}

void Target::update(const solver::Armor_pose & armor_pose)
{
  // 装甲板匹配
  int id;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = utils::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = utils::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取前3个distance最小的装甲板
  for (int i = 0; i < 3 && i < xyza_i_list.size(); i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));
    
    // 使用 world_spherical 和 world_orientation
    auto angle_error = std::abs(utils::limit_rad(armor_pose.world_orientation.yaw - xyza[3])) +
                       std::abs(utils::limit_rad(armor_pose.world_spherical.yaw - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor_pose, id);

  
}

void Target::update_ypda(const solver::Armor_pose & armor_pose, int id)
{
  // 观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  
  auto center_yaw = std::atan2(armor_pose.world_position[1], armor_pose.world_position[0]);
  auto delta_angle = utils::limit_rad(armor_pose.world_orientation.yaw - center_yaw);
  
  Eigen::VectorXd R_dig(4);
  R_dig << 4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
           log(std::abs(armor_pose.world_spherical.distance) + 1) / 200 + 9e-2;

  // 测量过程噪声协方差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = utils::xyz2ypd(xyz);
    auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return Eigen::Vector4d(ypd[0], ypd[1], ypd[2], angle);
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = utils::limit_rad(c[0]);
    c[1] = utils::limit_rad(c[1]);
    c[3] = utils::limit_rad(c[3]);
    return c;
  };

  // 使用 world_spherical 和 world_orientation
  Eigen::VectorXd z(4);
  z << armor_pose.world_spherical.yaw,
       armor_pose.world_spherical.pitch,
       armor_pose.world_spherical.distance,
       armor_pose.world_orientation.yaw;

  ekf_.update(z, H, R, h, z_subtract);

  Eigen::Vector4d z_pred = h(ekf_.x);
  Eigen::Vector4d innovation = z_subtract(z, z_pred);
  
  // 计算新息协方差 S = H * P * H^T + R
  Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;
  
  // 计算NIS值（归一化新息平方）
  double nis = innovation.transpose() * S.inverse() * innovation;
  
  // 记录每个维度的新息和测量噪声
  utils::logger()->debug(
    "【EKF更新详情】更新次数:{} | 装甲板ID:{}\n"
    "  观测值 z: [yaw={:.4f}, pitch={:.4f}, dist={:.3f}m, angle={:.4f}]\n"
    "  预测值 h(x): [yaw={:.4f}, pitch={:.4f}, dist={:.3f}m, angle={:.4f}]\n"
    "  新息 (z-h): [Δyaw={:.4f}, Δpitch={:.4f}, Δdist={:.3f}m, Δangle={:.4f}]\n"
    "  测量噪声R对角线: [{:.4e}, {:.4e}, {:.4e}, {:.4e}]\n"
    "  NIS值: {:.2f} (理论chi2(4)分布，95%置信区间约9.49)",
    update_count_, id,
    z[0], z[1], z[2], z[3],
    z_pred[0], z_pred[1], z_pred[2], z_pred[3],
    innovation[0], innovation[1], innovation[2], innovation[3],
    R(0,0), R(1,1), R(2,2), R(3,3),
    nis
  );
  
  // 如果NIS值异常，发出警告
  if (nis > 15.0) {  // 远超过95%置信区间的阈值
    utils::logger()->warn(
      "⚠️ NIS值异常偏高！NIS={:.2f} >> 9.49\n"
      "  最大新息维度: {}",
      nis,
      std::abs(innovation[0]) > std::abs(innovation[1]) && 
      std::abs(innovation[0]) > std::abs(innovation[2]) && 
      std::abs(innovation[0]) > std::abs(innovation[3]) ? "yaw角度" :
      std::abs(innovation[1]) > std::abs(innovation[2]) && 
      std::abs(innovation[1]) > std::abs(innovation[3]) ? "pitch角度" :
      std::abs(innovation[2]) > std::abs(innovation[3]) ? "距离distance" : "装甲板angle"
    );
  }
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const motion_model::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = utils::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back(Eigen::Vector4d(xyz[0], xyz[1], xyz[2], angle));
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  // 获取当前状态的关键参数
  double r = ekf_.x[8];           // 半径
  double l = ekf_.x[9];           // 长短轴差
  double r_plus_l = r + l;        // 长轴半径
  double cov_trace = ekf_.P.trace();  // 协方差矩阵的迹
  
  // 计算速度
  double vx = ekf_.x[1];
  double vy = ekf_.x[3]; 
  double vz = ekf_.x[5];
  double speed = std::sqrt(vx*vx + vy*vy + vz*vz);
  
  // 原始的判定逻辑
  auto r_ok = r > 0.05 && r < 0.5;
  auto l_ok = r_plus_l > 0.05 && r_plus_l < 0.5;
  
  // 详细记录各项检查结果，这是理解问题的关键
  // utils::logger()->debug(
  //   "发散检测详情 - [更新次数:{}] "
  //   "r:{:.3f}(ok:{}), r+l:{:.3f}(ok:{}), "
  //   "速度:{:.3f}m/s, 协方差迹:{:.3f}, "
  //   "目标类型:{}",
  //   update_count_,
  //   r, r_ok ? "是" : "否",
  //   r_plus_l, l_ok ? "是" : "否",
  //   speed, cov_trace,
  //   name == armor_auto_aim::ArmorName::outpost ? "前哨站" : 
  //   (name == armor_auto_aim::ArmorName::base ? "基地" : "步兵")
  // );
  
  // // 如果判定为发散，详细说明原因
  // if (!r_ok || !l_ok) {
  //   utils::logger()->warn(
  //     "目标发散! 原因: {} "
  //     "[r={:.3f}(范围:0.05-0.5), r+l={:.3f}(范围:0.05-0.5)]",
  //     !r_ok ? "半径超范围" : "长轴超范围",
  //     r, r_plus_l
  //   );
    
  //   // 输出更多诊断信息帮助分析
  //   utils::logger()->warn(
  //     "发散时的状态向量: x={:.2f}, vx={:.2f}, y={:.2f}, vy={:.2f}, "
  //     "z={:.2f}, vz={:.2f}, angle={:.2f}, w={:.2f}",
  //     ekf_.x[0], ekf_.x[1], ekf_.x[2], ekf_.x[3],
  //     ekf_.x[4], ekf_.x[5], ekf_.x[6], ekf_.x[7]
  //   );
    
  //   return true;
  // }
  
  return false;
}
bool Target::convergened()
{
  if (this->name != armor_auto_aim::ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  // 前哨站特殊判断
  if (this->name == armor_auto_aim::ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  return Eigen::Vector3d(armor_x, armor_y, armor_z);
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  Eigen::MatrixXd H_armor_xyza(4, 11);
  H_armor_xyza << 1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0,
                   0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0,
                   0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh,
                   0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0;

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = utils::xyz2ypd_jacobian(armor_xyz);
  
  Eigen::MatrixXd H_armor_ypda(4, 4);
  H_armor_ypda << H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0,
                   H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0,
                   H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0,
                                   0,                 0,                 0, 1;

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace predict