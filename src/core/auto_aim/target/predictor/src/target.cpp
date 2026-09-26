#include "target.hpp"

#include <algorithm>
#include <cmath>
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
: name(armor_pose.id),
  armor_type(armor_pose.type),
  priority(armor_auto_aim::ArmorPriority::fifth),
  jumped(false),
  last_id(0),
  armor_num_(armor_num),
  switch_count_(0),
  update_count_(0),
  is_switch_(false),
  is_converged_(false),
  t_(t)
{
  auto r = radius;
  const Eigen::Vector3d & xyz = armor_pose.world_position;
  // 使用 world_orientation 的 yaw
  double yaw = armor_pose.world_orientation.yaw;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(yaw);
  auto center_y = xyz[1] + r * std::sin(yaw);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  Eigen::VectorXd x0(11);
  x0 << center_x, 0, center_y, 0, center_z, 0, yaw, 0, r, 0, 0;
  
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = utils::limit_rad(c[6]);
    return c;
  };

  ekf_ = motion_model::ExtendedKalmanFilter(x0, P0, x_add);
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
  double v1 = 100;  // 加速度方差
  double v2 = 400;  // 角加速度方差
  
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
  
  Q(10, 10) = 0.001;  // h高低差的过程噪声，允许缓慢的纠正
  // Q(8, 8) = 0.001;
  // Q(9, 9) = 0.01;
  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = utils::limit_rad(x_prior[6]);
    return x_prior;
  };

  ekf_.predict(F, Q, f);
}

int Target::match_armor_id(const solver::Armor_pose & armor_pose) const
{
  // 纯关联：把观测匹配到某个装甲板槽位（按距离取最近 3 个，选 angle_error 最小者）。无副作用。
  int id = 0;
  auto min_error = 1e10;
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

  for (int i = 0; i < 3 && i < static_cast<int>(xyza_i_list.size()); i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));

    double angle_error = std::abs(utils::limit_rad(armor_pose.world_orientation.yaw - xyza[3])) +
                         std::abs(utils::limit_rad(armor_pose.world_spherical.yaw - ypd[0]));

    double total_error = angle_error;

    if (std::abs(total_error) < std::abs(min_error)) {
      id = xyza_i_list[i].second;
      min_error = total_error;
    }
  }

  return id;
}

void Target::update_frame_bookkeeping(int primary_id)
{
  // 每帧调一次：以 primary_id 判断是否板切换，维护单板模式与各计数。
  if (primary_id != 0) jumped = true;

  is_switch_ = (primary_id != last_id);

  if (is_switch_) switch_count_++;

  last_id = primary_id;
  update_count_++;

  // 维护单板观测模式
  if (is_switch_) {
    no_switch_count_ = 0;
    single_plate_mode = false;  // 发生板切换，退出单板模式
  } else {
    no_switch_count_++;
  }
  // 进入条件：长时间无切换 且 角速度低（排除高速旋转恰好看同一面的情况）
  if (no_switch_count_ > single_plate_threshold &&
      std::abs(ekf_.x[7]) < omega_threshold) {
    single_plate_mode = true;
  }
}

void Target::set_uv_context(
  solver::Solver * solver, bool use_uv, double pixel_std, double radius_min, double radius_max,
  bool jacobian_analytic)
{
  solver_ = solver;
  use_uv_ = use_uv;
  uv_pixel_std_ = pixel_std;
  radius_min_ = radius_min;
  radius_max_ = radius_max;
  uv_jacobian_analytic_ = jacobian_analytic;
}

void Target::update(const solver::Armor_pose & armor_pose)
{
  // 单观测委托到多观测路径，保证单块场景与多块场景逐位一致。
  update(std::vector<solver::Armor_pose>{armor_pose});
}

void Target::update(const std::vector<solver::Armor_pose> & armor_poses)
{
  if (armor_poses.empty()) return;

  // uv 路径：§9-D 索引递推关联 + 逐块 update_uv（YPDA 路径完全不变，仅在 use_uv 且有重投影能力时进入）。
  if (use_uv_ && solver_ != nullptr) {
    int primary_id = 0;
    const std::vector<int> slots = uv_associate(armor_poses, primary_id);
    update_frame_bookkeeping(primary_id);
    for (std::size_t i = 0; i < armor_poses.size(); ++i) {
      if (armor_poses[i].detected_corners.size() >= 4) {
        update_uv(armor_poses[i], slots[i]);
      } else {
        update_ypda(armor_poses[i], slots[i]);
      }
    }
    return;
  }

  // 1. 每个观测算候选槽位；按匹配误差升序做无冲突贪心分配（两观测不占同一槽位）。
  struct Assoc {
    const solver::Armor_pose * pose;
    int id;
    double error;
  };
  std::vector<Assoc> assocs;
  assocs.reserve(armor_poses.size());
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  for (const auto & pose : armor_poses) {
    int id = match_armor_id(pose);
    // 记录该观测对其候选槽位的匹配误差（用于冲突时优先级排序 / primary 选择）。
    const auto & xyza = xyza_list[id];
    Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));
    double error = std::abs(utils::limit_rad(pose.world_orientation.yaw - xyza[3])) +
                   std::abs(utils::limit_rad(pose.world_spherical.yaw - ypd[0]));
    assocs.push_back({&pose, id, error});
  }

  std::sort(assocs.begin(), assocs.end(),
            [](const Assoc & a, const Assoc & b) { return a.error < b.error; });

  std::vector<bool> slot_taken(std::max(1, armor_num_), false);
  std::vector<Assoc> final_assocs;
  for (auto & a : assocs) {
    int id = a.id;
    if (id >= 0 && id < armor_num_ && slot_taken[id]) {
      // 槽位已被更优观测占用：退到相邻未占用槽位（mod armor_num_），仍冲突则丢弃该观测。
      int alt = -1;
      for (int d = 1; d < armor_num_; d++) {
        int cand = ((id + d) % armor_num_ + armor_num_) % armor_num_;
        if (!slot_taken[cand]) { alt = cand; break; }
      }
      if (alt < 0) continue;
      id = alt;
    }
    if (id < 0 || id >= armor_num_) continue;
    slot_taken[id] = true;
    a.id = id;
    final_assocs.push_back(a);
  }

  if (final_assocs.empty()) return;

  // 2. primary id：优先选槽位 == last_id 的观测（视为未切换）；否则选误差最小者。
  int primary_id = final_assocs.front().id;  // 已按 error 升序，front 即最小误差
  for (const auto & a : final_assocs) {
    if (a.id == last_id) { primary_id = a.id; break; }
  }

  // 3. 每帧记账一次。
  update_frame_bookkeeping(primary_id);

  // 4. 序贯 EKF：逐块更新，中间不 predict（predict 已在 tracker 中每帧调过一次）。
  //    （到此处必为非 uv 路径；uv 路径已在函数开头 §9-D 关联后 return。）
  for (const auto & a : final_assocs) {
    update_ypda(*a.pose, a.id);
  }

  // utils::logger()->info(
  //   "[Target收敛] 更新:{} | l={:.4f} P_l={:.4f} | h={:.4f} P_h={:.4f} | ω={:.2f}rad/s | r={:.3f}",
  //   update_count_,
  //   ekf_.x[9], ekf_.P(9, 9),   // l 及其协方差
  //   ekf_.x[10], ekf_.P(10, 10), // h 及其协方差
  //   ekf_.x[7],                  // 角速度
  //   ekf_.x[8]                   // 半径
  // );


  // // 每30帧输出一次h可观测性诊断 (约1秒一次)
  // if (armor_num_ == 4 && update_count_ % 30 == 0) {
  //   double obs = h_observability();
  //   double w = omega();
  //   double rate = id_switch_rate();
  //   double h = ekf_.x[10];
  //   double P_h = ekf_.P(10, 10);

  //   if (obs < 0.5) {
  //     utils::logger()->warn(
  //       "[Target] h可观测性低! ω={:.2f}rad/s, 可观测性={:.1f}%, ID切换率={:.1f}%, h={:.3f}m, P_h={:.3f}",
  //       w, obs * 100, rate * 100, h, P_h);
  //   } else {
  //     utils::logger()->debug(
  //       "[Target] h诊断: ω={:.2f}rad/s, 可观测性={:.1f}%, ID切换率={:.1f}%, h={:.3f}m, P_h={:.3f}",
  //       w, obs * 100, rate * 100, h, P_h);
  //   }
  // }
}

void Target::update_uv(const solver::Armor_pose & armor_pose, int id)
{
  // uv 像素观测：单灯条 4 维、逐根序贯（左杆角点{0,3}、右杆角点{1,2}），复用 YawOptimizer 重投影作 h(x)。
  solver::YawOptimizer * yopt = solver_->getYawOptimizer();
  if (yopt == nullptr) { update_ypda(armor_pose, id); return; }

  const auto & corners = armor_pose.detected_corners;  // PW_SMALL 序：左上/右上/右下/左下
  const auto type = armor_pose.type;
  const auto name = armor_pose.id;

  // §9-B 可见性门控：若该槽位装甲板背向相机（多为快转时关联错到远侧槽位/极端斜视），
  //   其重投影与检测角点严重不符，用像素紧 R 会喷出巨大 innovation → 跳过该观测，防污染 EKF。
  if (!uv_slot_faces_camera(id)) return;

  // h(x)：状态 -> 槽位 id 装甲板世界中心 + 世界 yaw -> 重投影 4 角点像素（内部钉死 ±15° pitch）。
  auto reproject = [&](const Eigen::VectorXd & x) -> std::vector<cv::Point2f> {
    Eigen::Vector3d c = h_armor_xyz(x, id);
    double yaw = utils::limit_rad(x[6] + id * 2.0 * CV_PI / armor_num_);
    return yopt->reproject_armor_out(c, yaw, type, name);
  };

  // 观测只依赖 7 个几何维；速度维(1,3,5,7)对像素导数为 0（H 列留 0）。
  static const int kGeoDims[] = {0, 2, 4, 6, 8, 9, 10};
  constexpr double kEps = 1e-4;

  // 对一根灯条（上/下端点两个角点）做一次 4 维 EKF update。
  auto update_bar = [&](int top_idx, int bot_idx) {
    Eigen::VectorXd z(4);
    z << corners[top_idx].x, corners[top_idx].y, corners[bot_idx].x, corners[bot_idx].y;

    auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
      const auto pts = reproject(x);
      Eigen::VectorXd zp(4);
      zp << pts[top_idx].x, pts[top_idx].y, pts[bot_idx].x, pts[bot_idx].y;
      return zp;
    };

    // 雅可比 H(4×11)：uv_jacobian_analytic_ ? §9-C 混合解析 : 数值中心差分（对拍/回退）。
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 11);
    if (uv_jacobian_analytic_) {
      // 投影解析 × world→camera 旋转 × (θ 数值 + xyz/r/l/h 解析)；逐角点 2×11。
      const int idx[2] = {top_idx, bot_idx};
      for (int e = 0; e < 2; ++e) {
        const Eigen::Vector3d cw = uv_armor_corner_world(ekf_.x, id, idx[e]);
        const Eigen::Matrix<double, 2, 3> Jpix = yopt->pixel_jacobian(cw);
        const Eigen::Matrix<double, 3, 11> Jcw = uv_corner_state_jacobian(id, idx[e]);
        H.block<2, 11>(e * 2, 0) = Jpix * Jcw;
      }
    } else {
      const Eigen::VectorXd x0 = ekf_.x;
      for (int d : kGeoDims) {
        Eigen::VectorXd xp = x0, xm = x0;
        xp[d] += kEps;
        xm[d] -= kEps;
        if (d == 6) { xp[6] = utils::limit_rad(xp[6]); xm[6] = utils::limit_rad(xm[6]); }
        H.col(d) = (h(xp) - h(xm)) / (2.0 * kEps);
      }
    }

    Eigen::MatrixXd R = (uv_pixel_std_ * uv_pixel_std_) * Eigen::MatrixXd::Identity(4, 4);

    // 观测级卡方门（robust）：mis-association / 坏帧会喷出巨大 innovation。若该次 NIS 超阈值，
    //   丢弃此灯条 update，避免污染滤波并触发整目标重置（补足 §9-D 关联未上线前的鲁棒性）。
    {
      const Eigen::VectorXd innov = z - h(ekf_.x);
      const Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;
      const double nis = innov.transpose() * S.inverse() * innov;
      if (!std::isfinite(nis) || nis > uv_nis_reject_) return;
    }

    ekf_.update(z, H, R, h);  // 默认 z_subtract 直减（像素无需角度环绕）
    clamp_state();            // 每根 update 后 clamp，并刷新下一根的线性化点
  };

  update_bar(0, 3);  // 左灯条（y=+67.5）：上=左上(0)、下=左下(3)
  update_bar(1, 2);  // 右灯条（y=-67.5）：上=右上(1)、下=右下(2)
}

void Target::clamp_state()
{
  // §9-F 状态 clamp：防半径穿负 / 速度爆炸。仅在 uv 路径调用，不影响 YPDA 行为。
  ekf_.x[8] = std::clamp(ekf_.x[8], radius_min_, radius_max_);                          // r 前板半径
  ekf_.x[9] = std::clamp(ekf_.x[9], radius_min_ - ekf_.x[8], radius_max_ - ekf_.x[8]);  // l：保证 r+l∈[min,max]
  ekf_.x[10] = std::clamp(ekf_.x[10], -0.2, 0.2);                                       // h 高度差
  ekf_.x[1] = std::clamp(ekf_.x[1], -10.0, 10.0);                                       // vx
  ekf_.x[3] = std::clamp(ekf_.x[3], -10.0, 10.0);                                       // vy
}

bool Target::uv_slot_faces_camera(int id) const
{
  // §9-B：槽位 id 的装甲板外法向与「装甲板→相机」视线夹角 < 门限则朝相机（喂），否则背向（跳）。
  if (solver_ == nullptr) return true;
  solver::CoordConverter * cc = solver_->getCoordConverter();
  const Eigen::Vector3d p = h_armor_xyz(ekf_.x, id);
  const double yaw = utils::limit_rad(ekf_.x[6] + id * 2.0 * CV_PI / armor_num_);
  const double pitch =
    (name == armor_auto_aim::ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  const Eigen::Matrix3d R = utils::rotation_matrix_zyx(yaw, pitch, 0.0);
  const Eigen::Vector3d normal_world = -R.col(0);  // 外法向（面朝外，指向相机侧为正）
  const Eigen::Vector3d cam_world = cc->transform(
    Eigen::Vector3d::Zero(), solver::CoordinateFrame::CAMERA, solver::CoordinateFrame::WORLD);
  const Eigen::Vector3d view_dir = (cam_world - p).normalized();
  return normal_world.dot(view_dir) >= uv_gate_cos_;
}

Eigen::Vector3d Target::uv_armor_corner_world(const Eigen::VectorXd & x, int id, int k) const
{
  // 世界角点 = 装甲板世界中心 + R_armor_world(yaw,±15°) · (PW[k]/1000)，与 reproject_armor_out 同源。
  const Eigen::Vector3d center = h_armor_xyz(x, id);
  const double yaw = utils::limit_rad(x[6] + id * 2.0 * CV_PI / armor_num_);
  const double pitch =
    (name == armor_auto_aim::ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  const Eigen::Matrix3d R = utils::rotation_matrix_zyx(yaw, pitch, 0.0);
  const auto & PW = (armor_type == armor_auto_aim::ArmorType::big) ? solver::PW_BIG : solver::PW_SMALL;
  const Eigen::Vector3d off(PW[k].x / 1000.0, PW[k].y / 1000.0, PW[k].z / 1000.0);
  return center + R * off;
}

Eigen::Matrix<double, 3, 11> Target::uv_corner_state_jacobian(int id, int k) const
{
  // ∂世界角点/∂状态 (3×11)：θ 数值中心差分；xyz/r/l/h 解析（复用 h_jacobian 的几何导数）。
  Eigen::Matrix<double, 3, 11> J = Eigen::Matrix<double, 3, 11>::Zero();
  const double angle = utils::limit_rad(ekf_.x[6] + id * 2.0 * CV_PI / armor_num_);
  const bool use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  J(0, 0) = 1.0;  J(1, 2) = 1.0;  J(2, 4) = 1.0;                     // 中心平移
  J(0, 8) = -std::cos(angle);  J(1, 8) = -std::sin(angle);          // r
  if (use_l_h) { J(0, 9) = -std::cos(angle);  J(1, 9) = -std::sin(angle);  J(2, 10) = 1.0; }  // l,h
  constexpr double eps = 1e-4;                                       // θ 数值差分（含中心+板旋转）
  Eigen::VectorXd xp = ekf_.x, xm = ekf_.x;
  xp[6] = utils::limit_rad(xp[6] + eps);
  xm[6] = utils::limit_rad(xm[6] - eps);
  J.col(6) = (uv_armor_corner_world(xp, id, k) - uv_armor_corner_world(xm, id, k)) / (2.0 * eps);
  return J;
}

double Target::uv_reproj_error(const solver::Armor_pose & pose, int id) const
{
  // 槽位 id 预测四角点 vs 观测检测角点的平均像素误差（§9-D 锚点/兜底度量）。
  solver::YawOptimizer * yopt = solver_ ? solver_->getYawOptimizer() : nullptr;
  if (yopt == nullptr || pose.detected_corners.size() < 4) return 1e18;
  const Eigen::Vector3d c = h_armor_xyz(ekf_.x, id);
  const double yaw = utils::limit_rad(ekf_.x[6] + id * 2.0 * CV_PI / armor_num_);
  const auto pts = yopt->reproject_armor_out(c, yaw, pose.type, pose.id);
  if (pts.size() < 4) return 1e18;
  double e = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double dx = pts[i].x - pose.detected_corners[i].x;
    const double dy = pts[i].y - pose.detected_corners[i].y;
    e += std::sqrt(dx * dx + dy * dy);
  }
  return e / 4.0;
}

void Target::disambiguate_pi()
{
  // §9-E：初始观测板(slot0)应朝相机。若模型判其背向，说明初始 yaw 选反 π —— 翻转 yaw 并镜像中心。
  if (solver_ == nullptr) return;
  if (uv_slot_faces_camera(0)) return;
  const double yaw = ekf_.x[6];
  const double r = ekf_.x[8];
  ekf_.x[0] -= 2.0 * r * std::cos(yaw);  // slot0 armor 位置不变，中心镜像到另一侧
  ekf_.x[2] -= 2.0 * r * std::sin(yaw);
  ekf_.x[6] = utils::limit_rad(yaw + CV_PI);
}

std::vector<int> Target::uv_associate(
  const std::vector<solver::Armor_pose> & poses, int & primary_id) const
{
  // §9-D 索引递推关联：锚点(最小重投影误差) + 按检测中心像素横坐标排序 + ±1 mod armor_num_ 递推。
  const int n = static_cast<int>(poses.size());
  std::vector<int> slot(std::max(1, n), 0);
  if (n == 0) { primary_id = 0; return slot; }

  // 1. 每个观测的最优槽位与误差；全局最小者为锚点。
  std::vector<int> best_slot(n, 0);
  std::vector<double> best_err(n, 1e18);
  int anchor = 0;
  double anchor_err = 1e18;
  for (int i = 0; i < n; ++i) {
    for (int s = 0; s < armor_num_; ++s) {
      const double e = uv_reproj_error(poses[i], s);
      if (e < best_err[i]) { best_err[i] = e; best_slot[i] = s; }
    }
    if (best_err[i] < anchor_err) { anchor_err = best_err[i]; anchor = i; }
  }

  // 2. 按检测角点中心像素横坐标升序排序观测索引。
  auto cx = [&](int i) {
    if (poses[i].detected_corners.empty()) return 0.0;
    double s = 0.0;
    for (const auto & p : poses[i].detected_corners) s += p.x;
    return s / static_cast<double>(poses[i].detected_corners.size());
  };
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return cx(a) < cx(b); });

  // 3. 从锚点在排序序列中的位置向两侧 ±1 mod armor_num_ 递推分配槽位。
  int anchor_pos = 0;
  for (int k = 0; k < n; ++k) if (order[k] == anchor) { anchor_pos = k; break; }
  slot[anchor] = best_slot[anchor];
  for (int k = anchor_pos + 1; k < n; ++k)
    slot[order[k]] = ((slot[order[k - 1]] + 1) % armor_num_ + armor_num_) % armor_num_;
  for (int k = anchor_pos; k > 0; --k)
    slot[order[k - 1]] = ((slot[order[k]] - 1) % armor_num_ + armor_num_) % armor_num_;

  // 4. 兜底：递推槽位若明显劣于该观测自身最优（递推方向/锚点误判），退回最优槽位。
  for (int i = 0; i < n; ++i)
    if (uv_reproj_error(poses[i], slot[i]) > best_err[i] + 5.0) slot[i] = best_slot[i];

  // primary：优先命中 last_id（视为未切换），否则锚点槽位。
  primary_id = slot[anchor];
  for (int i = 0; i < n; ++i) if (slot[i] == last_id) { primary_id = last_id; break; }
  return slot;
}

void Target::update_ypda(const solver::Armor_pose & armor_pose, int id)
{
  // 观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  
  auto center_yaw = std::atan2(armor_pose.world_position[1], armor_pose.world_position[0]);
  auto delta_angle = utils::limit_rad(armor_pose.world_orientation.yaw - center_yaw);
  
  // 观测噪声，对应四个观测向量 ypda = [方位yaw, 俯仰pitch, 距离distance, 装甲板朝向angle]。
  // 修复：原先第3项(距离)误用了 delta_angle 公式、第4项(朝向)误用了 distance 公式，二者对调。
  // 正确语义：距离噪声随距离增大；朝向噪声随装甲板偏角(delta_angle)增大。
  Eigen::VectorXd R_dig(4);
  R_dig << 4e-3, 4e-3,
           log(std::abs(armor_pose.world_spherical.distance) + 1) / 200 + 9e-2,  // distance
           log(std::abs(delta_angle) + 1) + 1;                                    // angle(装甲板朝向)

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

  // 定义观测值，即实际解算得到的ypda
  Eigen::VectorXd z(4);
  z << armor_pose.world_spherical.yaw,
       armor_pose.world_spherical.pitch,
       armor_pose.world_spherical.distance,
       armor_pose.world_orientation.yaw;

  ekf_.update(z, H, R, h, z_subtract);

  // 非线性转换函数得到预测的ypda
  Eigen::Vector4d z_pred = h(ekf_.x);
  Eigen::Vector4d innovation = z_subtract(z, z_pred);
  
  // // 计算新息协方差 S = H * P * H^T + R
  // Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;
  
  // // 计算NIS值（归一化新息平方）
  // double nis = innovation.transpose() * S.inverse() * innovation;
  
  // // 记录每个维度的新息和测量噪声
  // utils::logger()->debug(
  //   "【EKF更新详情】更新次数:{} | 装甲板ID:{}\n"
  //   "  观测值 z: [yaw={:.4f}, pitch={:.4f}, dist={:.3f}m, angle={:.4f}]\n"
  //   "  预测值 h(x): [yaw={:.4f}, pitch={:.4f}, dist={:.3f}m, angle={:.4f}]\n"
  //   "  新息 (z-h): [Δyaw={:.4f}, Δpitch={:.4f}, Δdist={:.3f}m, Δangle={:.4f}]\n"
  //   "  测量噪声R对角线: [{:.4e}, {:.4e}, {:.4e}, {:.4e}]\n"
  //   "  NIS值: {:.2f} (理论chi2(4)分布，95%置信区间约9.49)",
  //   update_count_, id,
  //   z[0], z[1], z[2], z[3],
  //   z_pred[0], z_pred[1], z_pred[2], z_pred[3],
  //   innovation[0], innovation[1], innovation[2], innovation[3],
  //   R(0,0), R(1,1), R(2,2), R(3,3),
  //   nis
  // );
  
  // // 如果NIS值异常，发出警告
  // if (nis > 15.0) {  // 远超过95%置信区间的阈值
  //   utils::logger()->warn(
  //     "⚠️ NIS值异常偏高！NIS={:.2f} >> 9.49\n"
  //     "  最大新息维度: {}",
  //     nis,
  //     std::abs(innovation[0]) > std::abs(innovation[1]) && 
  //     std::abs(innovation[0]) > std::abs(innovation[2]) && 
  //     std::abs(innovation[0]) > std::abs(innovation[3]) ? "yaw角度" :
  //     std::abs(innovation[1]) > std::abs(innovation[2]) && 
  //     std::abs(innovation[1]) > std::abs(innovation[3]) ? "pitch角度" :
  //     std::abs(innovation[2]) > std::abs(innovation[3]) ? "距离distance" : "装甲板angle"
  //   );
  // }
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

  // 发散判定逻辑（与参考程序一致）
  auto r_ok = r > 0.05 && r < 0.5;
  auto l_ok = r_plus_l > 0.05 && r_plus_l < 0.5;

  if (r_ok && l_ok) return false;

  utils::logger()->debug("[Target] 发散检测: r={:.3f}, l={:.3f}", r, l);
  return true;
}
bool Target::convergened()
{
  if (update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}


// 置信度检查
bool Target::is_parameter_reliable(int param_idx, double threshold) const 
{
  // 检查协方差矩阵对应位置的不确定度
  return ekf_.P(param_idx, param_idx) < threshold;
}



// 计算出装甲板中心的坐标（考虑长短轴和高度差）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);

  double r;
  double armor_z;

  if (armor_num_ == 4) {
    auto use_l_h = (id == 1 || id == 3);
    r = (use_l_h) ? x[8] + x[9] : x[8];
    armor_z = (use_l_h) ? x[4] + x[10] : x[4];
  }
  else {
    // 基地与平衡步兵
    r = x[8];
    armor_z = x[4];
  }

  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);

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

  // dz/dh 计算
  double dz_dh = (armor_num_ == 4 && (id == 1 || id == 3)) ? 1.0 : 0.0;

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

// bool Target::checkinit() { return isinit; }

// // ID切换率: 切换次数 / 总更新次数
// double Target::id_switch_rate() const {
//   if (update_count_ == 0) return 0.0;
//   return static_cast<double>(switch_count_) / update_count_;
// }

// // 角速度绝对值
// double Target::omega() const {
//   return std::abs(ekf_.x[7]);
// }

// // h可观测性评分 (基于角速度，转速越高可观测性越好)
// // 经验阈值: |ω| < 0.5 rad/s 时可观测性差
// double Target::h_observability() const {
//   double w = omega();
//   // 使用sigmoid函数平滑映射: 0.5 rad/s为中点
//   return 1.0 / (1.0 + std::exp(-10.0 * (w - 0.5)));
// }

}  // namespace predict
