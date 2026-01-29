#include "light_model.hpp"

#include <cmath>

#include "logger.hpp"
#include "math_tools.hpp"

namespace auto_base
{

LightTarget::LightTarget(
  const OpenvinoInfer::GreenLight & detection,
  std::chrono::steady_clock::time_point t,
  Eigen::VectorXd P0_dig)
: t_(t)
{
  // 初始化 8 维状态向量 [cx, cy, w, h, dx, dy, dw, dh]
  Eigen::VectorXd x0(8);
  x0 << detection.center.x,
    detection.center.y,
    detection.box.width,
    detection.box.height,
    0.0,  // dx = 0
    0.0,  // dy = 0
    0.0,  // dw = 0
    0.0;  // dh = 0

  // 初始协方差矩阵 P0 (8x8)，对角矩阵
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 初始化 EKF，这里使用默认的加法运算
  ekf_ = utils::ExtendedKalmanFilter(x0, P0);
}

void LightTarget::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = utils::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void LightTarget::predict(double dt)
{
  // 构建状态转移矩阵 F 和 过程噪声 Q
  Eigen::MatrixXd F = build_F(dt);
  Eigen::MatrixXd Q = build_Q(dt);

  // 简单线性预测：x_new = F * x
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd { return F * x; };

  ekf_.predict(F, Q, f);
}

void LightTarget::update(const OpenvinoInfer::GreenLight & detection)
{
  // 提取观测向量 z [cx, cy, w, h]
  Eigen::Vector4d z = extract_measurement(detection);

  // 构建观测矩阵 H 和 观测噪声 R
  Eigen::MatrixXd H = build_H();
  Eigen::MatrixXd R = build_R();

  // 定义观测函数 h：从状态向量中提取观测部分
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::Vector4d z_pred;
    z_pred << x[0], x[1], x[2], x[3];  // [cx, cy, w, h]
    return z_pred;
  };

  // 定义观测差分运算（简单减法）
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b)
    -> Eigen::VectorXd { return a - b; };

  // 执行 EKF 更新
  ekf_.update(z, H, R, h, z_subtract);

  update_count_++;
}

Eigen::VectorXd LightTarget::ekf_x() const { return ekf_.x; }

const utils::ExtendedKalmanFilter & LightTarget::ekf() const { return ekf_; }

OpenvinoInfer::GreenLight LightTarget::predicted_detection() const
{
  const Eigen::VectorXd & x = ekf_.x;

  OpenvinoInfer::GreenLight pred;
  pred.box = cv::Rect2d(x[0] - x[2] / 2.0, x[1] - x[3] / 2.0, x[2], x[3]);
  pred.center = cv::Point2d(x[0], x[1]);
  pred.score = 1.0;
  pred.class_id = 0;

  return pred;
}

bool LightTarget::is_converged() const
{
  // 至少更新 5 次，且状态不发散
  return update_count_ >= 5 && !is_diverged();
}

bool LightTarget::is_diverged() const
{
  const Eigen::VectorXd & x = ekf_.x;

  // 检查 w 和 h 是否合理（都应该大于 0）
  // 也检查协方差是否过大（发散指示）
  bool w_ok = x[2] > 1.0 && x[2] < 1000.0;
  bool h_ok = x[3] > 1.0 && x[3] < 1000.0;
  bool p_ok = ekf_.P(0, 0) < 1000.0 && ekf_.P(1, 1) < 1000.0;

  if (!w_ok || !h_ok || !p_ok) {
    utils::logger()->warn(
      "[LightTarget] 发散检测: w={:.2f}, h={:.2f}, P_cx={:.2f}, P_cy={:.2f}", x[2], x[3],
      ekf_.P(0, 0), ekf_.P(1, 1));
    return true;
  }

  return false;
}

Eigen::MatrixXd LightTarget::build_F(double dt)
{
  // 8x8 状态转移矩阵
  // 状态：[cx, cy, w, h, dx, dy, dw, dh]
  // 假设速度恒定：x_new = x_old + v * dt
  Eigen::MatrixXd F(8, 8);
  F << 1, 0, 0, 0, dt, 0, 0, 0,   // cx' = cx + dx*dt
    0, 1, 0, 0, 0, dt, 0, 0,      // cy' = cy + dy*dt
    0, 0, 1, 0, 0, 0, dt, 0,      // w'  = w  + dw*dt
    0, 0, 0, 1, 0, 0, 0, dt,      // h'  = h  + dh*dt
    0, 0, 0, 0, 1, 0, 0, 0,       // dx' = dx（恒定）
    0, 0, 0, 0, 0, 1, 0, 0,       // dy' = dy
    0, 0, 0, 0, 0, 0, 1, 0,       // dw' = dw
    0, 0, 0, 0, 0, 0, 0, 1;       // dh' = dh

  return F;
}

Eigen::MatrixXd LightTarget::build_Q(double dt)
{
  // 8x8 过程噪声协方差矩阵
  // 模型：匀速运动模型（piecewise constant white noise）
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(8, 8);

  // 位置部分 [cx, cy] 和速度 [dx, dy] 的耦合噪声
  double v_pos = 8.0;    // 位置加速度方差
  double v_size = 2.0;   // 尺寸变化加速度方差

  double a = dt * dt * dt * dt / 4.0;
  double b = dt * dt * dt / 2.0;
  double c = dt * dt;

  // 位置部分 (cx, cy, dx, dy)
  Q(0, 0) = a * v_pos;
  Q(0, 4) = b * v_pos;
  Q(4, 0) = b * v_pos;
  Q(4, 4) = c * v_pos;

  Q(1, 1) = a * v_pos;
  Q(1, 5) = b * v_pos;
  Q(5, 1) = b * v_pos;
  Q(5, 5) = c * v_pos;

  // 尺寸部分 (w, h, dw, dh)
  Q(2, 2) = a * v_size;
  Q(2, 6) = b * v_size;
  Q(6, 2) = b * v_size;
  Q(6, 6) = c * v_size;

  Q(3, 3) = a * v_size;
  Q(3, 7) = b * v_size;
  Q(7, 3) = b * v_size;
  Q(7, 7) = c * v_size;

  return Q;
}

Eigen::MatrixXd LightTarget::build_H()
{
  // 4x8 观测矩阵
  // 观测：[cx, cy, w, h]，只观测前 4 个状态
  Eigen::MatrixXd H(4, 8);
  H << 1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0;

  return H;
}

Eigen::MatrixXd LightTarget::build_R()
{
  // 4x4 观测噪声协方差矩阵
  // 对角线上为各观测的噪声方差
  Eigen::MatrixXd R(4, 4);
  R << 2.0, 0, 0, 0,    // cx 观测噪声（像素^2）
        0, 2.0, 0, 0,       // cy 观测噪声
        0, 0, 2.0, 0,       // w 观测噪声
        0, 0, 0, 2.0;       // h 观测噪声

  return R;
}

Eigen::Vector4d LightTarget::extract_measurement(const OpenvinoInfer::GreenLight & detection)
{
  // 从 GreenLight 检测结果中提取观测向量 [cx, cy, w, h]
  Eigen::Vector4d z;
  z << detection.center.x,
    detection.center.y,
    detection.box.width,
    detection.box.height;

  return z;
}

}  // namespace auto_base
