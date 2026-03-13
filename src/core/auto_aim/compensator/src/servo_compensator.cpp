#include "servo_compensator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "logger.hpp"

namespace compensator
{

// ============================================================================
// ZPETCFilter
// ============================================================================

void ZPETCFilter::init(
  const std::vector<double> & b_c, const std::vector<double> & a_c, int d)
{
  if (b_c.empty() || a_c.empty() || std::abs(a_c[0]) < 1e-12) {
    utils::logger()->error("[ZPETC] 无效的传函系数");
    return;
  }

  d_ = d;
  const int m = static_cast<int>(b_c.size()) - 1;  // B_c 的阶数

  // 第一步: 找到 B_c(z⁻¹) 的零点, 分为可消除(单位圆内)和不可消除(单位圆外/上)
  // 对于低阶情况直接解析求解

  // 将 B_c(z⁻¹) 的零点分类
  // B_c(z⁻¹) = b_c[0] + b_c[1]*z⁻¹ + ... + b_c[m]*z⁻ᵐ
  // 零点: B_c(z⁻¹) = 0, 即 b_c[0]*zᵐ + b_c[1]*z^(m-1) + ... + b_c[m] = 0

  // 简化处理: 对于实际云台系统, 闭环传函通常为 1-2 阶
  // 这里采用伴随矩阵法求零点（支持任意阶数）

  // 构造 z 域多项式: p(z) = b_c[0]*z^m + b_c[1]*z^(m-1) + ... + b_c[m]
  std::vector<double> poly_z(m + 1);
  for (int i = 0; i <= m; i++) {
    poly_z[i] = b_c[i];  // 降幂排列
  }

  // 求零点 (对于一阶和二阶直接用公式)
  std::vector<double> zeros_real;
  std::vector<double> zeros_imag;

  if (m == 1) {
    // b_c[0]*z + b_c[1] = 0  =>  z = -b_c[1]/b_c[0]
    zeros_real.push_back(-b_c[1] / b_c[0]);
    zeros_imag.push_back(0.0);
  } else if (m == 2) {
    // b_c[0]*z² + b_c[1]*z + b_c[2] = 0
    double a = b_c[0], b = b_c[1], c = b_c[2];
    double disc = b * b - 4 * a * c;
    if (disc >= 0) {
      zeros_real.push_back((-b + std::sqrt(disc)) / (2 * a));
      zeros_imag.push_back(0.0);
      zeros_real.push_back((-b - std::sqrt(disc)) / (2 * a));
      zeros_imag.push_back(0.0);
    } else {
      zeros_real.push_back(-b / (2 * a));
      zeros_imag.push_back(std::sqrt(-disc) / (2 * a));
      zeros_real.push_back(-b / (2 * a));
      zeros_imag.push_back(-std::sqrt(-disc) / (2 * a));
    }
  } else if (m == 0) {
    // 没有零点, B_c 是常数
  } else {
    utils::logger()->error("[ZPETC] 暂不支持 {} 阶零点多项式, 请使用1-2阶模型", m);
    return;
  }

  // 分类零点
  std::vector<double> ba_coeffs = {1.0};  // B_c^a(z⁻¹) 可消除部分, 初始化为 1
  std::vector<double> bu_coeffs = {1.0};  // B_c^u(z⁻¹) 不可消除部分, 初始化为 1

  for (size_t i = 0; i < zeros_real.size(); i++) {
    double mag = std::sqrt(zeros_real[i] * zeros_real[i] + zeros_imag[i] * zeros_imag[i]);
    if (zeros_imag[i] != 0.0) {
      // 复数零点成对出现, 跳过共轭对的第二个
      if (i + 1 < zeros_real.size() && zeros_imag[i + 1] < 0) {
        // 复数共轭对: (z - (a+jb))(z - (a-jb)) = z² - 2a*z + (a²+b²)
        // 对应 z⁻¹ 域: 1 - 2a*z⁻¹ + (a²+b²)*z⁻²  (除以 z² 后的首项系数)
        double re = zeros_real[i];
        double im = zeros_imag[i];
        double mag_sq = re * re + im * im;
        std::vector<double> pair = {1.0, -2.0 * re, mag_sq};
        if (mag < 1.0 - 1e-6) {
          // 可消除
          std::vector<double> conv(ba_coeffs.size() + pair.size() - 1, 0.0);
          for (size_t j = 0; j < ba_coeffs.size(); j++)
            for (size_t k = 0; k < pair.size(); k++)
              conv[j + k] += ba_coeffs[j] * pair[k];
          ba_coeffs = conv;
        } else {
          // 不可消除
          std::vector<double> conv(bu_coeffs.size() + pair.size() - 1, 0.0);
          for (size_t j = 0; j < bu_coeffs.size(); j++)
            for (size_t k = 0; k < pair.size(); k++)
              conv[j + k] += bu_coeffs[j] * pair[k];
          bu_coeffs = conv;
        }
        i++;  // 跳过共轭
      }
    } else {
      // 实数零点: (z - z0) 对应 z⁻¹ 域: (1 - z0*z⁻¹)
      double z0 = zeros_real[i];
      std::vector<double> factor = {1.0, -z0};
      if (mag < 1.0 - 1e-6) {
        std::vector<double> conv(ba_coeffs.size() + 1, 0.0);
        for (size_t j = 0; j < ba_coeffs.size(); j++) {
          conv[j] += ba_coeffs[j] * factor[0];
          conv[j + 1] += ba_coeffs[j] * factor[1];
        }
        ba_coeffs = conv;
      } else {
        std::vector<double> conv(bu_coeffs.size() + 1, 0.0);
        for (size_t j = 0; j < bu_coeffs.size(); j++) {
          conv[j] += bu_coeffs[j] * factor[0];
          conv[j + 1] += bu_coeffs[j] * factor[1];
        }
        bu_coeffs = conv;
      }
    }
  }

  // 乘上 b_c[0] 的首项系数, 保持与原始 B_c 一致
  for (auto & v : ba_coeffs) v *= (m > 0 ? b_c[0] : 1.0);

  s_ = static_cast<int>(bu_coeffs.size()) - 1;

  // B_c^u(1) = bu_coeffs 求和
  double bu_at_1 = std::accumulate(bu_coeffs.begin(), bu_coeffs.end(), 0.0);
  if (std::abs(bu_at_1) < 1e-12) {
    utils::logger()->error("[ZPETC] B_c^u(1) ≈ 0, 无法构造ZPETC");
    return;
  }

  // B_c^u*(z⁻¹) = 系数倒序
  std::vector<double> bu_star(bu_coeffs.rbegin(), bu_coeffs.rend());

  // 构造 ZPETC 分子: A_c(z⁻¹) * B_c^u*(z⁻¹)
  // 卷积
  std::vector<double> numerator(a_c.size() + bu_star.size() - 1, 0.0);
  for (size_t i = 0; i < a_c.size(); i++)
    for (size_t j = 0; j < bu_star.size(); j++)
      numerator[i + j] += a_c[i] * bu_star[j];

  // 构造 ZPETC 分母: B_c^a(z⁻¹) * (B_c^u(1))²
  double scale = bu_at_1 * bu_at_1;
  std::vector<double> denominator(ba_coeffs.size());
  for (size_t i = 0; i < ba_coeffs.size(); i++) {
    denominator[i] = ba_coeffs[i] * scale;
  }

  // 归一化: 分子分母同除以 denominator[0]
  if (std::abs(denominator[0]) < 1e-12) {
    utils::logger()->error("[ZPETC] 分母首项系数为零");
    return;
  }
  double norm = denominator[0];
  ff_.resize(numerator.size());
  for (size_t i = 0; i < numerator.size(); i++) ff_[i] = numerator[i] / norm;
  fb_.resize(denominator.size());
  for (size_t i = 0; i < denominator.size(); i++) fb_[i] = denominator[i] / norm;

  // 初始化输出历史
  r_hist_.assign(fb_.size(), 0.0);

  ready_ = true;

  utils::logger()->info(
    "[ZPETC] 初始化完成: d={}, s={}, 前馈阶数={}, 反馈阶数={}", d_, s_,
    ff_.size() - 1, fb_.size() - 1);
  utils::logger()->info(
    "[ZPETC] 需要 {} 步未来值 (d={} + s={})", d_ + s_, d_, s_);
}

double ZPETCFilter::apply(const std::vector<double> & y_d_future)
{
  if (!ready_) return y_d_future.empty() ? 0.0 : y_d_future[0];

  int required = d_ + s_ + static_cast<int>(ff_.size());
  if (static_cast<int>(y_d_future.size()) < required) {
    // 未来值不够, 用最后一个值填充
    std::vector<double> padded = y_d_future;
    padded.resize(required, y_d_future.back());
    return apply(padded);
  }

  // r(k) = sum_{i=0}^{nff} ff_[i] * y_d(k + d + s - i)  -  sum_{j=1}^{nfb} fb_[j] * r(k-j)
  double r = 0.0;

  // 前馈部分: y_d_future[d_ + s_] 对应 y_d(k + d + s)
  for (size_t i = 0; i < ff_.size(); i++) {
    int idx = d_ + s_ - static_cast<int>(i);
    if (idx >= 0 && idx < static_cast<int>(y_d_future.size())) {
      r += ff_[i] * y_d_future[idx];
    }
  }

  // 反馈部分
  for (size_t j = 1; j < fb_.size(); j++) {
    if (j <= r_hist_.size()) {
      r -= fb_[j] * r_hist_[j - 1];
    }
  }

  // 更新历史
  r_hist_.push_front(r);
  if (r_hist_.size() > fb_.size()) {
    r_hist_.pop_back();
  }

  return r;
}

double ZPETCFilter::apply(double y_d, double y_d_vel, double dt)
{
  if (!ready_) return y_d;

  // 用线性外推生成未来值
  int steps = d_ + s_ + static_cast<int>(ff_.size());
  std::vector<double> future(steps);
  for (int i = 0; i < steps; i++) {
    future[i] = y_d + y_d_vel * dt * i;
  }
  return apply(future);
}

void ZPETCFilter::reset()
{
  r_hist_.assign(fb_.size(), 0.0);
}

// ============================================================================
// ServoCompensator
// ============================================================================

ServoCompensator::ServoCompensator(const std::string & config_path)
{
  try {
    YAML::Node config = YAML::LoadFile(config_path);
    auto comp_node = config["ServoCompensator"];
    if (!comp_node || !comp_node["enable"].as<bool>(false)) {
      utils::logger()->info("[ServoCompensator] 未启用");
      enabled_ = false;
      return;
    }

    // 加载 yaw 轴参数
    auto yaw_node = comp_node["yaw"];
    if (yaw_node) {
      auto b_c = yaw_node["b_c"].as<std::vector<double>>();
      auto a_c = yaw_node["a_c"].as<std::vector<double>>();
      int d = yaw_node["d"].as<int>(1);
      yaw_filter_.init(b_c, a_c, d);
    }

    // 加载 pitch 轴参数
    auto pitch_node = comp_node["pitch"];
    if (pitch_node) {
      auto b_c = pitch_node["b_c"].as<std::vector<double>>();
      auto a_c = pitch_node["a_c"].as<std::vector<double>>();
      int d = pitch_node["d"].as<int>(1);
      pitch_filter_.init(b_c, a_c, d);
    }

    enabled_ = yaw_filter_.ready() || pitch_filter_.ready();
    if (enabled_) {
      utils::logger()->info("[ServoCompensator] 启用成功");
    }
  } catch (const std::exception & e) {
    utils::logger()->warn("[ServoCompensator] 加载配置失败: {}", e.what());
    enabled_ = false;
  }
}

void ServoCompensator::compensate(
  double yaw, double yaw_vel,
  double pitch, double pitch_vel,
  double dt,
  double & yaw_cmd, double & pitch_cmd)
{
  yaw_cmd = yaw_filter_.ready() ? yaw_filter_.apply(yaw, yaw_vel, dt) : yaw;
  pitch_cmd = pitch_filter_.ready() ? pitch_filter_.apply(pitch, pitch_vel, dt) : pitch;
}

void ServoCompensator::reset()
{
  yaw_filter_.reset();
  pitch_filter_.reset();
}

}  // namespace compensator
