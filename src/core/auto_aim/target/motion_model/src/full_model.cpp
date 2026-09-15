#include "full_model.hpp"

#include <cmath>
#include <numeric>

namespace motion_model
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  // 逐维归一化残差 residual[i]/sqrt(S(i,i))：一致时每维应 ~N(0,1)。
  // 若 R 的距离项与 yaw 项对调，这里会看到 nres_distance 与 nres_angle 的量级明显偏离 1。
  data["nres_yaw"] = 0.0;
  data["nres_pitch"] = 0.0;
  data["nres_distance"] = 0.0;
  data["nres_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  P = F * P * F.transpose() + Q;
  x = f(x);
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  Eigen::VectorXd x_prior = x;

  // pre-fit 新息：更新前用先验状态 x̂⁻ 与先验协方差 P⁻ 计算。
  // 卡方检验(NIS)与卡尔曼增益复用同一套 (innovation, S)，二者口径一致，NIS 才服从 χ²(dim z)。
  Eigen::VectorXd innovation = z_subtract(z, h(x));  // 此刻 x 仍是先验 x̂⁻
  Eigen::MatrixXd S = H * P * H.transpose() + R;      // 此刻 P 仍是先验 P⁻

  Eigen::MatrixXd K = P * H.transpose() * S.inverse();

  // Stable Compution of the Posterior Covariance
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  x = x_add(x, K * innovation);  // 复用 pre-fit 新息，滤波输出与原实现一致

  /// 卡方检验（pre-fit 口径）
  double nis = innovation.transpose() * S.inverse() * innovation;
  // 注：这不是真 NEES(缺真值 x_true)，仅为更新步长的归一化量，参考用。
  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

  // 卡方检验阈值：dim z = 4 → χ²(4) 95% 上分位 = 9.488（NIS > 此值判失败）。
  constexpr double nis_threshold = 9.488;
  constexpr double nees_threshold = 9.488;

  if (nis > nis_threshold) nis_count_++, data["nis_fail"] = 1;
  if (nees > nees_threshold) nees_count_++, data["nees_fail"] = 1;
  total_count_++;
  last_nis = nis;

  recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);

  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

  data["residual_yaw"] = innovation[0];
  data["residual_pitch"] = innovation[1];
  data["residual_distance"] = innovation[2];
  data["residual_angle"] = innovation[3];
  // 逐维归一化新息：innovation[i]/sqrt(S(i,i))，pre-fit 口径。一致时每维应 ~N(0,1)。
  const char * nres_keys[4] = {"nres_yaw", "nres_pitch", "nres_distance", "nres_angle"};
  for (int i = 0; i < innovation.size() && i < 4; ++i) {
    double var = S(i, i);
    data[nres_keys[i]] = innovation[i] / std::sqrt(var > 1e-12 ? var : 1e-12);
  }
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = recent_rate;

  return x;
}

}  // namespace 