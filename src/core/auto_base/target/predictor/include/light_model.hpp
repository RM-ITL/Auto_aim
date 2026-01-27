#ifndef AUTO_BASE__LIGHT_MODEL_HPP_
#define AUTO_BASE__LIGHT_MODEL_HPP_

#include <Eigen/Dense>
#include <chrono>
#include <optional>

#include "extended_kalman_filter.hpp"
#include "openvino_infer.hpp"

namespace auto_base
{

class LightTarget
{
public:
  // 构造函数：从 GreenLight 检测结果初始化，可指定初始协方差
  LightTarget(
    const OpenvinoInfer::GreenLight & detection,
    std::chrono::steady_clock::time_point t,
    Eigen::VectorXd P0_dig);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);

  void update(const OpenvinoInfer::GreenLight & detection);

  Eigen::VectorXd ekf_x() const;

  const utils::ExtendedKalmanFilter & ekf() const;

  OpenvinoInfer::GreenLight predicted_detection() const;

  bool is_converged() const;

  bool is_diverged() const;

  int update_count() const { return update_count_; }

private:
  utils::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;
  int update_count_ = 0;

  Eigen::MatrixXd build_F(double dt);

  Eigen::MatrixXd build_Q(double dt);

  Eigen::MatrixXd build_H();

  Eigen::MatrixXd build_R();

  Eigen::Vector4d extract_measurement(const OpenvinoInfer::GreenLight & detection);
};

}  // namespace auto_base

#endif  // AUTO_BASE__LIGHT_MODEL_HPP_
