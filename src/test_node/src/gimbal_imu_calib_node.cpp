#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "app_config/app_config.hpp"
#include "calibration_common.hpp"
#include "logger.hpp"
#include "math_tools.hpp"

namespace Application
{

namespace
{

// 把 SubConfig 的 vector<double>(9) 转 Eigen::Matrix3d（行优先），与各节点字节级一致。
Eigen::Matrix3d unflatten_3x3(const std::vector<double> & data)
{
  if (data.size() != 9) {
    throw std::runtime_error("rotation_matrix_gimbal_to_imu 数据长度不是 9");
  }
  return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(data.data());
}

std::string matrix_to_yaml_data(const Eigen::Matrix3d & m)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(10);
  oss << "      data: [" << m(0, 0) << ", " << m(0, 1) << ", " << m(0, 2) << ",\n";
  oss << "             " << m(1, 0) << ", " << m(1, 1) << ", " << m(1, 2) << ",\n";
  oss << "             " << m(2, 0) << ", " << m(2, 1) << ", " << m(2, 2) << "]";
  return oss.str();
}

double angle_between_deg(const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  const double cos_theta =
    std::max(-1.0, std::min(1.0, a.normalized().dot(b.normalized())));
  return std::acos(cos_theta) * 180.0 / CV_PI;
}

// 单轴扫描的转动轴估计（Method A）。
// 取参考帧 0，relative ΔR_i = R_0ᵀ·R_i 都绕同一物理云台轴转动，
// 该轴在 IMU 系中是常量，等于待求 G 的对应列（up to sign）。
struct AxisEstimate
{
  Eigen::Vector3d axis{Eigen::Vector3d::UnitZ()};
  double max_dev_deg{0.0};   // 各帧轴对加权均值轴的最大夹角（一致性，越小越好）
  double angle_span_deg{0.0};  // 相对帧0的最大转角（扫描幅度，越大信噪比越好）
  int used{0};               // 参与估计的有效帧数
};

AxisEstimate estimate_axis(
  const std::vector<std::pair<int, Eigen::Quaterniond>> & samples, double min_angle_deg)
{
  if (samples.size() < 2) {
    throw std::runtime_error("单轴扫描样本不足（至少 2 帧），无法估计转动轴");
  }

  const Eigen::Matrix3d r0 = samples.front().second.toRotationMatrix();
  std::vector<std::pair<double, Eigen::Vector3d>> axes;  // (angle_rad, axis)
  axes.reserve(samples.size());

  for (size_t i = 1; i < samples.size(); ++i) {
    const Eigen::Matrix3d ri = samples[i].second.toRotationMatrix();
    const Eigen::AngleAxisd aa(r0.transpose() * ri);
    const double angle_deg = aa.angle() * 180.0 / CV_PI;
    if (angle_deg < min_angle_deg) {
      continue;
    }
    axes.emplace_back(aa.angle(), aa.axis());
  }

  if (axes.empty()) {
    throw std::runtime_error(
      "没有相对帧0转角 >= min-angle-deg 的样本，请扫过更大角度或调小 --min-angle-deg");
  }

  // 统一符号：以首个有效轴为参考，dot<0 翻转。
  const Eigen::Vector3d ref = axes.front().second;
  Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
  double weight_sum = 0.0;
  for (auto & item : axes) {
    if (item.second.dot(ref) < 0.0) {
      item.second = -item.second;
    }
    weighted_sum += item.first * item.second;  // 按转角加权
    weight_sum += item.first;
  }

  AxisEstimate est;
  est.axis = (weighted_sum / weight_sum).normalized();
  est.used = static_cast<int>(axes.size());
  for (const auto & item : axes) {
    est.max_dev_deg = std::max(est.max_dev_deg, angle_between_deg(item.second, est.axis));
    est.angle_span_deg = std::max(est.angle_span_deg, item.first * 180.0 / CV_PI);
  }
  return est;
}

// 把 [x|y|z] 三列投影到最近的旋转矩阵 SO(3)。
Eigen::Matrix3d project_to_so3(const Eigen::Matrix3d & raw)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d d = Eigen::Matrix3d::Identity();
  d(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0 ? -1.0 : 1.0;
  return svd.matrixU() * d * svd.matrixV().transpose();
}

int run_check(const std::string & input_folder, const Eigen::Matrix3d & g)
{
  const auto samples = calibration::enumerate_quaternions(input_folder);
  if (samples.empty()) {
    utils::logger()->error("[GimbalImu] 未在 {} 中找到四元数文件 N.txt", input_folder);
    return 1;
  }

  utils::logger()->info(
    "[GimbalImu] check 模式: 用当前 G 计算 gimbal_to_world = Gᵀ·R_imu·G 并逐帧打印欧拉角");

  Eigen::Vector3d ypr_min(1e9, 1e9, 1e9);
  Eigen::Vector3d ypr_max(-1e9, -1e9, -1e9);
  for (const auto & [index, q] : samples) {
    const Eigen::Matrix3d r_gimbal_to_world = g.transpose() * q.toRotationMatrix() * g;
    const Eigen::Vector3d ypr = utils::eulers(r_gimbal_to_world, 2, 1, 0) * 57.2957795;
    ypr_min = ypr_min.cwiseMin(ypr);
    ypr_max = ypr_max.cwiseMax(ypr);
    utils::logger()->info(
      "[GimbalImu] sample {:03d}: yaw/pitch/roll = {:8.2f} {:8.2f} {:8.2f} deg",
      index, ypr[0], ypr[1], ypr[2]);
  }

  const Eigen::Vector3d range = ypr_max - ypr_min;
  utils::logger()->info(
    "[GimbalImu] 范围(max-min): yaw={:.2f}° pitch={:.2f}° roll={:.2f}° （{} 帧）",
    range[0], range[1], range[2], samples.size());
  utils::logger()->info(
    "[GimbalImu] 判读: 纯 yaw 扫描应 yaw 范围大、pitch/roll 范围小；纯 pitch 反之。"
    "若该不动的轴范围也很大，说明当前 G 的轴定义/符号有误。");
  return 0;
}

int run_solve(
  const std::string & yaw_dir, const std::string & pitch_dir, const Eigen::Matrix3d & g_old,
  double min_angle_deg)
{
  if (yaw_dir.empty() || pitch_dir.empty()) {
    utils::logger()->error("[GimbalImu] solve 模式需要 --yaw-dir= 和 --pitch-dir= 两个目录");
    return 1;
  }

  const auto yaw_samples = calibration::enumerate_quaternions(yaw_dir);
  const auto pitch_samples = calibration::enumerate_quaternions(pitch_dir);
  if (yaw_samples.empty() || pitch_samples.empty()) {
    utils::logger()->error(
      "[GimbalImu] 未读到四元数: yaw-dir {} 帧 / pitch-dir {} 帧",
      yaw_samples.size(), pitch_samples.size());
    return 1;
  }

  // G 第3列 = yaw 轴在 IMU 系；第2列 = pitch 轴在 IMU 系。
  const AxisEstimate yaw_est = estimate_axis(yaw_samples, min_angle_deg);
  const AxisEstimate pitch_est = estimate_axis(pitch_samples, min_angle_deg);

  // 符号歧义（取决于扫描方向）：对齐到当前 G 的对应列（当前 G 近似正确）。
  Eigen::Vector3d z_axis = yaw_est.axis;
  Eigen::Vector3d y_axis = pitch_est.axis;
  if (z_axis.dot(g_old.col(2)) < 0.0) {
    z_axis = -z_axis;
  }
  if (y_axis.dot(g_old.col(1)) < 0.0) {
    y_axis = -y_axis;
  }

  const Eigen::Vector3d x_axis = y_axis.cross(z_axis);  // 右手系: x = y × z
  Eigen::Matrix3d g_raw;
  g_raw.col(0) = x_axis;
  g_raw.col(1) = y_axis;
  g_raw.col(2) = z_axis;
  const Eigen::Matrix3d g_new = project_to_so3(g_raw);

  // 与旧 G 的整体夹角。
  const Eigen::AngleAxisd delta(g_old.transpose() * g_new);

  utils::logger()->info("[GimbalImu] ===== solve 诊断 =====");
  utils::logger()->info(
    "[GimbalImu] yaw 段:   有效 {} 帧, 扫描幅度 {:.1f}°, 轴一致性(最大偏差) {:.3f}°",
    yaw_est.used, yaw_est.angle_span_deg, yaw_est.max_dev_deg);
  utils::logger()->info(
    "[GimbalImu] pitch 段: 有效 {} 帧, 扫描幅度 {:.1f}°, 轴一致性(最大偏差) {:.3f}°",
    pitch_est.used, pitch_est.angle_span_deg, pitch_est.max_dev_deg);
  utils::logger()->info(
    "[GimbalImu] 实测 yaw 轴 vs 当前 G 第3列 夹角: {:.3f}°",
    angle_between_deg(z_axis, g_old.col(2)));
  utils::logger()->info(
    "[GimbalImu] 实测 pitch 轴 vs 当前 G 第2列 夹角: {:.3f}°",
    angle_between_deg(y_axis, g_old.col(1)));
  utils::logger()->info(
    "[GimbalImu] 实测 yaw⊥pitch 夹角: {:.3f}° (理想 90°)", angle_between_deg(y_axis, z_axis));
  utils::logger()->info(
    "[GimbalImu] 新 G 行列式 = {:.6f} (应 ≈ +1), 新G vs 旧G 整体夹角 = {:.3f}°",
    g_new.determinant(), delta.angle() * 180.0 / CV_PI);
  utils::logger()->info(
    "[GimbalImu] 质量提示: 轴一致性应 < ~1°、yaw⊥pitch 应接近 90°、扫描幅度建议 > 20°；"
    "否则检查是否真为纯单轴、底盘/IMU 是否固连。");

  std::ostringstream yaml;
  yaml << "    rotation_matrix_gimbal_to_imu:\n" << matrix_to_yaml_data(g_new);
  std::cout << "\n# === 粘贴到 standard3.yaml 的 Solver.coord_converter 下 ===\n"
            << yaml.str() << "\n";
  utils::logger()->info(
    "[GimbalImu] 写回后请用 --mode=check 复验：原本应不动的轴范围应明显变小。");
  return 0;
}

}  // namespace
}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@input          | assets/img_with_q | check 模式输入目录（含四元数 N.txt）}"
    "{config-path c   | src/config/config.yaml | 配置文件路径（读取当前 R_gimbal_to_imu）}"
    "{mode m          | check | check=用当前G打印gimbal欧拉角自检; solve=单轴求解G}"
    "{yaw-dir         | | solve 模式: 纯 yaw 扫描目录}"
    "{pitch-dir       | | solve 模式: 纯 pitch 扫描目录}"
    "{min-angle-deg   | 5.0 | 参与轴估计的最小相对旋转角(度)，过滤噪声}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<std::string>("config-path");
  const std::string mode = cli.get<std::string>("mode");

  try {
    const auto app_config = app_config::AppConfig::load(config_path);
    const Eigen::Matrix3d g =
      Application::unflatten_3x3(app_config.solver.coord_converter.rotation_matrix_gimbal_to_imu);

    if (mode == "solve") {
      return Application::run_solve(
        cli.get<std::string>("yaw-dir"), cli.get<std::string>("pitch-dir"), g,
        cli.get<double>("min-angle-deg"));
    }
    return Application::run_check(cli.get<std::string>(0), g);
  } catch (const std::exception & e) {
    utils::logger()->error("[GimbalImu] 程序异常终止: {}", e.what());
    return 1;
  }
}
