#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef HANDEYE_WITH_CERES
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/types.h>
#include <ceres/version.h>
#endif

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "app_config/app_config.hpp"
#include "calibration_common.hpp"
#include "logger.hpp"
#include "math_tools.hpp"

namespace Application
{

namespace
{
struct ReprojectionStats
{
  double mean_px{std::numeric_limits<double>::infinity()};
  double max_px{std::numeric_limits<double>::infinity()};
  int residual_count{0};
};

struct BoardWorldConsistencyStats
{
  double rotation_mean_deg{std::numeric_limits<double>::infinity()};
  double rotation_max_deg{std::numeric_limits<double>::infinity()};
  double translation_mean_mm{std::numeric_limits<double>::infinity()};
  double translation_max_mm{std::numeric_limits<double>::infinity()};
  cv::Vec3d translation_center_mm{0.0, 0.0, 0.0};
  int sample_count{0};
};

struct BaResult
{
  bool attempted{false};
  bool applied{false};
  std::string status{"not requested"};
  std::string ceres_version;
  int iterations{0};
  double initial_cost{-1.0};
  double final_cost{-1.0};
  ReprojectionStats before_reproj;
  ReprojectionStats after_reproj;
  Eigen::Matrix3d r_camera_to_gimbal{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_camera_to_gimbal_mm{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d r_board_to_world{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_board_to_world_mm{Eigen::Vector3d::Zero()};
};

// solvePnPGeneric 的 reprojectionError 输出是 CV_32F（N×1），
// 直接用 pnp_errors.at<double>(s) 读取会把 float 字节当 double 解读（且会越界），
// 选解逻辑因此失效。这里先 reshape 成 1×N 再 convertTo CV_64F，做真正的类型转换。
std::vector<double> pnp_error_values(const cv::Mat & pnp_errors, int n_solutions)
{
  std::vector<double> values;
  if (pnp_errors.empty() || n_solutions <= 0) {
    return values;
  }
  cv::Mat errors64;
  pnp_errors.reshape(1, 1).convertTo(errors64, CV_64F);
  values.reserve(static_cast<size_t>(n_solutions));
  for (int i = 0; i < n_solutions && i < errors64.cols; ++i) {
    values.push_back(errors64.at<double>(0, i));
  }
  return values;
}

// 两个 3x3 旋转矩阵之间的夹角（度）。
double rotation_angle_deg(const cv::Mat & r_a, const cv::Mat & r_b)
{
  const cv::Mat r = r_a.t() * r_b;
  const double trace = r.at<double>(0, 0) + r.at<double>(1, 1) + r.at<double>(2, 2);
  const double cos_theta = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
  return std::acos(cos_theta) * 180.0 / CV_PI;
}

// P4-残差报告：board 与 world 都固定，故 board→world 在各帧应当一致，
// 其离散度直接反映手眼外参 X 的质量（不依赖真值，是自洽性指标）。
BoardWorldConsistencyStats report_board_to_world_consistency(
  const std::vector<cv::Mat> & r_gimbal_to_world_list,
  const std::vector<cv::Mat> & rvec_target_to_cam_list,
  const std::vector<cv::Mat> & tvec_target_to_cam_list,
  const cv::Mat & r_camera_to_gimbal, const cv::Mat & t_camera_to_gimbal_mm)
{
  BoardWorldConsistencyStats stats;
  const size_t n = r_gimbal_to_world_list.size();
  if (n == 0) {
    return stats;
  }

  std::vector<Eigen::Quaterniond> q_board_to_world;
  std::vector<cv::Vec3d> t_board_to_world_mm;
  q_board_to_world.reserve(n);
  t_board_to_world_mm.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    cv::Mat r_t2c;
    cv::Rodrigues(rvec_target_to_cam_list[i], r_t2c);
    const cv::Mat r_t2w = r_gimbal_to_world_list[i] * r_camera_to_gimbal * r_t2c;
    const cv::Mat t_t2w = r_gimbal_to_world_list[i] *
      (r_camera_to_gimbal * tvec_target_to_cam_list[i] + t_camera_to_gimbal_mm);
    Eigen::Matrix3d r_eigen;
    cv::cv2eigen(r_t2w, r_eigen);
    q_board_to_world.emplace_back(Eigen::Quaterniond(r_eigen).normalized());
    t_board_to_world_mm.emplace_back(
      t_t2w.at<double>(0), t_t2w.at<double>(1), t_t2w.at<double>(2));
  }

  // 平移均值与各帧偏差（mm）。
  cv::Vec3d t_mean(0.0, 0.0, 0.0);
  for (const auto & t : t_board_to_world_mm) {
    t_mean += t;
  }
  t_mean *= 1.0 / static_cast<double>(n);
  double t_dev_sum = 0.0;
  double t_dev_max = 0.0;
  for (const auto & t : t_board_to_world_mm) {
    const double d = cv::norm(t - t_mean);
    t_dev_sum += d;
    t_dev_max = std::max(t_dev_max, d);
  }

  // 旋转：四元数平均作参考，报告各帧角度偏差（度）。
  const Eigen::Quaterniond & q_ref = q_board_to_world.front();
  Eigen::Vector4d acc = Eigen::Vector4d::Zero();
  for (const auto & q : q_board_to_world) {
    Eigen::Vector4d v(q.w(), q.x(), q.y(), q.z());
    if (v.dot(Eigen::Vector4d(q_ref.w(), q_ref.x(), q_ref.y(), q_ref.z())) < 0.0) {
      v = -v;
    }
    acc += v;
  }
  acc.normalize();
  const Eigen::Quaterniond q_mean(acc[0], acc[1], acc[2], acc[3]);
  double r_dev_sum = 0.0;
  double r_dev_max = 0.0;
  for (const auto & q : q_board_to_world) {
    const double a = q_mean.angularDistance(q) * 180.0 / CV_PI;
    r_dev_sum += a;
    r_dev_max = std::max(r_dev_max, a);
  }

  stats.rotation_mean_deg = r_dev_sum / static_cast<double>(n);
  stats.rotation_max_deg = r_dev_max;
  stats.translation_mean_mm = t_dev_sum / static_cast<double>(n);
  stats.translation_max_mm = t_dev_max;
  stats.translation_center_mm = t_mean;
  stats.sample_count = static_cast<int>(n);

  utils::logger()->info(
    "[HandeyeCalib] board→world 自洽性: 旋转偏差 mean={:.3f}° max={:.3f}° | 平移偏差 mean={:.2f}mm max={:.2f}mm",
    stats.rotation_mean_deg, stats.rotation_max_deg, stats.translation_mean_mm,
    stats.translation_max_mm);
  utils::logger()->info(
    "[HandeyeCalib]   board 原点世界系均值 = [{:.1f}, {:.1f}, {:.1f}] mm（仅作量级参考）",
    t_mean[0], t_mean[1], t_mean[2]);
  return stats;
}

// P4-多方法对比：用五种 calibrateHandEye 方法各解一次，报告旋转与首个方法(TSAI)的夹角。
// 旋转高度一致说明结果可信；某方法平移明显跑偏多半是样本姿态覆盖不足。
void report_method_agreement(
  const std::vector<cv::Mat> & r_gripper_to_base, const std::vector<cv::Mat> & t_gripper_to_base,
  const std::vector<cv::Mat> & r_target_to_cam, const std::vector<cv::Mat> & t_target_to_cam)
{
  struct Entry
  {
    const char * name;
    cv::HandEyeCalibrationMethod method;
  };
  const std::vector<Entry> methods = {
    {"TSAI", cv::CALIB_HAND_EYE_TSAI},
    {"PARK", cv::CALIB_HAND_EYE_PARK},
    {"HORAUD", cv::CALIB_HAND_EYE_HORAUD},
    {"ANDREFF", cv::CALIB_HAND_EYE_ANDREFF},
    {"DANIILIDIS", cv::CALIB_HAND_EYE_DANIILIDIS},
  };

  cv::Mat r_ref;
  for (const auto & entry : methods) {
    cv::Mat r_c2g;
    cv::Mat t_c2g;
    try {
      cv::calibrateHandEye(
        r_gripper_to_base, t_gripper_to_base, r_target_to_cam, t_target_to_cam, r_c2g, t_c2g,
        entry.method);
    } catch (const cv::Exception & e) {
      utils::logger()->warn("[HandeyeCalib] 方法 {} 求解失败: {}", entry.name, e.what());
      continue;
    }
    if (r_ref.empty()) {
      r_ref = r_c2g.clone();
    }
    utils::logger()->info(
      "[HandeyeCalib] 方法对比 {:<11}: R 与TSAI差 {:.3f}° | t=[{:.4f}, {:.4f}, {:.4f}] m",
      entry.name, rotation_angle_deg(r_ref, r_c2g),
      t_c2g.at<double>(0) / 1e3, t_c2g.at<double>(1) / 1e3, t_c2g.at<double>(2) / 1e3);
  }
}

double project_point(
  const Eigen::Vector3d & p_camera, const cv::Mat & camera_matrix, const cv::Mat & dist_coeffs,
  cv::Point2d & pixel)
{
  if (p_camera.z() <= 1e-9) {
    pixel = cv::Point2d(
      std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    return std::numeric_limits<double>::infinity();
  }

  const double fx = camera_matrix.at<double>(0, 0);
  const double fy = camera_matrix.at<double>(1, 1);
  const double cx = camera_matrix.at<double>(0, 2);
  const double cy = camera_matrix.at<double>(1, 2);

  const double k1 = dist_coeffs.cols > 0 ? dist_coeffs.at<double>(0, 0) : 0.0;
  const double k2 = dist_coeffs.cols > 1 ? dist_coeffs.at<double>(0, 1) : 0.0;
  const double p1 = dist_coeffs.cols > 2 ? dist_coeffs.at<double>(0, 2) : 0.0;
  const double p2 = dist_coeffs.cols > 3 ? dist_coeffs.at<double>(0, 3) : 0.0;
  const double k3 = dist_coeffs.cols > 4 ? dist_coeffs.at<double>(0, 4) : 0.0;

  const double x = p_camera.x() / p_camera.z();
  const double y = p_camera.y() / p_camera.z();
  const double r2 = x * x + y * y;
  const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const double x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

  pixel.x = fx * x_dist + cx;
  pixel.y = fy * y_dist + cy;
  return 0.0;
}

ReprojectionStats compute_global_reprojection_stats(
  const std::vector<cv::Mat> & r_gimbal_to_world_list,
  const std::vector<std::vector<cv::Point2f>> & image_points_list,
  const std::vector<cv::Point3f> & object_points,
  const Eigen::Matrix3d & r_camera_to_gimbal,
  const Eigen::Vector3d & t_camera_to_gimbal_mm,
  const Eigen::Matrix3d & r_board_to_world,
  const Eigen::Vector3d & t_board_to_world_mm,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs)
{
  ReprojectionStats stats;
  double sum_px = 0.0;
  double max_px = 0.0;
  int count = 0;

  for (size_t frame = 0; frame < r_gimbal_to_world_list.size(); ++frame) {
    Eigen::Matrix3d r_gimbal_to_world;
    cv::cv2eigen(r_gimbal_to_world_list[frame], r_gimbal_to_world);
    const Eigen::Matrix3d r_world_to_gimbal = r_gimbal_to_world.transpose();

    for (size_t i = 0; i < object_points.size(); ++i) {
      const auto & p = object_points[i];
      const Eigen::Vector3d p_board(p.x, p.y, p.z);
      const Eigen::Vector3d p_world = r_board_to_world * p_board + t_board_to_world_mm;
      const Eigen::Vector3d p_gimbal = r_world_to_gimbal * p_world;
      const Eigen::Vector3d p_camera =
        r_camera_to_gimbal.transpose() * (p_gimbal - t_camera_to_gimbal_mm);

      cv::Point2d projected;
      if (!std::isfinite(project_point(p_camera, camera_matrix, dist_coeffs, projected))) {
        continue;
      }
      const cv::Point2d observed(image_points_list[frame][i].x, image_points_list[frame][i].y);
      const double error = cv::norm(projected - observed);
      sum_px += error;
      max_px = std::max(max_px, error);
      ++count;
    }
  }

  if (count > 0) {
    stats.mean_px = sum_px / static_cast<double>(count);
    stats.max_px = max_px;
    stats.residual_count = count;
  }
  return stats;
}

void mat_to_angle_axis_translation(
  const Eigen::Matrix3d & rotation, const Eigen::Vector3d & translation, double * angle_axis,
  double * t)
{
  cv::Mat r_cv;
  cv::eigen2cv(rotation, r_cv);
  cv::Mat rvec_cv;
  cv::Rodrigues(r_cv, rvec_cv);
  angle_axis[0] = rvec_cv.at<double>(0);
  angle_axis[1] = rvec_cv.at<double>(1);
  angle_axis[2] = rvec_cv.at<double>(2);
  t[0] = translation.x();
  t[1] = translation.y();
  t[2] = translation.z();
}

void angle_axis_translation_to_mat(
  const double * angle_axis, const double * t, Eigen::Matrix3d & rotation,
  Eigen::Vector3d & translation)
{
  cv::Mat rvec_cv = (cv::Mat_<double>(3, 1) << angle_axis[0], angle_axis[1], angle_axis[2]);
  cv::Mat r_cv;
  cv::Rodrigues(rvec_cv, r_cv);
  cv::cv2eigen(r_cv, rotation);
  translation = Eigen::Vector3d(t[0], t[1], t[2]);
}

#ifdef HANDEYE_WITH_CERES
struct HandeyeReprojectionCost
{
  HandeyeReprojectionCost(
    Eigen::Matrix3d r_world_to_gimbal, Eigen::Vector3d p_board, cv::Point2d observed,
    double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2,
    double k3)
  : r_world_to_gimbal_(std::move(r_world_to_gimbal)),
    p_board_(std::move(p_board)),
    observed_(observed),
    fx_(fx),
    fy_(fy),
    cx_(cx),
    cy_(cy),
    k1_(k1),
    k2_(k2),
    p1_(p1),
    p2_(p2),
    k3_(k3)
  {
  }

  template<typename T>
  bool operator()(
    const T * const c2g_aa, const T * const c2g_t, const T * const b2w_aa,
    const T * const b2w_t, T * residuals) const
  {
    const T p_board[3] = {T(p_board_.x()), T(p_board_.y()), T(p_board_.z())};
    T p_world_rot[3];
    ceres::AngleAxisRotatePoint(b2w_aa, p_board, p_world_rot);
    const T p_world[3] = {
      p_world_rot[0] + b2w_t[0],
      p_world_rot[1] + b2w_t[1],
      p_world_rot[2] + b2w_t[2],
    };

    T p_gimbal[3];
    for (int row = 0; row < 3; ++row) {
      p_gimbal[row] =
        T(r_world_to_gimbal_(row, 0)) * p_world[0] +
        T(r_world_to_gimbal_(row, 1)) * p_world[1] +
        T(r_world_to_gimbal_(row, 2)) * p_world[2];
    }
    const T p_gimbal_minus_t[3] = {
      p_gimbal[0] - c2g_t[0],
      p_gimbal[1] - c2g_t[1],
      p_gimbal[2] - c2g_t[2],
    };

    T g2c_aa[3] = {-c2g_aa[0], -c2g_aa[1], -c2g_aa[2]};
    T p_camera[3];
    ceres::AngleAxisRotatePoint(g2c_aa, p_gimbal_minus_t, p_camera);

    const T xp = p_camera[0] / p_camera[2];
    const T yp = p_camera[1] / p_camera[2];
    const T r2 = xp * xp + yp * yp;
    const T radial = T(1.0) + T(k1_) * r2 + T(k2_) * r2 * r2 + T(k3_) * r2 * r2 * r2;
    const T x_dist = xp * radial + T(2.0 * p1_) * xp * yp + T(p2_) * (r2 + T(2.0) * xp * xp);
    const T y_dist = yp * radial + T(p1_) * (r2 + T(2.0) * yp * yp) + T(2.0 * p2_) * xp * yp;
    const T u = T(fx_) * x_dist + T(cx_);
    const T v = T(fy_) * y_dist + T(cy_);

    residuals[0] = u - T(observed_.x);
    residuals[1] = v - T(observed_.y);
    return true;
  }

  Eigen::Matrix3d r_world_to_gimbal_;
  Eigen::Vector3d p_board_;
  cv::Point2d observed_;
  double fx_;
  double fy_;
  double cx_;
  double cy_;
  double k1_;
  double k2_;
  double p1_;
  double p2_;
  double k3_;
};

BaResult run_minimal_ba(
  const std::vector<cv::Mat> & r_gimbal_to_world_list,
  const std::vector<std::vector<cv::Point2f>> & image_points_list,
  const std::vector<cv::Point3f> & object_points,
  const Eigen::Matrix3d & initial_r_camera_to_gimbal,
  const Eigen::Vector3d & initial_t_camera_to_gimbal_mm,
  const Eigen::Matrix3d & initial_r_board_to_world,
  const Eigen::Vector3d & initial_t_board_to_world_mm,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs,
  int max_iterations,
  double huber_delta)
{
  BaResult result;
  result.attempted = true;
  result.ceres_version = CERES_VERSION_STRING;
  result.r_camera_to_gimbal = initial_r_camera_to_gimbal;
  result.t_camera_to_gimbal_mm = initial_t_camera_to_gimbal_mm;
  result.r_board_to_world = initial_r_board_to_world;
  result.t_board_to_world_mm = initial_t_board_to_world_mm;
  result.before_reproj = compute_global_reprojection_stats(
    r_gimbal_to_world_list, image_points_list, object_points, initial_r_camera_to_gimbal,
    initial_t_camera_to_gimbal_mm, initial_r_board_to_world, initial_t_board_to_world_mm,
    camera_matrix, dist_coeffs);

  double c2g_aa[3];
  double c2g_t[3];
  double b2w_aa[3];
  double b2w_t[3];
  mat_to_angle_axis_translation(
    initial_r_camera_to_gimbal, initial_t_camera_to_gimbal_mm, c2g_aa, c2g_t);
  mat_to_angle_axis_translation(
    initial_r_board_to_world, initial_t_board_to_world_mm, b2w_aa, b2w_t);

  const double fx = camera_matrix.at<double>(0, 0);
  const double fy = camera_matrix.at<double>(1, 1);
  const double cx = camera_matrix.at<double>(0, 2);
  const double cy = camera_matrix.at<double>(1, 2);
  const double k1 = dist_coeffs.cols > 0 ? dist_coeffs.at<double>(0, 0) : 0.0;
  const double k2 = dist_coeffs.cols > 1 ? dist_coeffs.at<double>(0, 1) : 0.0;
  const double p1 = dist_coeffs.cols > 2 ? dist_coeffs.at<double>(0, 2) : 0.0;
  const double p2 = dist_coeffs.cols > 3 ? dist_coeffs.at<double>(0, 3) : 0.0;
  const double k3 = dist_coeffs.cols > 4 ? dist_coeffs.at<double>(0, 4) : 0.0;

  ceres::Problem problem;
  for (size_t frame = 0; frame < r_gimbal_to_world_list.size(); ++frame) {
    Eigen::Matrix3d r_gimbal_to_world;
    cv::cv2eigen(r_gimbal_to_world_list[frame], r_gimbal_to_world);
    const Eigen::Matrix3d r_world_to_gimbal = r_gimbal_to_world.transpose();
    for (size_t i = 0; i < object_points.size(); ++i) {
      const auto & p = object_points[i];
      auto * cost = new ceres::AutoDiffCostFunction<HandeyeReprojectionCost, 2, 3, 3, 3, 3>(
        new HandeyeReprojectionCost(
          r_world_to_gimbal, Eigen::Vector3d(p.x, p.y, p.z),
          cv::Point2d(image_points_list[frame][i].x, image_points_list[frame][i].y),
          fx, fy, cx, cy, k1, k2, p1, p2, k3));
      ceres::LossFunction * loss = huber_delta > 0.0
        ? static_cast<ceres::LossFunction *>(new ceres::HuberLoss(huber_delta))
        : nullptr;
      problem.AddResidualBlock(cost, loss, c2g_aa, c2g_t, b2w_aa, b2w_t);
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = max_iterations;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  result.iterations = static_cast<int>(summary.iterations.size());
  result.initial_cost = summary.initial_cost;
  result.final_cost = summary.final_cost;
  result.status = summary.BriefReport();

  Eigen::Matrix3d refined_r_camera_to_gimbal;
  Eigen::Vector3d refined_t_camera_to_gimbal_mm;
  Eigen::Matrix3d refined_r_board_to_world;
  Eigen::Vector3d refined_t_board_to_world_mm;
  angle_axis_translation_to_mat(
    c2g_aa, c2g_t, refined_r_camera_to_gimbal, refined_t_camera_to_gimbal_mm);
  angle_axis_translation_to_mat(
    b2w_aa, b2w_t, refined_r_board_to_world, refined_t_board_to_world_mm);

  result.after_reproj = compute_global_reprojection_stats(
    r_gimbal_to_world_list, image_points_list, object_points, refined_r_camera_to_gimbal,
    refined_t_camera_to_gimbal_mm, refined_r_board_to_world, refined_t_board_to_world_mm,
    camera_matrix, dist_coeffs);

  const bool reproj_not_worse =
    result.after_reproj.mean_px <= result.before_reproj.mean_px + 1e-9;
  if (summary.IsSolutionUsable() && reproj_not_worse && std::isfinite(result.after_reproj.mean_px)) {
    result.applied = true;
    result.r_camera_to_gimbal = refined_r_camera_to_gimbal;
    result.t_camera_to_gimbal_mm = refined_t_camera_to_gimbal_mm;
    result.r_board_to_world = refined_r_board_to_world;
    result.t_board_to_world_mm = refined_t_board_to_world_mm;
  }
  return result;
}
#endif
}  // namespace

struct Options
{
  std::string input_folder;
  std::string mode{"handeye"};
  bool show_detection{false};
  bool use_ba{false};
  bool verify_only{false};
  int min_samples{15};
  int ba_max_iterations{80};
  double max_reproj_error{1.5};  // PnP 重投影误差闸门(px)，超过丢弃该帧
  double ambiguity_ratio{1.5};   // IPPE 次优/最优重投影误差比，低于此视为高二义性丢弃
  double ba_huber_delta{1.0};     // BA 像素残差 Huber 阈值(px)，<=0 时不用鲁棒核
};

class HandeyeCalibApp
{
public:
  HandeyeCalibApp(app_config::AppConfig app_config, Options options)
  : app_config_(std::move(app_config)), opt_(std::move(options))
  {
  }

  int run()
  {
#ifdef HANDEYE_WITH_CERES
    utils::logger()->info("[HandeyeCalib] build: BA support = ON (Ceres {})", CERES_VERSION_STRING);
#else
    utils::logger()->info("[HandeyeCalib] build: BA support = OFF (构建时未找到 Ceres)");
#endif

    const auto sample_paths = calibration::enumerate_samples(opt_.input_folder);
    if (sample_paths.empty()) {
      utils::logger()->error("[HandeyeCalib] 未在 {} 中找到采集图像", opt_.input_folder);
      return 1;
    }

    const auto unflatten_3x3 = [](
      const std::vector<double> & data, const std::string & field_name) -> Eigen::Matrix3d {
      if (data.size() != 9) {
        throw std::runtime_error(field_name + " 数据长度不是 9");
      }
      return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(data.data());
    };

    const Eigen::Matrix3d r_gimbal_to_imu = unflatten_3x3(
      app_config_.solver.coord_converter.rotation_matrix_gimbal_to_imu,
      "rotation_matrix_gimbal_to_imu");
    const auto & focal_length = app_config_.solver.camera_intri.focal_length;
    const auto & principal_point = app_config_.solver.camera_intri.principal_point;
    const auto & disto_param = app_config_.solver.camera_intri.disto_param;

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = focal_length.at(0);
    camera_matrix.at<double>(1, 1) = focal_length.at(1);
    camera_matrix.at<double>(0, 2) = principal_point.at(0);
    camera_matrix.at<double>(1, 2) = principal_point.at(1);

    cv::Mat dist_coeffs(1, static_cast<int>(disto_param.size()), CV_64F);
    for (size_t i = 0; i < disto_param.size(); ++i) {
      dist_coeffs.at<double>(0, static_cast<int>(i)) = disto_param[i];
    }

    const auto object_points = calibration::chessboard_corners_3d(pattern_config_);

    std::vector<cv::Mat> r_gripper_to_base_list;
    std::vector<cv::Mat> t_gripper_to_base_list;
    std::vector<cv::Mat> r_target_to_cam_list;
    std::vector<cv::Mat> t_target_to_cam_list;
    std::vector<std::vector<cv::Point2f>> image_points_list;

    std::vector<cv::Mat> r_world_to_gimbal_list;
    std::vector<cv::Mat> t_world_to_gimbal_list;

    int success_count = 0;
    int pnp_fail_count = 0;
    int ambiguous_count = 0;
    int gated_count = 0;
    for (const auto & paths : sample_paths) {
      const auto sample = calibration::load_sample(paths);
      std::vector<cv::Point2f> centers;
      const bool detected = calibration::find_chessboard_corners(sample.image, pattern_config_, centers);
      if (!detected) {
        utils::logger()->warn("[HandeyeCalib] sample {:03d}: 棋盘格角点检测失败", sample.index);
        continue;
      }

      // P1: solvePnPGeneric 返回平面靶标的两个候选位姿，按重投影误差消歧，
      // 而非 IPPE 单解硬选——避免一帧翻转污染整个闭式解。
      std::vector<cv::Mat> pnp_rvecs;
      std::vector<cv::Mat> pnp_tvecs;
      cv::Mat pnp_errors;
      int n_solutions = 0;
      try {
        n_solutions = cv::solvePnPGeneric(
          object_points, centers, camera_matrix, dist_coeffs, pnp_rvecs, pnp_tvecs, false,
          cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(), pnp_errors);
      } catch (const cv::Exception & e) {
        utils::logger()->warn("[HandeyeCalib] sample {:03d}: solvePnPGeneric 异常: {}", sample.index, e.what());
        ++pnp_fail_count;
        continue;
      }
      if (n_solutions < 1 || pnp_rvecs.empty()) {
        utils::logger()->warn("[HandeyeCalib] sample {:03d}: solvePnP 无解", sample.index);
        ++pnp_fail_count;
        continue;
      }

      const auto pnp_errors_vec = pnp_error_values(pnp_errors, n_solutions);
      if (pnp_errors_vec.size() != static_cast<size_t>(n_solutions)) {
        utils::logger()->warn(
          "[HandeyeCalib] sample {:03d}: 无法读取 PnP 重投影误差，丢弃", sample.index);
        ++pnp_fail_count;
        continue;
      }

      int best = 0;
      for (int s = 1; s < n_solutions; ++s) {
        if (pnp_errors_vec[s] < pnp_errors_vec[best]) {
          best = s;
        }
      }
      cv::Mat rvec = pnp_rvecs[best].clone();
      cv::Mat tvec = pnp_tvecs[best].clone();

      // P1: 最优/次优重投影误差比过小 → 二义性强（多见于接近正对），整帧丢弃。
      if (n_solutions >= 2) {
        const double best_err = pnp_errors_vec[best];
        double second_err = std::numeric_limits<double>::infinity();
        for (int s = 0; s < n_solutions; ++s) {
          if (s != best) {
            second_err = std::min(second_err, pnp_errors_vec[s]);
          }
        }
        const double ratio =
          best_err > 1e-9 ? second_err / best_err : std::numeric_limits<double>::infinity();
        if (ratio < opt_.ambiguity_ratio) {
          utils::logger()->warn(
            "[HandeyeCalib] sample {:03d}: 高二义性帧 (次优/最优={:.2f} < {:.2f})，丢弃",
            sample.index, ratio, opt_.ambiguity_ratio);
          ++ambiguous_count;
          continue;
        }
      }

      // P2: 重投影误差闸门——超阈值的坏帧不进 AX=XB。
      const double reprojection_error = calibration::compute_reprojection_error(
        object_points, centers, rvec, tvec, camera_matrix, dist_coeffs);
      if (reprojection_error > opt_.max_reproj_error) {
        utils::logger()->warn(
          "[HandeyeCalib] sample {:03d}: 重投影误差 {:.4f} px > 闸门 {:.4f} px，丢弃",
          sample.index, reprojection_error, opt_.max_reproj_error);
        ++gated_count;
        continue;
      }
      utils::logger()->info(
        "[HandeyeCalib] sample {:03d}: reprojection_error = {:.4f} px (解 {}/{})",
        sample.index, reprojection_error, best + 1, n_solutions);

      cv::Mat drawing;
      if (opt_.show_detection) {
        drawing = sample.image.clone();
        cv::drawChessboardCorners(drawing, pattern_config_.pattern_size, centers, true);
        cv::imshow(window_name_, drawing);
        cv::waitKey(1);
      }

      const Eigen::Matrix3d r_imubody_to_imuabs = sample.q.toRotationMatrix();
      const Eigen::Matrix3d r_gimbal_to_world =
        r_gimbal_to_imu.transpose() * r_imubody_to_imuabs * r_gimbal_to_imu;
      calibration::log_gimbal_euler_hint(r_gimbal_to_world);

      cv::Mat r_gimbal_to_world_cv;
      cv::eigen2cv(r_gimbal_to_world, r_gimbal_to_world_cv);
      cv::Mat r_world_to_gimbal_cv;
      cv::transpose(r_gimbal_to_world_cv, r_world_to_gimbal_cv);

      const cv::Mat t_zero = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
      r_gripper_to_base_list.push_back(r_gimbal_to_world_cv);
      t_gripper_to_base_list.push_back(t_zero);
      r_target_to_cam_list.push_back(rvec.clone());
      t_target_to_cam_list.push_back(tvec.clone());
      image_points_list.push_back(centers);
      r_world_to_gimbal_list.push_back(r_world_to_gimbal_cv);
      t_world_to_gimbal_list.push_back(t_zero.clone());
      ++success_count;
    }

    if (opt_.show_detection) {
      cv::destroyWindow(window_name_);
    }

    utils::logger()->info(
      "[HandeyeCalib] 采样统计: 保留 {} 帧 | PnP失败 {} | 高二义性丢弃 {} | 重投影超阈值丢弃 {}",
      success_count, pnp_fail_count, ambiguous_count, gated_count);

    if (success_count < opt_.min_samples) {
      utils::logger()->error(
        "[HandeyeCalib] 有效样本过少: {}，需要 >= {} 张（建议 15-30，且 yaw 与 pitch 都要散开成二维网格，勿只动单轴）",
        success_count, opt_.min_samples);
      return 1;
    }

    if (opt_.verify_only) {
      const auto & coord_converter = app_config_.solver.coord_converter;
      const Eigen::Matrix3d r_camera_to_gimbal = unflatten_3x3(
        coord_converter.rotation_matrix_camera_to_gimbal,
        "rotation_matrix_camera_to_gimbal");
      if (coord_converter.t_camera_to_gimbal.size() != 3) {
        throw std::runtime_error("t_camera_to_gimbal 数据长度不是 3");
      }
      const Eigen::Vector3d t_camera_to_gimbal_m(
        coord_converter.t_camera_to_gimbal[0], coord_converter.t_camera_to_gimbal[1],
        coord_converter.t_camera_to_gimbal[2]);
      const Eigen::Vector3d t_camera_to_gimbal_mm = t_camera_to_gimbal_m * 1e3;

      cv::Mat r_camera_to_gimbal_cv;
      cv::Mat t_camera_to_gimbal_cv;
      cv::eigen2cv(r_camera_to_gimbal, r_camera_to_gimbal_cv);
      cv::eigen2cv(t_camera_to_gimbal_mm, t_camera_to_gimbal_cv);

      const double orthogonality_error =
        (r_camera_to_gimbal.transpose() * r_camera_to_gimbal - Eigen::Matrix3d::Identity()).norm();
      const double determinant = r_camera_to_gimbal.determinant();
      const bool finite = r_camera_to_gimbal.allFinite() && t_camera_to_gimbal_m.allFinite() &&
        std::isfinite(orthogonality_error) && std::isfinite(determinant);
      utils::logger()->info(
        "[HandeyeCalib] verify-only: 使用配置中的固定 Camera→Gimbal 外参，不执行闭式求解或 BA");
      utils::logger()->info(
        "[HandeyeCalib] verify-only: lens={}, R={}, t=[{:.6f}, {:.6f}, {:.6f}] m",
        app_config_.camera.lens,
        calibration::format_vector(
          calibration::eigen_matrix_to_row_major_vector(r_camera_to_gimbal)),
        t_camera_to_gimbal_m.x(), t_camera_to_gimbal_m.y(), t_camera_to_gimbal_m.z());
      utils::logger()->info(
        "[HandeyeCalib] verify-only: R 正交误差={:.3e}, det={:.9f}",
        orthogonality_error, determinant);
      if (!finite || orthogonality_error > 1e-3 || std::abs(determinant - 1.0) > 1e-3) {
        throw std::runtime_error(
          "verify-only: 配置外参包含非有限值，或旋转矩阵不是有效的 SO(3) 旋转");
      }
      if (opt_.use_ba) {
        utils::logger()->warn("[HandeyeCalib] verify-only: 已忽略 --ba=true");
      }

      const auto stats = report_board_to_world_consistency(
        r_gripper_to_base_list, r_target_to_cam_list, t_target_to_cam_list,
        r_camera_to_gimbal_cv, t_camera_to_gimbal_cv);
      utils::logger()->info(
        "[HandeyeCalib] VERIFY_RESULT input={} samples={} rotation_mean_deg={:.6f} "
        "rotation_max_deg={:.6f} translation_mean_mm={:.6f} translation_max_mm={:.6f}",
        opt_.input_folder, stats.sample_count, stats.rotation_mean_deg, stats.rotation_max_deg,
        stats.translation_mean_mm, stats.translation_max_mm);
      return 0;
    }

    cv::Mat r_camera_to_gimbal_cv;
    cv::Mat t_camera_to_gimbal_cv;
    std::optional<Eigen::Matrix3d> r_board_to_world;
    std::optional<Eigen::Vector3d> t_board_to_world_m;

    if (opt_.mode == "robotworld") {
      cv::Mat r_gimbal_to_camera_cv;
      cv::Mat t_gimbal_to_camera_cv;
      cv::Mat r_world_to_board_cv;
      cv::Mat t_world_to_board_cv;
      cv::calibrateRobotWorldHandEye(
        r_target_to_cam_list, t_target_to_cam_list,
        r_world_to_gimbal_list, t_world_to_gimbal_list,
        r_world_to_board_cv, t_world_to_board_cv,
        r_gimbal_to_camera_cv, t_gimbal_to_camera_cv);

      cv::transpose(r_gimbal_to_camera_cv, r_camera_to_gimbal_cv);
      t_camera_to_gimbal_cv = -r_camera_to_gimbal_cv * t_gimbal_to_camera_cv;

      cv::Mat r_board_to_world_cv;
      cv::transpose(r_world_to_board_cv, r_board_to_world_cv);
      cv::Mat t_board_to_world_cv = -r_board_to_world_cv * t_world_to_board_cv;

      Eigen::Matrix3d r_board_to_world_eigen;
      cv::cv2eigen(r_board_to_world_cv, r_board_to_world_eigen);
      Eigen::Vector3d t_board_to_world_eigen;
      cv::cv2eigen(t_board_to_world_cv, t_board_to_world_eigen);
      r_board_to_world = r_board_to_world_eigen;
      t_board_to_world_m = t_board_to_world_eigen / 1e3;
    } else {
      cv::calibrateHandEye(
        r_gripper_to_base_list, t_gripper_to_base_list,
        r_target_to_cam_list, t_target_to_cam_list,
        r_camera_to_gimbal_cv, t_camera_to_gimbal_cv);
    }

    Eigen::Matrix3d r_camera_to_gimbal;
    cv::cv2eigen(r_camera_to_gimbal_cv, r_camera_to_gimbal);
    Eigen::Vector3d t_camera_to_gimbal_mm;
    cv::cv2eigen(t_camera_to_gimbal_cv, t_camera_to_gimbal_mm);

    BaResult ba_result;
    if (opt_.use_ba) {
      if (opt_.mode != "robotworld") {
        ba_result.attempted = true;
        ba_result.status = "requested but skipped: BA 需要 -m=robotworld 提供 board_to_world 初值";
        utils::logger()->warn(
          "[HandeyeCalib] BA: 你传了 --ba=true，但当前 mode={}，BA 需要 -m=robotworld 初值，已退回闭式解",
          opt_.mode);
      } else if (!r_board_to_world.has_value() || !t_board_to_world_m.has_value()) {
        ba_result.attempted = true;
        ba_result.status = "requested but skipped: missing robotworld initial board_to_world";
        utils::logger()->warn(
          "[HandeyeCalib] BA: robotworld 未产生 board_to_world 初值，已退回闭式解");
      } else {
#ifdef HANDEYE_WITH_CERES
        utils::logger()->info(
          "[HandeyeCalib] BA: 启用 (ceres {}, max_iter={}, huber_delta={:.3f}px)",
          CERES_VERSION_STRING, opt_.ba_max_iterations, opt_.ba_huber_delta);
        ba_result = run_minimal_ba(
          r_gripper_to_base_list, image_points_list, object_points, r_camera_to_gimbal,
          t_camera_to_gimbal_mm, *r_board_to_world, *t_board_to_world_m * 1e3, camera_matrix,
          dist_coeffs, opt_.ba_max_iterations, opt_.ba_huber_delta);
        utils::logger()->info(
          "[HandeyeCalib] BA: 迭代 {} 次, cost {:.6g} -> {:.6g}, {}",
          ba_result.iterations, ba_result.initial_cost, ba_result.final_cost, ba_result.status);
        utils::logger()->info(
          "[HandeyeCalib] BA: 全局重投影 mean {:.4f}->{:.4f}px | max {:.4f}->{:.4f}px | residuals {}",
          ba_result.before_reproj.mean_px, ba_result.after_reproj.mean_px,
          ba_result.before_reproj.max_px, ba_result.after_reproj.max_px,
          ba_result.after_reproj.residual_count);
        if (ba_result.applied) {
          r_camera_to_gimbal = ba_result.r_camera_to_gimbal;
          t_camera_to_gimbal_mm = ba_result.t_camera_to_gimbal_mm;
          r_board_to_world = ba_result.r_board_to_world;
          t_board_to_world_m = ba_result.t_board_to_world_mm / 1e3;
          cv::eigen2cv(r_camera_to_gimbal, r_camera_to_gimbal_cv);
          cv::eigen2cv(t_camera_to_gimbal_mm, t_camera_to_gimbal_cv);
          utils::logger()->info("[HandeyeCalib] BA: 已采用 BA 优化结果");
        } else {
          utils::logger()->warn("[HandeyeCalib] BA: 未采用 BA 结果，保留 robotworld 闭式解");
        }
#else
        ba_result.attempted = true;
        ba_result.status = "requested but unavailable: not compiled with Ceres";
        utils::logger()->warn(
          "[HandeyeCalib] BA: 你传了 --ba=true，但此二进制未编译 Ceres 支持，已退回闭式解");
#endif
      }
    } else {
#ifdef HANDEYE_WITH_CERES
      utils::logger()->info("[HandeyeCalib] BA: 已编译但未启用 (--ba=false)，仅输出闭式解");
#else
      utils::logger()->info("[HandeyeCalib] BA: 未编译且未请求，仅输出闭式解");
#endif
    }

    const Eigen::Vector3d t_camera_to_gimbal_m = t_camera_to_gimbal_mm / 1e3;

    // P4: 残差/一致性报告。
    report_board_to_world_consistency(
      r_gripper_to_base_list, r_target_to_cam_list, t_target_to_cam_list, r_camera_to_gimbal_cv,
      t_camera_to_gimbal_cv);
    if (opt_.mode != "robotworld") {
      report_method_agreement(
        r_gripper_to_base_list, t_gripper_to_base_list, r_target_to_cam_list, t_target_to_cam_list);
    }

    utils::logger()->info(
      "[HandeyeCalib] 模式: {}, 有效样本: {}", opt_.mode, success_count);
    std::ostringstream provenance;
    provenance << "handeye solver: " << opt_.mode;
    if (ba_result.applied) {
      provenance << " + BA(ceres " << ba_result.ceres_version << "), reproj "
                 << std::fixed << std::setprecision(4)
                 << ba_result.before_reproj.mean_px << "->" << ba_result.after_reproj.mean_px
                 << "px";
    } else if (ba_result.attempted) {
      provenance << " closed-form (BA not applied: " << ba_result.status << ")";
    } else {
      provenance << " closed-form (BA disabled)";
    }
    std::cout << calibration::make_handeye_yaml(
                   r_camera_to_gimbal, r_gimbal_to_imu, t_camera_to_gimbal_m,
                   r_board_to_world, t_board_to_world_m, provenance.str())
              << std::endl;
    return 0;
  }

private:
  calibration::PatternConfig pattern_config_;
  app_config::AppConfig app_config_;
  Options opt_;
  const std::string window_name_{"handeye_calib_node"};
};

}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@input-folder   | assets/img_with_q | 输入数据文件夹}"
    "{config-path c   | src/config/config.yaml | 配置文件路径}"
    "{mode m          | handeye | 标定模式: handeye 或 robotworld}"
    "{show s          | false | 是否显示棋盘格角点检测结果}"
    "{ba              | false | 是否启用最小 BA（需 -m=robotworld 且编译期找到 Ceres）}"
    "{verify-only     | false | 固定使用配置外参，仅回代检查 board→world 自洽性，不重新求解}"
    "{min-samples     | 15    | 最少有效样本数（建议 15-30，yaw/pitch 二维散开）}"
    "{max-reproj-error| 1.5   | PnP 重投影误差闸门(px)，超过丢弃该帧}"
    "{ambiguity-ratio | 1.5   | IPPE 次优/最优重投影误差比，低于此视为高二义性丢弃}"
    "{ba-max-iter     | 80    | BA 最大迭代次数}"
    "{ba-huber-delta  | 1.0   | BA 像素残差 Huber 阈值(px)，<=0 表示不用鲁棒核}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<std::string>("config-path");

  Application::Options options;
  options.input_folder = cli.get<std::string>(0);
  options.mode = cli.get<std::string>("mode");
  options.show_detection = cli.get<bool>("show");
  options.use_ba = cli.get<bool>("ba");
  options.verify_only = cli.get<bool>("verify-only");
  options.min_samples = cli.get<int>("min-samples");
  options.max_reproj_error = cli.get<double>("max-reproj-error");
  options.ambiguity_ratio = cli.get<double>("ambiguity-ratio");
  options.ba_max_iterations = cli.get<int>("ba-max-iter");
  options.ba_huber_delta = cli.get<double>("ba-huber-delta");

  try {
    Application::HandeyeCalibApp app(app_config::AppConfig::load(config_path), std::move(options));
    return app.run();
  } catch (const std::exception & e) {
    utils::logger()->error("[HandeyeCalib] 程序异常终止: {}", e.what());
    return 1;
  }
}
