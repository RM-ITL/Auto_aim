#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
void report_board_to_world_consistency(
  const std::vector<cv::Mat> & r_gimbal_to_world_list,
  const std::vector<cv::Mat> & rvec_target_to_cam_list,
  const std::vector<cv::Mat> & tvec_target_to_cam_list,
  const cv::Mat & r_camera_to_gimbal, const cv::Mat & t_camera_to_gimbal_mm)
{
  const size_t n = r_gimbal_to_world_list.size();
  if (n == 0) {
    return;
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

  utils::logger()->info(
    "[HandeyeCalib] board→world 自洽性: 旋转偏差 mean={:.3f}° max={:.3f}° | 平移偏差 mean={:.2f}mm max={:.2f}mm",
    r_dev_sum / static_cast<double>(n), r_dev_max, t_dev_sum / static_cast<double>(n), t_dev_max);
  utils::logger()->info(
    "[HandeyeCalib]   board 原点世界系均值 = [{:.1f}, {:.1f}, {:.1f}] mm（仅作量级参考）",
    t_mean[0], t_mean[1], t_mean[2]);
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
}  // namespace

struct Options
{
  std::string input_folder;
  std::string mode{"handeye"};
  bool show_detection{false};
  int min_samples{15};
  double max_reproj_error{1.5};  // PnP 重投影误差闸门(px)，超过丢弃该帧
  double ambiguity_ratio{1.5};   // IPPE 次优/最优重投影误差比，低于此视为高二义性丢弃
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
    const auto sample_paths = calibration::enumerate_samples(opt_.input_folder);
    if (sample_paths.empty()) {
      utils::logger()->error("[HandeyeCalib] 未在 {} 中找到采集图像", opt_.input_folder);
      return 1;
    }

    const auto unflatten_3x3 = [](const std::vector<double> & data) -> Eigen::Matrix3d {
      if (data.size() != 9) {
        throw std::runtime_error("rotation_matrix_gimbal_to_imu 数据长度不是 9");
      }
      return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(data.data());
    };

    const Eigen::Matrix3d r_gimbal_to_imu =
      unflatten_3x3(app_config_.solver.coord_converter.rotation_matrix_gimbal_to_imu);
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

    const auto object_points = calibration::circle_centers_3d(pattern_config_);

    std::vector<cv::Mat> r_gripper_to_base_list;
    std::vector<cv::Mat> t_gripper_to_base_list;
    std::vector<cv::Mat> r_target_to_cam_list;
    std::vector<cv::Mat> t_target_to_cam_list;

    std::vector<cv::Mat> r_world_to_gimbal_list;
    std::vector<cv::Mat> t_world_to_gimbal_list;

    int success_count = 0;
    int pnp_fail_count = 0;
    int ambiguous_count = 0;
    int gated_count = 0;
    for (const auto & paths : sample_paths) {
      const auto sample = calibration::load_sample(paths);
      std::vector<cv::Point2f> centers;
      const bool detected = calibration::find_circle_centers(sample.image, pattern_config_, centers);
      if (!detected) {
        utils::logger()->warn("[HandeyeCalib] sample {:03d}: 圆点板检测失败", sample.index);
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

      int best = 0;
      for (int s = 1; s < n_solutions; ++s) {
        if (pnp_errors.at<double>(s) < pnp_errors.at<double>(best)) {
          best = s;
        }
      }
      cv::Mat rvec = pnp_rvecs[best].clone();
      cv::Mat tvec = pnp_tvecs[best].clone();

      // P1: 最优/次优重投影误差比过小 → 二义性强（多见于接近正对），整帧丢弃。
      if (n_solutions >= 2) {
        const double best_err = pnp_errors.at<double>(best);
        double second_err = std::numeric_limits<double>::infinity();
        for (int s = 0; s < n_solutions; ++s) {
          if (s != best) {
            second_err = std::min(second_err, pnp_errors.at<double>(s));
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
    std::cout << calibration::make_handeye_yaml(
                   r_camera_to_gimbal, r_gimbal_to_imu, t_camera_to_gimbal_m,
                   r_board_to_world, t_board_to_world_m)
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
    "{show s          | false | 是否显示圆点检测结果}"
    "{min-samples     | 15    | 最少有效样本数（建议 15-30，yaw/pitch 二维散开）}"
    "{max-reproj-error| 1.5   | PnP 重投影误差闸门(px)，超过丢弃该帧}"
    "{ambiguity-ratio | 1.5   | IPPE 次优/最优重投影误差比，低于此视为高二义性丢弃}";

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
  options.min_samples = cli.get<int>("min-samples");
  options.max_reproj_error = cli.get<double>("max-reproj-error");
  options.ambiguity_ratio = cli.get<double>("ambiguity-ratio");

  try {
    Application::HandeyeCalibApp app(app_config::AppConfig::load(config_path), std::move(options));
    return app.run();
  } catch (const std::exception & e) {
    utils::logger()->error("[HandeyeCalib] 程序异常终止: {}", e.what());
    return 1;
  }
}
