#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "app_config/app_config.hpp"
#include "calibration_common.hpp"
#include "logger.hpp"
#include "math_tools.hpp"

namespace Application
{

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
