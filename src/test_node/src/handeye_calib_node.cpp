#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "calibration_common.hpp"
#include "logger.hpp"
#include "math_tools.hpp"

namespace Application
{

class HandeyeCalibApp
{
public:
  HandeyeCalibApp(std::string input_folder, std::string config_path, std::string mode, bool show_detection)
  : input_folder_(std::move(input_folder)),
    config_path_(std::move(config_path)),
    mode_(std::move(mode)),
    show_detection_(show_detection)
  {
  }

  int run()
  {
    const auto sample_paths = calibration::enumerate_samples(input_folder_);
    if (sample_paths.empty()) {
      utils::logger()->error("[HandeyeCalib] 未在 {} 中找到采集图像", input_folder_);
      return 1;
    }

    const auto yaml = YAML::LoadFile(config_path_);
    const Eigen::Matrix3d r_gimbal_to_imu = calibration::read_r_gimbal_to_imu(config_path_);
    const auto camera_node = yaml["CalibParam"]["INTRI"]["Camera"][0]["value"]["ptr_wrapper"]["data"];
    const auto focal_length = camera_node["focal_length"].as<std::vector<double>>();
    const auto principal_point = camera_node["principal_point"].as<std::vector<double>>();
    const auto disto_param = camera_node["disto_param"].as<std::vector<double>>();

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
    for (const auto & paths : sample_paths) {
      const auto sample = calibration::load_sample(paths);
      std::vector<cv::Point2f> centers;
      const bool detected = calibration::find_circle_centers(sample.image, pattern_config_, centers);
      if (!detected) {
        utils::logger()->warn("[HandeyeCalib] sample {:03d}: 圆点板检测失败", sample.index);
        continue;
      }

      cv::Mat rvec;
      cv::Mat tvec;
      const bool pnp_success = cv::solvePnP(
        object_points, centers, camera_matrix, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);
      if (!pnp_success) {
        utils::logger()->warn("[HandeyeCalib] sample {:03d}: solvePnP 失败", sample.index);
        continue;
      }

      const double reprojection_error = calibration::compute_reprojection_error(
        object_points, centers, rvec, tvec, camera_matrix, dist_coeffs);
      utils::logger()->info(
        "[HandeyeCalib] sample {:03d}: reprojection_error = {:.4f} px",
        sample.index, reprojection_error);

      cv::Mat drawing;
      if (show_detection_) {
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

    if (show_detection_) {
      cv::destroyWindow(window_name_);
    }

    if (success_count < 5) {
      utils::logger()->error("[HandeyeCalib] 有效样本过少: {}", success_count);
      return 1;
    }

    cv::Mat r_camera_to_gimbal_cv;
    cv::Mat t_camera_to_gimbal_cv;
    std::optional<Eigen::Matrix3d> r_board_to_world;
    std::optional<Eigen::Vector3d> t_board_to_world_m;

    if (mode_ == "robotworld") {
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
      "[HandeyeCalib] 模式: {}, 有效样本: {}", mode_, success_count);
    std::cout << calibration::make_handeye_yaml(
                   r_camera_to_gimbal, r_gimbal_to_imu, t_camera_to_gimbal_m,
                   r_board_to_world, t_board_to_world_m)
              << std::endl;
    return 0;
  }

private:
  calibration::PatternConfig pattern_config_;
  std::string input_folder_;
  std::string config_path_;
  std::string mode_;
  bool show_detection_{false};
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
    "{show s          | false | 是否显示圆点检测结果}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string input_folder = cli.get<std::string>(0);
  const std::string config_path = cli.get<std::string>("config-path");
  const std::string mode = cli.get<std::string>("mode");
  const bool show_detection = cli.get<bool>("show");

  try {
    Application::HandeyeCalibApp app(input_folder, config_path, mode, show_detection);
    return app.run();
  } catch (const std::exception & e) {
    utils::logger()->error("[HandeyeCalib] 程序异常终止: {}", e.what());
    return 1;
  }
}
