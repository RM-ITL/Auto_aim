#include <cfloat>
#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "calibration_common.hpp"
#include "logger.hpp"

namespace Application
{

class CameraCalibApp
{
public:
  CameraCalibApp(std::string input_folder, bool show_detection)
  : input_folder_(std::move(input_folder)), show_detection_(show_detection)
  {
  }

  int run()
  {
    const auto sample_paths = calibration::enumerate_samples(input_folder_);
    if (sample_paths.empty()) {
      utils::logger()->error("[CameraCalib] 未在 {} 中找到采集图像", input_folder_);
      return 1;
    }

    const auto object_points_template = calibration::circle_centers_3d(pattern_config_);
    std::vector<std::vector<cv::Point3f>> object_points_list;
    std::vector<std::vector<cv::Point2f>> image_points_list;
    cv::Size image_size;

    int success_count = 0;
    for (const auto & paths : sample_paths) {
      cv::Mat image = cv::imread(paths.image_path);
      if (image.empty()) {
        utils::logger()->warn("[CameraCalib] 跳过无法读取的图像 {}", paths.image_path);
        continue;
      }
      if (image_size.width == 0) {
        image_size = image.size();
      }

      std::vector<cv::Point2f> centers;
      const bool success = calibration::find_circle_centers(image, pattern_config_, centers);
      utils::logger()->info(
        "[CameraCalib] sample {:03d}: {}",
        paths.index, success ? "detected" : "missed");

      if (show_detection_) {
        cv::Mat drawing = image.clone();
        cv::drawChessboardCorners(drawing, pattern_config_.pattern_size, centers, success);
        cv::imshow(window_name_, drawing);
        cv::waitKey(1);
      }

      if (!success) {
        continue;
      }

      image_points_list.push_back(centers);
      object_points_list.push_back(object_points_template);
      ++success_count;
    }

    if (show_detection_) {
      cv::destroyWindow(window_name_);
    }

    if (success_count < 5) {
      utils::logger()->error("[CameraCalib] 有效样本过少: {}", success_count);
      return 1;
    }

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    const auto criteria = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);

    cv::calibrateCamera(
      object_points_list, image_points_list, image_size, camera_matrix, dist_coeffs, rvecs, tvecs,
      cv::CALIB_FIX_K3, criteria);

    double error_sum = 0.0;
    size_t total_points = 0;
    for (size_t i = 0; i < object_points_list.size(); ++i) {
      std::vector<cv::Point2f> reprojected_points;
      cv::projectPoints(
        object_points_list[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs, reprojected_points);
      total_points += reprojected_points.size();
      for (size_t j = 0; j < reprojected_points.size(); ++j) {
        error_sum += cv::norm(image_points_list[i][j] - reprojected_points[j]);
      }
    }
    const double mean_error = total_points > 0 ? error_sum / static_cast<double>(total_points) : 0.0;

    utils::logger()->info(
      "[CameraCalib] 检测成功 {}/{} 张，平均重投影误差 {:.4f} px",
      success_count, sample_paths.size(), mean_error);

    std::cout << calibration::make_camera_yaml(camera_matrix, dist_coeffs, mean_error, image_size)
              << std::endl;
    return 0;
  }

private:
  calibration::PatternConfig pattern_config_;
  std::string input_folder_;
  bool show_detection_{false};
  const std::string window_name_{"camera_calib_node"};
};

}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@input-folder   | assets/img_with_q | 输入数据文件夹}"
    "{show s          | false | 是否显示圆点检测结果}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string input_folder = cli.get<std::string>(0);
  const bool show_detection = cli.get<bool>("show");

  try {
    Application::CameraCalibApp app(input_folder, show_detection);
    return app.run();
  } catch (const std::exception & e) {
    utils::logger()->error("[CameraCalib] 程序异常终止: {}", e.what());
    return 1;
  }
}
