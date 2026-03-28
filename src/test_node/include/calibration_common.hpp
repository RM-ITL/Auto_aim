#ifndef TEST_NODE__CALIBRATION_COMMON_HPP_
#define TEST_NODE__CALIBRATION_COMMON_HPP_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "logger.hpp"
#include "math_tools.hpp"

namespace calibration
{

struct PatternConfig
{
  cv::Size pattern_size{11, 8};
  double center_distance_mm{20.0};
};

struct SamplePaths
{
  int index{0};
  std::string image_path;
  std::string quaternion_path;
  std::string timestamp_path;
};

struct CalibrationSample
{
  int index{0};
  cv::Mat image;
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  std::optional<long long> timestamp_ns;
};

inline std::vector<cv::Point3f> circle_centers_3d(const PatternConfig & config)
{
  std::vector<cv::Point3f> centers;
  centers.reserve(static_cast<size_t>(config.pattern_size.width * config.pattern_size.height));

  for (int row = 0; row < config.pattern_size.height; ++row) {
    for (int col = 0; col < config.pattern_size.width; ++col) {
      centers.emplace_back(
        static_cast<float>(col * config.center_distance_mm),
        static_cast<float>(row * config.center_distance_mm),
        0.0f);
    }
  }

  return centers;
}

inline std::vector<SamplePaths> enumerate_samples(const std::string & input_folder)
{
  std::vector<SamplePaths> samples;
  for (int index = 1;; ++index) {
    SamplePaths sample;
    sample.index = index;
    sample.image_path = input_folder + "/" + std::to_string(index) + ".jpg";
    sample.quaternion_path = input_folder + "/" + std::to_string(index) + ".txt";
    sample.timestamp_path = input_folder + "/" + std::to_string(index) + "_timestamp.txt";

    cv::Mat image = cv::imread(sample.image_path);
    if (image.empty()) {
      break;
    }
    samples.push_back(sample);
  }
  return samples;
}

inline std::optional<long long> read_timestamp_ns(const std::string & timestamp_path)
{
  std::ifstream timestamp_file(timestamp_path);
  if (!timestamp_file.is_open()) {
    return std::nullopt;
  }

  long long timestamp_ns = 0;
  timestamp_file >> timestamp_ns;
  if (!timestamp_file.fail()) {
    return timestamp_ns;
  }
  return std::nullopt;
}

inline Eigen::Quaterniond read_quaternion_wxyz(const std::string & quaternion_path)
{
  std::ifstream quaternion_file(quaternion_path);
  if (!quaternion_file.is_open()) {
    throw std::runtime_error("无法打开四元数文件: " + quaternion_path);
  }

  double w = 0.0, x = 0.0, y = 0.0, z = 0.0;
  quaternion_file >> w >> x >> y >> z;
  if (quaternion_file.fail()) {
    throw std::runtime_error("四元数文件格式错误: " + quaternion_path);
  }

  Eigen::Quaterniond q(w, x, y, z);
  return q.normalized();
}

inline CalibrationSample load_sample(const SamplePaths & paths)
{
  CalibrationSample sample;
  sample.index = paths.index;
  sample.image = cv::imread(paths.image_path);
  if (sample.image.empty()) {
    throw std::runtime_error("无法读取图像: " + paths.image_path);
  }
  sample.q = read_quaternion_wxyz(paths.quaternion_path);
  sample.timestamp_ns = read_timestamp_ns(paths.timestamp_path);
  return sample;
}

inline bool find_circle_centers(
  const cv::Mat & image, const PatternConfig & config, std::vector<cv::Point2f> & centers)
{
  return cv::findCirclesGrid(
    image, config.pattern_size, centers,
    cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING);
}

inline double compute_reprojection_error(
  const std::vector<cv::Point3f> & object_points,
  const std::vector<cv::Point2f> & image_points,
  const cv::Mat & rvec,
  const cv::Mat & tvec,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs)
{
  std::vector<cv::Point2f> reprojected_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, reprojected_points);

  if (reprojected_points.size() != image_points.size() || image_points.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  double total_error = 0.0;
  for (size_t i = 0; i < image_points.size(); ++i) {
    total_error += cv::norm(image_points[i] - reprojected_points[i]);
  }
  return total_error / static_cast<double>(image_points.size());
}

inline std::string format_vector(const std::vector<double> & values, int precision = 10)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

inline std::vector<double> mat_to_row_major_vector(const cv::Mat & mat)
{
  cv::Mat mat64;
  mat.convertTo(mat64, CV_64F);
  std::vector<double> data;
  data.reserve(static_cast<size_t>(mat64.rows * mat64.cols));
  for (int row = 0; row < mat64.rows; ++row) {
    for (int col = 0; col < mat64.cols; ++col) {
      data.push_back(mat64.at<double>(row, col));
    }
  }
  return data;
}

inline std::vector<double> eigen_matrix_to_row_major_vector(const Eigen::Matrix3d & matrix)
{
  std::vector<double> data;
  data.reserve(9);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      data.push_back(matrix(row, col));
    }
  }
  return data;
}

inline std::vector<double> eigen_vector_to_std(const Eigen::Vector3d & vector)
{
  return {vector(0), vector(1), vector(2)};
}

inline std::string make_camera_yaml(
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs,
  double reprojection_error,
  const cv::Size & image_size)
{
  const auto camera_matrix_data = mat_to_row_major_vector(camera_matrix);
  const auto dist_coeffs_data = mat_to_row_major_vector(dist_coeffs.reshape(1, 1));

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(10);
  oss << "CalibParam:\n";
  oss << "  INTRI:\n";
  oss << "    Camera:\n";
  oss << "      - key: /image_topic\n";
  oss << "        value:\n";
  oss << "          polymorphic_id: 2147483649\n";
  oss << "          polymorphic_name: pinhole_brown_t2\n";
  oss << "          ptr_wrapper:\n";
  oss << "            id: 2147483650\n";
  oss << "            data:\n";
  oss << "              img_width: " << image_size.width << "\n";
  oss << "              img_height: " << image_size.height << "\n";
  oss << "              focal_length: "
      << format_vector({camera_matrix_data[0], camera_matrix_data[4]}) << "\n";
  oss << "              principal_point: "
      << format_vector({camera_matrix_data[2], camera_matrix_data[5]}) << "\n";
  oss << "              disto_param: " << format_vector(dist_coeffs_data) << "\n";
  oss << "# mean_reprojection_error_px: " << reprojection_error << "\n";
  return oss.str();
}

inline std::string make_handeye_yaml(
  const Eigen::Matrix3d & r_camera_to_gimbal,
  const Eigen::Matrix3d & r_gimbal_to_imu,
  const Eigen::Vector3d & t_camera_to_gimbal_m,
  const std::optional<Eigen::Matrix3d> & r_board_to_world = std::nullopt,
  const std::optional<Eigen::Vector3d> & t_board_to_world_m = std::nullopt)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(10);
  oss << "Solver:\n";
  oss << "  coord_converter:\n";
  oss << "    rotation_matrix_camera_to_gimbal:\n";
  oss << "      rows: 3\n";
  oss << "      cols: 3\n";
  oss << "      dt: float\n";
  oss << "      data: " << format_vector(eigen_matrix_to_row_major_vector(r_camera_to_gimbal)) << "\n";
  oss << "    rotation_matrix_gimbal_to_imu:\n";
  oss << "      rows: 3\n";
  oss << "      cols: 3\n";
  oss << "      dt: float\n";
  oss << "      data: " << format_vector(eigen_matrix_to_row_major_vector(r_gimbal_to_imu)) << "\n";
  oss << "t_camera_to_gimbal: " << format_vector(eigen_vector_to_std(t_camera_to_gimbal_m)) << "\n";

  if (r_board_to_world.has_value() && t_board_to_world_m.has_value()) {
    oss << "# board_to_world_rotation: "
        << format_vector(eigen_matrix_to_row_major_vector(*r_board_to_world)) << "\n";
    oss << "# board_to_world_translation_m: "
        << format_vector(eigen_vector_to_std(*t_board_to_world_m)) << "\n";
  }
  return oss.str();
}

inline Eigen::Matrix3d read_r_gimbal_to_imu(const std::string & config_path)
{
  const YAML::Node yaml = YAML::LoadFile(config_path);
  const auto matrix_data =
    yaml["Solver"]["coord_converter"]["rotation_matrix_gimbal_to_imu"]["data"].as<std::vector<double>>();

  if (matrix_data.size() != 9) {
    throw std::runtime_error("rotation_matrix_gimbal_to_imu 数据长度不是 9");
  }

  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> matrix(matrix_data.data());
  return matrix;
}

inline void log_gimbal_euler_hint(const Eigen::Matrix3d & r_gimbal_to_world)
{
  const Eigen::Vector3d ypr_deg = utils::eulers(r_gimbal_to_world, 2, 1, 0) * 57.3;
  utils::logger()->info(
    "[Calibration] gimbal yaw/pitch/roll = {:.2f} {:.2f} {:.2f} deg",
    ypr_deg[0], ypr_deg[1], ypr_deg[2]);
}

}  // namespace calibration

#endif  // TEST_NODE__CALIBRATION_COMMON_HPP_
