#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "calibration_common.hpp"
#include "logger.hpp"

namespace Application
{

namespace
{
double vector_mean(const std::vector<double> & values)
{
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const double v : values) {
    sum += v;
  }
  return sum / static_cast<double>(values.size());
}

double vector_std(const std::vector<double> & values, double mean)
{
  if (values.size() < 2) {
    return 0.0;
  }
  double sum_sq = 0.0;
  for (const double v : values) {
    const double d = v - mean;
    sum_sq += d * d;
  }
  return std::sqrt(sum_sq / static_cast<double>(values.size()));
}
}  // namespace

struct Options
{
  std::string input_folder;
  bool show_detection{false};
  int min_samples{15};
  double max_rms{1.0};
  bool free_k3{false};
  double reject_sigma{3.0};
  double reject_px{2.0};
  int max_reject_iters{3};
  bool no_reject{false};
  int holdout{0};
};

class CameraCalibApp
{
public:
  explicit CameraCalibApp(Options options) : opt_(std::move(options)) {}

  int run()
  {
    const auto sample_paths = calibration::enumerate_samples(opt_.input_folder);
    if (sample_paths.empty()) {
      utils::logger()->error("[CameraCalib] 未在 {} 中找到采集图像", opt_.input_folder);
      return 1;
    }

    const auto object_points_template = calibration::circle_centers_3d(pattern_config_);
    std::vector<std::vector<cv::Point3f>> all_object_points;
    std::vector<std::vector<cv::Point2f>> all_image_points;
    std::vector<int> all_indices;
    cv::Size image_size;

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
        "[CameraCalib] sample {:03d}: {}", paths.index, success ? "detected" : "missed");

      if (opt_.show_detection) {
        cv::Mat drawing = image.clone();
        cv::drawChessboardCorners(drawing, pattern_config_.pattern_size, centers, success);
        cv::imshow(window_name_, drawing);
        cv::waitKey(1);
      }

      if (!success) {
        continue;
      }

      all_image_points.push_back(centers);
      all_object_points.push_back(object_points_template);
      all_indices.push_back(paths.index);
    }

    if (opt_.show_detection) {
      cv::destroyWindow(window_name_);
    }

    const int detected = static_cast<int>(all_indices.size());
    if (detected < opt_.min_samples) {
      utils::logger()->error(
        "[CameraCalib] 有效样本过少: {}/{}，需要 >= {} 张（建议 15-30，覆盖画面中心/边缘/四角与多种尺度）",
        detected, sample_paths.size(), opt_.min_samples);
      return 1;
    }

    // 留出验证划分：等间隔抽取（确定性，无随机），训练集需保证 >= min_samples。
    std::vector<bool> is_holdout(detected, false);
    int holdout = opt_.holdout;
    if (holdout > 0) {
      if (detected - holdout < opt_.min_samples) {
        utils::logger()->warn(
          "[CameraCalib] 留出 {} 帧后训练集不足 {} 张，已禁用留出验证", holdout, opt_.min_samples);
        holdout = 0;
      } else {
        for (int k = 0; k < holdout; ++k) {
          size_t pos = static_cast<size_t>((k + 0.5) * detected / holdout);
          if (pos >= static_cast<size_t>(detected)) {
            pos = static_cast<size_t>(detected - 1);
          }
          while (is_holdout[pos] && pos + 1 < static_cast<size_t>(detected)) {
            ++pos;
          }
          is_holdout[pos] = true;
        }
      }
    }

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<int> indices;
    std::vector<std::vector<cv::Point3f>> holdout_object_points;
    std::vector<std::vector<cv::Point2f>> holdout_image_points;
    std::vector<int> holdout_indices;
    for (int i = 0; i < detected; ++i) {
      if (is_holdout[i]) {
        holdout_object_points.push_back(all_object_points[i]);
        holdout_image_points.push_back(all_image_points[i]);
        holdout_indices.push_back(all_indices[i]);
      } else {
        object_points.push_back(all_object_points[i]);
        image_points.push_back(all_image_points[i]);
        indices.push_back(all_indices[i]);
      }
    }

    const int flags = opt_.free_k3 ? 0 : cv::CALIB_FIX_K3;
    const auto criteria =
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    auto calibrate = [&]() -> double {
      return cv::calibrateCamera(
        object_points, image_points, image_size, camera_matrix, dist_coeffs, rvecs, tvecs, flags,
        criteria);
    };

    double rms = calibrate();

    // 离群帧剔除：超过 max(reject_px, mean + sigma*std) 的帧逐轮剔除并重标，
    // 始终保证剩余帧 >= min_samples。
    int dropped = 0;
    if (!opt_.no_reject) {
      for (int iter = 0; iter < opt_.max_reject_iters; ++iter) {
        const auto errors = calibration::compute_per_view_errors(
          object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs);
        const double mean = vector_mean(errors);
        const double stddev = vector_std(errors, mean);
        const double threshold = std::max(opt_.reject_px, mean + opt_.reject_sigma * stddev);

        std::vector<int> outliers;
        for (int i = 0; i < static_cast<int>(errors.size()); ++i) {
          if (errors[i] > threshold) {
            outliers.push_back(i);
          }
        }
        if (outliers.empty()) {
          break;
        }

        const int droppable = static_cast<int>(object_points.size()) - opt_.min_samples;
        if (droppable <= 0) {
          utils::logger()->warn(
            "[CameraCalib] 检测到 {} 个离群帧，但剔除后将不足 {} 张，跳过剔除",
            outliers.size(), opt_.min_samples);
          break;
        }

        std::sort(
          outliers.begin(), outliers.end(),
          [&](int a, int b) { return errors[a] > errors[b]; });
        const int to_drop = std::min(static_cast<int>(outliers.size()), droppable);
        std::vector<int> remove_positions(outliers.begin(), outliers.begin() + to_drop);
        std::sort(remove_positions.begin(), remove_positions.end(), std::greater<int>());
        for (const int pos : remove_positions) {
          utils::logger()->info(
            "[CameraCalib] 剔除离群帧 sample {:03d} (err={:.4f} px > thr={:.4f} px)",
            indices[pos], errors[pos], threshold);
          object_points.erase(object_points.begin() + pos);
          image_points.erase(image_points.begin() + pos);
          indices.erase(indices.begin() + pos);
          ++dropped;
        }

        rms = calibrate();
      }
    }

    const auto final_errors = calibration::compute_per_view_errors(
      object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs);

    calibration::CalibrationReport report;
    report.rms_px = rms;
    report.mean_px = vector_mean(final_errors);
    report.max_view_px =
      final_errors.empty() ? 0.0 : *std::max_element(final_errors.begin(), final_errors.end());
    report.std_view_px = vector_std(final_errors, report.mean_px);
    report.used_samples = static_cast<int>(object_points.size());
    report.dropped_samples = dropped;

    if (holdout > 0) {
      std::vector<double> holdout_errors;
      for (size_t i = 0; i < holdout_object_points.size(); ++i) {
        cv::Mat rvec;
        cv::Mat tvec;
        const bool ok = cv::solvePnP(
          holdout_object_points[i], holdout_image_points[i], camera_matrix, dist_coeffs, rvec, tvec,
          false, cv::SOLVEPNP_IPPE);
        if (!ok) {
          utils::logger()->warn(
            "[CameraCalib] 留出帧 sample {:03d} solvePnP 失败，跳过", holdout_indices[i]);
          continue;
        }
        holdout_errors.push_back(
          calibration::compute_reprojection_error(
            holdout_object_points[i], holdout_image_points[i], rvec, tvec, camera_matrix,
            dist_coeffs));
      }
      if (!holdout_errors.empty()) {
        report.holdout_mean_px = vector_mean(holdout_errors);
        report.holdout_samples = static_cast<int>(holdout_errors.size());
      }
    }

    utils::logger()->info(
      "[CameraCalib] 标定完成: RMS={:.4f} mean={:.4f} max={:.4f} std={:.4f} px | 使用 {} 帧（剔除 {}）",
      report.rms_px, report.mean_px, report.max_view_px, report.std_view_px, report.used_samples,
      report.dropped_samples);
    if (report.holdout_samples > 0) {
      utils::logger()->info(
        "[CameraCalib] 留出验证: mean={:.4f} px（{} 帧）", report.holdout_mean_px,
        report.holdout_samples);
    }

    if (report.rms_px > opt_.max_rms) {
      utils::logger()->warn(
        "[CameraCalib] RMS {:.4f} px 超过阈值 {:.4f} px，建议检查模糊/曝光/圆点完整度/样本覆盖/是否混入错误分辨率图片",
        report.rms_px, opt_.max_rms);
    }

    std::cout << calibration::make_camera_yaml(camera_matrix, dist_coeffs, report, image_size)
              << std::endl;
    return 0;
  }

private:
  calibration::PatternConfig pattern_config_;
  Options opt_;
  const std::string window_name_{"camera_calib_node"};
};

}  // namespace Application

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@input-folder   | assets/img_with_q | 输入数据文件夹}"
    "{show s          | false | 是否显示圆点检测结果}"
    "{min-samples     | 15    | 最少有效样本数（建议 15-30）}"
    "{max-rms         | 1.0   | RMS 重投影误差告警阈值(px)，超过仅告警不失败}"
    "{free-k3         | false | 放开 k3（默认锁 k3=0，CALIB_FIX_K3）}"
    "{reject-sigma    | 3.0   | 离群帧统计阈值：均值 + sigma*std}"
    "{reject-px       | 2.0   | 离群帧绝对阈值(px)，与统计阈值取较大者}"
    "{max-reject-iters| 3     | 离群剔除最大迭代轮数}"
    "{no-reject       | false | 关闭离群点剔除}"
    "{holdout         | 0     | 留出验证帧数（0=关闭），等间隔抽取}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  Application::Options options;
  options.input_folder = cli.get<std::string>(0);
  options.show_detection = cli.get<bool>("show");
  options.min_samples = cli.get<int>("min-samples");
  options.max_rms = cli.get<double>("max-rms");
  options.free_k3 = cli.get<bool>("free-k3");
  options.reject_sigma = cli.get<double>("reject-sigma");
  options.reject_px = cli.get<double>("reject-px");
  options.max_reject_iters = cli.get<int>("max-reject-iters");
  options.no_reject = cli.get<bool>("no-reject");
  options.holdout = cli.get<int>("holdout");

  try {
    Application::CameraCalibApp app(std::move(options));
    return app.run();
  } catch (const std::exception & e) {
    utils::logger()->error("[CameraCalib] 程序异常终止: {}", e.what());
    return 1;
  }
}
