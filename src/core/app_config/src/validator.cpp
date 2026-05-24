// ===================================================================================
// validator.cpp
//
// Sprint 1.C.4：AppConfig post-load 语义校验。
//
// 注意：本文件的 section presence 守卫必须与 app_config.cpp 中对应 load_xxx 的
// IsDefined 守卫保持同步。validator 只做 post-load 的 range / length / non-empty
// 校验，不做 enum、类型错位、路径存在、跨字段依赖或矩阵正交性校验。
// ===================================================================================
#include "app_config/validator.hpp"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "logger.hpp"

namespace app_config
{
namespace
{

class ConfigValidator
{
public:
  void check_size(const std::string & field, std::size_t actual, std::size_t expected)
  {
    if (actual != expected) {
      errors_.push_back(fmt::format("{} size={}, expected {}", field, actual, expected));
    }
  }

  void check_size_or_empty(const std::string & field, std::size_t actual, std::size_t expected)
  {
    if (actual != 0 && actual != expected) {
      errors_.push_back(fmt::format("{} size={}, expected {} or empty", field, actual, expected));
    }
  }

  template <typename T>
  void check_positive(const std::string & field, const T & value)
  {
    if (!(value > 0)) {
      errors_.push_back(fmt::format("{} = {} not positive", field, value));
    }
  }

  void check_range(const std::string & field, double value, double lo, double hi)
  {
    if (!(value >= lo && value <= hi)) {
      errors_.push_back(fmt::format("{} = {} not in [{}, {}]", field, value, lo, hi));
    }
  }

  void check_non_empty(const std::string & field, const std::string & value)
  {
    if (value.empty()) {
      errors_.push_back(fmt::format("{} is empty string", field));
    }
  }

  std::vector<std::string> take()
  {
    return std::move(errors_);
  }

private:
  std::vector<std::string> errors_;
};

bool aim_planner_loaded(const YAML::Node & root)
{
  if (!root["Planner"]) {
    return false;
  }
  const auto pl = root["Planner"];
  return pl["armor_hysteresis"] && pl["omega_threshold"] && pl["window_angle"];
}

void validate_camera_intri(ConfigValidator & v, const CameraIntriConfig & c)
{
  v.check_size("solver.camera_intri.focal_length", c.focal_length.size(), 2);
  v.check_size("solver.camera_intri.principal_point", c.principal_point.size(), 2);
  v.check_size_or_empty("solver.camera_intri.disto_param", c.disto_param.size(), 5);
}

void validate_coord_converter(ConfigValidator & v, const CoordConverterConfig & c)
{
  v.check_size_or_empty(
    "solver.coord_converter.rotation_matrix_camera_to_gimbal",
    c.rotation_matrix_camera_to_gimbal.size(), 9);
  v.check_size_or_empty(
    "solver.coord_converter.rotation_matrix_gimbal_to_imu",
    c.rotation_matrix_gimbal_to_imu.size(), 9);
  v.check_size_or_empty(
    "solver.coord_converter.t_camera_to_gimbal",
    c.t_camera_to_gimbal.size(), 3);
}

template <typename PlannerLike>
void validate_planner_common(ConfigValidator & v, const std::string & prefix, const PlannerLike & p)
{
  v.check_size(prefix + ".Q_yaw", p.Q_yaw.size(), 2);
  v.check_size(prefix + ".R_yaw", p.R_yaw.size(), 1);
  v.check_size(prefix + ".Q_pitch", p.Q_pitch.size(), 2);
  v.check_size(prefix + ".R_pitch", p.R_pitch.size(), 1);
  v.check_positive(prefix + ".max_yaw_acc", p.max_yaw_acc);
  v.check_positive(prefix + ".max_pitch_acc", p.max_pitch_acc);
}

void validate_planner(ConfigValidator & v, const PlannerConfig & p)
{
  validate_planner_common(v, "planner", p);
}

void validate_aim_planner(ConfigValidator & v, const AimPlannerConfig & p)
{
  validate_planner_common(v, "aim_planner", p);
  v.check_positive("aim_planner.window_angle", p.window_angle);
  v.check_positive("aim_planner.omega_threshold", p.omega_threshold);
  v.check_positive("aim_planner.armor_hysteresis", p.armor_hysteresis);
}

void validate_aimer(ConfigValidator & v, const AimerConfig & a)
{
  v.check_positive("aimer.comming_angle", a.comming_angle);
  v.check_positive("aimer.leaving_angle", a.leaving_angle);
}

void validate_shooter(ConfigValidator & v, const ShooterConfig & s)
{
  v.check_positive("shooter.first_tolerance", s.first_tolerance);
  v.check_positive("shooter.second_tolerance", s.second_tolerance);
  v.check_positive("shooter.judge_distance", s.judge_distance);
}

void validate_tracker(ConfigValidator & v, const TrackerConfig & t)
{
  v.check_positive("tracker.min_detect_count", t.min_detect_count);
  v.check_positive("tracker.max_temp_lost_count", t.max_temp_lost_count);
  v.check_positive("tracker.outpost_max_temp_lost_count", t.outpost_max_temp_lost_count);
  v.check_positive("tracker.outpost_min_detect_count", t.outpost_min_detect_count);
  v.check_positive("tracker.outpost_detect_fail_tolerance", t.outpost_detect_fail_tolerance);
  v.check_positive("tracker.single_plate_threshold", t.single_plate_threshold);
  v.check_positive("tracker.omega_threshold", t.omega_threshold);
}

void validate_classifier(ConfigValidator & v, const ClassifierConfig & c)
{
  v.check_range("classifier.min_confidence", c.min_confidence, 0.0, 1.0);
}

void validate_detector_yolov5(ConfigValidator & v, const DetectorYOLOv5Config & d)
{
  v.check_range("detector.yolov5.min_confidence", d.min_confidence, 0.0, 1.0);
}

void validate_detector_yolo11(ConfigValidator & v, const DetectorYOLO11Config & d)
{
  v.check_range("detector.yolo11.min_confidence", d.min_confidence, 0.0, 1.0);
}

void validate_dm_imu(ConfigValidator & v, const DmImuConfig & d)
{
  v.check_non_empty("dm_imu.imu_com_port", d.imu_com_port);
  v.check_positive("dm_imu.baud", d.baud);
  v.check_positive("dm_imu.publish_rate", d.publish_rate);
}

void validate_gimbal(ConfigValidator & v, const GimbalConfig & g)
{
  v.check_non_empty("gimbal.com_port", g.com_port);
}

void validate_dart(ConfigValidator & v, const DartConfig & d)
{
  v.check_non_empty("dart.com_port", d.com_port);
}

void validate_light_tracker(ConfigValidator & v, const LightTrackerConfig & t)
{
  v.check_positive("light_tracker.min_detect_count", t.min_detect_count);
  v.check_positive("light_tracker.max_temp_lost_count", t.max_temp_lost_count);
}

void validate_base_hit(ConfigValidator & v, const BaseHitConfig & b)
{
  v.check_range("base_hit.score_threshold", b.openvino.score_threshold, 0.0, 1.0);
  v.check_range("base_hit.nms_threshold", b.openvino.nms_threshold, 0.0, 1.0);
}

}  // namespace

std::vector<std::string> validate_collect_errors(const AppConfig & cfg, const YAML::Node & root)
{
  ConfigValidator v;

  validate_camera_intri(v, cfg.solver.camera_intri);

  if (root["Solver"] && root["Solver"]["coord_converter"]) {
    validate_coord_converter(v, cfg.solver.coord_converter);
  }
  if (root["Planner"]) {
    validate_planner(v, cfg.planner);
  }
  if (aim_planner_loaded(root)) {
    validate_aim_planner(v, cfg.aim_planner);
  }
  if (root["Aimer"]) {
    validate_aimer(v, cfg.aimer);
  }
  if (root["Shooter"]) {
    validate_shooter(v, cfg.shooter);
  }
  if (root["Tracker"]) {
    validate_tracker(v, cfg.tracker);
  }
  if (root["yolo"] && cfg.detector.yolo_name == "yolov5" && root["Classifier"]) {
    validate_classifier(v, cfg.detector.classifier);
  }
  if (root["yolo"]) {
    if (cfg.detector.yolo_name == "yolov5") {
      validate_detector_yolov5(v, cfg.detector.yolov5);
    } else if (cfg.detector.yolo_name == "yolo11") {
      validate_detector_yolo11(v, cfg.detector.yolo11);
    }
  }
  if (root["DM_IMU"]) {
    validate_dm_imu(v, cfg.dm_imu);
  }
  if (root["Gimbal"]) {
    validate_gimbal(v, cfg.gimbal);
  }
  if (root["lower_Dart"]) {
    validate_dart(v, cfg.dart);
  }
  if (root["LightTracker"]) {
    validate_light_tracker(v, cfg.light_tracker);
  }
  if (root["Base_Hit"]) {
    validate_base_hit(v, cfg.base_hit);
  }

  return v.take();
}

void validate(const AppConfig & cfg, const YAML::Node & root)
{
  const auto errors = validate_collect_errors(cfg, root);
  if (errors.empty()) {
    return;
  }

  for (const auto & e : errors) {
    utils::logger()->error("[Validator] {}", e);
  }
  utils::logger()->error("[Validator] {} rule(s) failed; aborting", errors.size());
  std::exit(1);
}

}  // namespace app_config
