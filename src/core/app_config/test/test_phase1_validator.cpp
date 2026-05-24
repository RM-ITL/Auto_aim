// ===================================================================================
// test_phase1_validator.cpp
//
// Sprint 1.C.4 回归：声明式 post-load validator 覆盖 vector 长度、数值范围和
// 非空字符串三类语义错误。测试只调用 validate_collect_errors，避免触发 validate 的
// exit(1) 路径。
// ===================================================================================
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "app_config/app_config.hpp"
#include "app_config/validator.hpp"
#include "yaml.hpp"

namespace
{

const std::vector<std::string> kConfigFiles{
  "standard3.yaml",
  "standard4.yaml",
  "sentry.yaml",
  "hero.yaml",
  "uav.yaml",
  "dart.yaml",
  "config.yaml",
};

struct LoadedConfig
{
  std::string file;
  app_config::AppConfig cfg;
  YAML::Node root;
};

std::filesystem::path find_repo_root()
{
  auto dir = std::filesystem::current_path();
  while (true) {
    if (std::filesystem::exists(dir / "src/config/standard3.yaml") &&
        std::filesystem::exists(dir / "src/core/app_config")) {
      return dir;
    }
    if (!dir.has_parent_path() || dir.parent_path() == dir) {
      return {};
    }
    dir = dir.parent_path();
  }
}

int fail(const std::string & msg)
{
  std::cerr << "[FAIL] " << msg << '\n';
  return 1;
}

std::string join_errors(const std::vector<std::string> & errors)
{
  std::ostringstream oss;
  for (const auto & e : errors) {
    oss << "\n  - " << e;
  }
  return oss.str();
}

bool contains_error(const std::vector<std::string> & errors, const std::string & needle)
{
  for (const auto & e : errors) {
    if (e.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

int assert_empty(const std::vector<std::string> & errors, const std::string & name)
{
  if (!errors.empty()) {
    return fail(name + ": expected no validator errors, got" + join_errors(errors));
  }
  return 0;
}

int assert_contains(
  const std::vector<std::string> & errors,
  const std::string & field,
  const std::string & name)
{
  if (!contains_error(errors, field)) {
    return fail(name + ": expected error containing '" + field + "', got" + join_errors(errors));
  }
  return 0;
}

int assert_not_contains(
  const std::vector<std::string> & errors,
  const std::string & field,
  const std::string & name)
{
  if (contains_error(errors, field)) {
    return fail(name + ": unexpected error containing '" + field + "', got" + join_errors(errors));
  }
  return 0;
}

LoadedConfig load_config(const std::filesystem::path & repo_root, const std::string & file)
{
  const auto yaml_path = repo_root / "src/config" / file;
  LoadedConfig loaded;
  loaded.file = file;
  loaded.cfg = app_config::AppConfig::load(yaml_path.string());
  loaded.root = utils::load(yaml_path.string());
  return loaded;
}

app_config::AppConfig minimal_valid_cfg()
{
  app_config::AppConfig cfg;
  cfg.solver.camera_intri.focal_length = {1000.0, 1000.0};
  cfg.solver.camera_intri.principal_point = {640.0, 512.0};
  cfg.solver.camera_intri.disto_param = {0.0, 0.0, 0.0, 0.0, 0.0};
  return cfg;
}

template <typename Mutator>
int expect_rejects(
  const LoadedConfig & loaded,
  const std::string & field,
  Mutator mutate)
{
  auto cfg = loaded.cfg;
  mutate(cfg);
  const auto errors = app_config::validate_collect_errors(cfg, loaded.root);
  return assert_contains(errors, field, loaded.file + " invalid " + field);
}

template <typename Mutator>
int expect_synthetic_rejects(
  const app_config::AppConfig & base_cfg,
  const YAML::Node & root,
  const std::string & field,
  Mutator mutate)
{
  auto cfg = base_cfg;
  mutate(cfg);
  const auto errors = app_config::validate_collect_errors(cfg, root);
  return assert_contains(errors, field, "synthetic invalid " + field);
}

int assert_all_configs_pass(const std::filesystem::path & repo_root)
{
  for (const auto & file : kConfigFiles) {
    const auto loaded = load_config(repo_root, file);
    const auto errors = app_config::validate_collect_errors(loaded.cfg, loaded.root);
    if (const int ret = assert_empty(errors, file)) {
      return ret;
    }
  }
  return 0;
}

int assert_common_rule_rejections(const LoadedConfig & standard3)
{
  if (const int ret = assert_empty(
        app_config::validate_collect_errors(standard3.cfg, standard3.root),
        "standard3 positive baseline")) {
    return ret;
  }

  if (const int ret = expect_rejects(standard3, "solver.camera_intri.focal_length", [](auto & cfg) {
        cfg.solver.camera_intri.focal_length = {1.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "solver.camera_intri.principal_point", [](auto & cfg) {
        cfg.solver.camera_intri.principal_point = {1.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "solver.camera_intri.disto_param", [](auto & cfg) {
        cfg.solver.camera_intri.disto_param = {1.0, 2.0, 3.0};
      })) {
    return ret;
  }
  {
    auto cfg = standard3.cfg;
    cfg.solver.camera_intri.disto_param.clear();
    if (const int ret = assert_empty(
          app_config::validate_collect_errors(cfg, standard3.root),
          "solver.camera_intri.disto_param empty positive")) {
      return ret;
    }
  }

  if (const int ret = expect_rejects(
        standard3, "solver.coord_converter.rotation_matrix_camera_to_gimbal", [](auto & cfg) {
          cfg.solver.coord_converter.rotation_matrix_camera_to_gimbal = {1.0};
        })) {
    return ret;
  }
  if (const int ret = expect_rejects(
        standard3, "solver.coord_converter.rotation_matrix_gimbal_to_imu", [](auto & cfg) {
          cfg.solver.coord_converter.rotation_matrix_gimbal_to_imu = {1.0};
        })) {
    return ret;
  }
  if (const int ret = expect_rejects(
        standard3, "solver.coord_converter.t_camera_to_gimbal", [](auto & cfg) {
          cfg.solver.coord_converter.t_camera_to_gimbal = {1.0, 2.0};
        })) {
    return ret;
  }
  {
    auto cfg = standard3.cfg;
    cfg.solver.coord_converter.rotation_matrix_camera_to_gimbal.clear();
    cfg.solver.coord_converter.rotation_matrix_gimbal_to_imu.clear();
    cfg.solver.coord_converter.t_camera_to_gimbal.clear();
    if (const int ret = assert_empty(
          app_config::validate_collect_errors(cfg, standard3.root),
          "solver.coord_converter empty positives")) {
      return ret;
    }
  }

  if (const int ret = expect_rejects(standard3, "planner.Q_yaw", [](auto & cfg) {
        cfg.planner.Q_yaw = {1.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "planner.R_yaw", [](auto & cfg) {
        cfg.planner.R_yaw = {1.0, 2.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "planner.Q_pitch", [](auto & cfg) {
        cfg.planner.Q_pitch = {1.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "planner.R_pitch", [](auto & cfg) {
        cfg.planner.R_pitch = {1.0, 2.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "planner.max_yaw_acc", [](auto & cfg) {
        cfg.planner.max_yaw_acc = 0.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "planner.max_pitch_acc", [](auto & cfg) {
        cfg.planner.max_pitch_acc = -1.0;
      })) {
    return ret;
  }

  if (const int ret = expect_rejects(standard3, "aimer.comming_angle", [](auto & cfg) {
        cfg.aimer.comming_angle = 0.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "aimer.leaving_angle", [](auto & cfg) {
        cfg.aimer.leaving_angle = -1.0;
      })) {
    return ret;
  }

  if (const int ret = expect_rejects(standard3, "shooter.first_tolerance", [](auto & cfg) {
        cfg.shooter.first_tolerance = 0.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "shooter.second_tolerance", [](auto & cfg) {
        cfg.shooter.second_tolerance = -1.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "shooter.judge_distance", [](auto & cfg) {
        cfg.shooter.judge_distance = 0.0;
      })) {
    return ret;
  }

  if (const int ret = expect_rejects(standard3, "tracker.min_detect_count", [](auto & cfg) {
        cfg.tracker.min_detect_count = 0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "tracker.max_temp_lost_count", [](auto & cfg) {
        cfg.tracker.max_temp_lost_count = 0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "tracker.outpost_max_temp_lost_count", [](auto & cfg) {
        cfg.tracker.outpost_max_temp_lost_count = 0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "tracker.outpost_min_detect_count", [](auto & cfg) {
        cfg.tracker.outpost_min_detect_count = 0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "tracker.outpost_detect_fail_tolerance", [](auto & cfg) {
        cfg.tracker.outpost_detect_fail_tolerance = 0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "tracker.single_plate_threshold", [](auto & cfg) {
        cfg.tracker.single_plate_threshold = 0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "tracker.omega_threshold", [](auto & cfg) {
        cfg.tracker.omega_threshold = 0.0;
      })) {
    return ret;
  }

  if (const int ret = expect_rejects(standard3, "classifier.min_confidence", [](auto & cfg) {
        cfg.detector.classifier.min_confidence = 2.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "detector.yolov5.min_confidence", [](auto & cfg) {
        cfg.detector.yolov5.min_confidence = -0.1;
      })) {
    return ret;
  }

  if (const int ret = expect_rejects(standard3, "dm_imu.imu_com_port", [](auto & cfg) {
        cfg.dm_imu.imu_com_port.clear();
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "dm_imu.baud", [](auto & cfg) {
        cfg.dm_imu.baud = 0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "dm_imu.publish_rate", [](auto & cfg) {
        cfg.dm_imu.publish_rate = -1;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "gimbal.com_port", [](auto & cfg) {
        cfg.gimbal.com_port.clear();
      })) {
    return ret;
  }

  if (const int ret = expect_rejects(standard3, "base_hit.score_threshold", [](auto & cfg) {
        cfg.base_hit.openvino.score_threshold = -0.1f;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(standard3, "base_hit.nms_threshold", [](auto & cfg) {
        cfg.base_hit.openvino.nms_threshold = 1.1f;
      })) {
    return ret;
  }

  return 0;
}

int assert_aim_planner_rules(const LoadedConfig & hero, const LoadedConfig & standard3)
{
  if (const int ret = assert_empty(
        app_config::validate_collect_errors(hero.cfg, hero.root),
        "hero aim_planner positive baseline")) {
    return ret;
  }

  if (const int ret = expect_rejects(hero, "aim_planner.Q_yaw", [](auto & cfg) {
        cfg.aim_planner.Q_yaw = {1.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(hero, "aim_planner.R_yaw", [](auto & cfg) {
        cfg.aim_planner.R_yaw = {1.0, 2.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(hero, "aim_planner.Q_pitch", [](auto & cfg) {
        cfg.aim_planner.Q_pitch = {1.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(hero, "aim_planner.R_pitch", [](auto & cfg) {
        cfg.aim_planner.R_pitch = {1.0, 2.0};
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(hero, "aim_planner.max_yaw_acc", [](auto & cfg) {
        cfg.aim_planner.max_yaw_acc = 0.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(hero, "aim_planner.max_pitch_acc", [](auto & cfg) {
        cfg.aim_planner.max_pitch_acc = -1.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(hero, "aim_planner.window_angle", [](auto & cfg) {
        cfg.aim_planner.window_angle = -1.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(hero, "aim_planner.omega_threshold", [](auto & cfg) {
        cfg.aim_planner.omega_threshold = 0.0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(hero, "aim_planner.armor_hysteresis", [](auto & cfg) {
        cfg.aim_planner.armor_hysteresis = -1.0;
      })) {
    return ret;
  }

  if (standard3.cfg.aim_planner.window_angle != 0.0) {
    return fail("standard3 aim_planner.window_angle expected default 0");
  }
  const auto standard3_errors = app_config::validate_collect_errors(standard3.cfg, standard3.root);
  if (const int ret = assert_not_contains(
        standard3_errors, "aim_planner.",
        "standard3 must not validate absent AimPlanner fields")) {
    return ret;
  }
  return 0;
}

int assert_dart_and_light_rules(const LoadedConfig & dart, const LoadedConfig & standard3)
{
  const auto dart_errors = app_config::validate_collect_errors(dart.cfg, dart.root);
  if (const int ret = assert_empty(dart_errors, "dart positive baseline")) {
    return ret;
  }
  for (const auto & absent_field : {
      std::string{"planner."},
      std::string{"aim_planner."},
      std::string{"tracker."},
      std::string{"aimer."},
      std::string{"shooter."},
      std::string{"dm_imu."},
      std::string{"detector.yolo"},
      std::string{"classifier."},
      std::string{"gimbal."},
    }) {
    if (const int ret = assert_not_contains(
          dart_errors, absent_field,
          "dart-like missing sections must not be reported")) {
      return ret;
    }
  }

  const auto standard3_errors = app_config::validate_collect_errors(standard3.cfg, standard3.root);
  if (const int ret = assert_not_contains(
        standard3_errors, "dart.com_port",
        "standard3 must not validate absent Dart fields")) {
    return ret;
  }

  if (const int ret = expect_rejects(dart, "dart.com_port", [](auto & cfg) {
        cfg.dart.com_port.clear();
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(dart, "light_tracker.min_detect_count", [](auto & cfg) {
        cfg.light_tracker.min_detect_count = 0;
      })) {
    return ret;
  }
  if (const int ret = expect_rejects(dart, "light_tracker.max_temp_lost_count", [](auto & cfg) {
        cfg.light_tracker.max_temp_lost_count = -1;
      })) {
    return ret;
  }
  return 0;
}

int assert_synthetic_yolo_rules()
{
  {
    auto cfg = minimal_valid_cfg();
    cfg.detector.yolo_name = "yolov5";
    cfg.detector.yolov5.min_confidence = 0.5;
    const auto root = YAML::Load(R"yaml(
yolo:
  yolo_name: "yolov5"
)yaml");
    if (const int ret = assert_empty(
          app_config::validate_collect_errors(cfg, root),
          "synthetic yolov5 positive baseline")) {
      return ret;
    }
    if (const int ret = expect_synthetic_rejects(cfg, root, "detector.yolov5.min_confidence", [](auto & bad) {
          bad.detector.yolov5.min_confidence = 2.0;
        })) {
      return ret;
    }
  }

  {
    auto cfg = minimal_valid_cfg();
    cfg.detector.yolo_name = "yolo11";
    cfg.detector.yolo11.min_confidence = 0.5;
    const auto root = YAML::Load(R"yaml(
yolo:
  yolo_name: "yolo11"
)yaml");
    if (const int ret = assert_empty(
          app_config::validate_collect_errors(cfg, root),
          "synthetic yolo11 positive baseline")) {
      return ret;
    }
    if (const int ret = expect_synthetic_rejects(cfg, root, "detector.yolo11.min_confidence", [](auto & bad) {
          bad.detector.yolo11.min_confidence = 2.0;
        })) {
      return ret;
    }
  }

  {
    auto cfg = minimal_valid_cfg();
    cfg.detector.yolo_name = "yolov5";
    cfg.detector.yolov5.min_confidence = 0.5;
    cfg.detector.classifier.min_confidence = 0.5;
    const auto root = YAML::Load(R"yaml(
yolo:
  yolo_name: "yolov5"
Classifier: {}
)yaml");
    if (const int ret = assert_empty(
          app_config::validate_collect_errors(cfg, root),
          "synthetic classifier positive baseline")) {
      return ret;
    }
    if (const int ret = expect_synthetic_rejects(cfg, root, "classifier.min_confidence", [](auto & bad) {
          bad.detector.classifier.min_confidence = -0.1;
        })) {
      return ret;
    }
  }

  return 0;
}

}  // namespace

int main()
{
  const auto repo_root = find_repo_root();
  if (repo_root.empty()) {
    return fail("cannot locate repo root");
  }

  const auto standard3 = load_config(repo_root, "standard3.yaml");
  const auto hero = load_config(repo_root, "hero.yaml");
  const auto dart = load_config(repo_root, "dart.yaml");

  if (const int ret = assert_all_configs_pass(repo_root)) {
    return ret;
  }
  if (const int ret = assert_common_rule_rejections(standard3)) {
    return ret;
  }
  if (const int ret = assert_aim_planner_rules(hero, standard3)) {
    return ret;
  }
  if (const int ret = assert_dart_and_light_rules(dart, standard3)) {
    return ret;
  }
  if (const int ret = assert_synthetic_yolo_rules()) {
    return ret;
  }

  std::cout << "[PASS] validator rules and section presence checks passed\n";
  return 0;
}
