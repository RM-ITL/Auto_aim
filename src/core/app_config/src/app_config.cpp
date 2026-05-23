// ===================================================================================
// app_config.cpp
//
// Sprint 1.C 引入：AppConfig::load 实现。
//
// 内部一次 YAML::LoadFile（走 utils::load，失败 exit(1)）拿到 root Node，
// 按 SubConfig 逐段填充。每段 from_yaml-style 函数贴着原模块 yaml 读取语义：
//
//   - required 字段：utils::read<T>(node, key) 无默认重载，缺崩 exit(1)
//   - 可选字段：utils::read<T>(node, key, default) 三参数重载
//   - 整组可缺嵌套段：IsDefined 守卫
//   - 条件读取（CalibParam / Solver.coord_converter 下 rotation_matrix_* / t_camera_to_gimbal 等）：
//     与原模块相同的 if(node[key]) 守卫
//
// 不在 AppConfig::load 内部打模块级启动日志（[Gimbal]/[Planner]/... 这些日志
// 是模块构造时的事，Phase 1~5 改造后由各模块自行打印）。本文件只打
// AppConfig 级别的加载诊断（来源文件路径 / 大小 / mtime）。
// ===================================================================================
#include "app_config/app_config.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "logger.hpp"
#include "yaml.hpp"

namespace app_config
{
namespace
{

// ---------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------

// 与 PnPSolver / CoordConverter / YawOptimizer 内部一致：
// CalibParam.INTRI.Camera[0].value.{ptr_wrapper.data 或 直接 value 节点}。
// 返回最终承载相机参数（focal_length / principal_point / disto_param）的 Node；
// 若上层 Camera 节点缺失，返回 invalid Node。
YAML::Node camera_intri_data_node(const YAML::Node & root)
{
  if (!root["CalibParam"] || !root["CalibParam"]["INTRI"] ||
      !root["CalibParam"]["INTRI"]["Camera"]) {
    return YAML::Node{};
  }
  const auto camera_node = root["CalibParam"]["INTRI"]["Camera"][0]["value"];
  if (camera_node["ptr_wrapper"] && camera_node["ptr_wrapper"]["data"]) {
    return camera_node["ptr_wrapper"]["data"];
  }
  return camera_node;
}

// ---------------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------------

void load_camera(CameraConfig & out, const YAML::Node & root)
{
  if (!root["camera"]) {
    // 与 camera::Camera 现行行为一致：缺整段也走默认 type=hik。
    return;
  }
  const auto camera_yaml = root["camera"];

  // type 条件读 + 默认 hik（同 camera.cpp:45 的 .as<string>("hik")）。
  out.type = utils::read<std::string>(camera_yaml, "type", out.type);

  const auto params = camera_yaml["parameters"];
  if (!params) return;

  // 按 type 条件填充：与原代码"按 type 选择性 new"等价。
  // type=hik：HikCamera::load_config 全 required（裸 .as<T>()）。
  // type=mindvision：Camera 外壳读时全部 .as<T>(default)。
  if (out.type == "hik") {
    out.hik.exposure_ms = params["exposure_ms"].as<double>();
    out.hik.gain = params["gain"].as<double>();
    out.hik.fps = params["fps"].as<double>();
    out.hik.target_width = params["target_width"].as<int>();
    out.hik.target_height = params["target_height"].as<int>();
    out.hik.image_topic = params["image_topic"].as<std::string>();
  } else if (out.type == "mindvision") {
    out.mindvision.exposure_ms =
      params["exposure_ms"] ? params["exposure_ms"].as<double>() : out.mindvision.exposure_ms;
    out.mindvision.gamma =
      params["gamma"] ? params["gamma"].as<double>() : out.mindvision.gamma;
    out.mindvision.vid_pid =
      params["vid_pid"] ? params["vid_pid"].as<std::string>() : out.mindvision.vid_pid;
  }
  // 其他 type 值：原代码会 cerr "Unsupported camera type" 但不退出；这里我们也不退出。
}

// ---------------------------------------------------------------------------------
// Detector / Classifier / Traditional
// ---------------------------------------------------------------------------------

void load_detector_yolov5(DetectorYOLOv5Config & out, const YAML::Node & yolo_yaml)
{
  out.yolov5_model_path = yolo_yaml["yolov5_model_path"].as<std::string>();
  out.device = yolo_yaml["device"].as<std::string>();
  out.min_confidence = yolo_yaml["min_confidence"].as<double>();
  out.score_threshold = yolo_yaml["score_threshold"].as<float>(out.score_threshold);
  out.nms_threshold = yolo_yaml["nms_threshold"].as<float>(out.nms_threshold);
  out.use_traditional = yolo_yaml["use_traditional"].as<bool>(out.use_traditional);
}

void load_detector_yolo11(DetectorYOLO11Config & out, const YAML::Node & yolo_yaml)
{
  out.yolo11_model_path = yolo_yaml["yolo11_model_path"].as<std::string>();
  out.device = yolo_yaml["device"].as<std::string>(out.device);
  out.min_confidence = yolo_yaml["min_confidence"].as<double>(out.min_confidence);
  out.score_threshold = yolo_yaml["score_threshold"].as<float>(out.score_threshold);
  out.nms_threshold = yolo_yaml["nms_threshold"].as<float>(out.nms_threshold);
  if (yolo_yaml["enemy_color"]) {
    out.enemy_color = yolo_yaml["enemy_color"].as<std::string>();
  }
}

void load_detector_traditional(DetectorTraditionalConfig & out, const YAML::Node & traditional_yaml)
{
  // Traditional_Detector 9 个几何字段 required；缺段/缺字段保持裸读失败。
  out.threshold = traditional_yaml["threshold"].as<double>();
  out.max_angle_error = traditional_yaml["max_angle_error"].as<double>();
  out.min_lightbar_ratio = traditional_yaml["min_lightbar_ratio"].as<double>();
  out.max_lightbar_ratio = traditional_yaml["max_lightbar_ratio"].as<double>();
  out.min_lightbar_length = traditional_yaml["min_lightbar_length"].as<double>();
  out.min_armor_ratio = traditional_yaml["min_armor_ratio"].as<double>();
  out.max_armor_ratio = traditional_yaml["max_armor_ratio"].as<double>();
  out.max_side_ratio = traditional_yaml["max_side_ratio"].as<double>();
  out.max_rectangular_error = traditional_yaml["max_rectangular_error"].as<double>();
}

void load_classifier(ClassifierConfig & out, const YAML::Node & classifier_yaml)
{
  out.classify_model = classifier_yaml["classify_model"].as<std::string>();
  out.min_confidence = classifier_yaml["min_confidence"].as<double>();
}

void load_detector(DetectorConfig & out, const YAML::Node & root)
{
  // detect_node.cpp:14 条件读 + 默认 yolo11
  if (root["yolo"] && root["yolo"]["yolo_name"]) {
    out.yolo_name = root["yolo"]["yolo_name"].as<std::string>();
  }

  // 按 yolo_name 条件填充。规则：与原代码"按 yolo_name 选择性 new"等价。
  if (out.yolo_name == "yolov5" && root["yolo"]) {
    load_detector_yolov5(out.yolov5, root["yolo"]);
    // YOLOv5 内部 new Traditional_Detector + Classifier，所以这两段也必须填。
    load_detector_traditional(out.traditional, root["Traditional_Detector"]);
    load_classifier(out.classifier, root["Classifier"]);
  } else if (out.yolo_name == "yolo11" && root["yolo"]) {
    load_detector_yolo11(out.yolo11, root["yolo"]);
    // YOLO11 不嵌套 Traditional/Classifier，但 SubConfig 仍预留默认值。
  }
}

// ---------------------------------------------------------------------------------
// Solver / PnP / CoordConverter / YawOptimizer
// ---------------------------------------------------------------------------------

// 三处共用：从 CalibParam 段读 focal_length / principal_point。
// disto_param 由调用方决定 IsDefined 守卫语义。
void load_camera_intri_required(
  std::vector<double> & focal_length,
  std::vector<double> & principal_point,
  const YAML::Node & data_node)
{
  focal_length = data_node["focal_length"].as<std::vector<double>>();
  principal_point = data_node["principal_point"].as<std::vector<double>>();
}

void load_pnp(PnPSolverConfig & out, const YAML::Node & root)
{
  const auto data = camera_intri_data_node(root);
  if (!data) {
    utils::logger()->error("[AppConfig] CalibParam.INTRI.Camera missing (PnPSolver expects it)");
    return;  // PnPSolver 在缺 Camera 节点时返回 false，不抛；保持与之一致。
  }
  load_camera_intri_required(out.focal_length, out.principal_point, data);

  // 与 pnp_solver.cpp:80 一致：IsDefined 守卫。
  if (data["disto_param"]) {
    out.disto_param = data["disto_param"].as<std::vector<double>>();
  }
  // 不在此处填 5 个 0：消费侧（PnPSolver 改造后）按 .empty() 检测填零，与原行为对齐。
}

void load_coord_converter(CoordConverterConfig & out, const YAML::Node & root)
{
  const auto data = camera_intri_data_node(root);
  if (data) {
    load_camera_intri_required(out.focal_length, out.principal_point, data);
    // CoordConverter 无 IsDefined 守卫，裸读；缺则抛 yaml 异常 → 与原 try/catch 行为一致：
    // 但 AppConfig::load 走 utils::load，不在我们捕获范围内；这里改用 IsDefined 守卫
    // 转换为"缺则空 vector"，把抛行为留给消费侧（CoordConverter 改造时用 .empty() 处理）。
    // 这是 SubConfig 加载阶段对消费侧抛行为的等价表达。
    if (data["disto_param"]) {
      out.disto_param = data["disto_param"].as<std::vector<double>>();
    }
  }

  // rotation_matrix_camera_to_gimbal / _gimbal_to_imu / t_camera_to_gimbal：
  // 在 Solver.coord_converter 守卫下读。
  // rotation_matrix_* 守卫内若 .data 字段缺会抛——保持与 coord_converter.cpp:326 行为：守卫存在但内部裸读。
  // SubConfig 加载阶段统一为 IsDefined 守卫（保护性更高），消费侧若需"裸抛"语义自行判断。
  if (root["Solver"] && root["Solver"]["coord_converter"]) {
    const auto cc = root["Solver"]["coord_converter"];
    if (cc["rotation_matrix_camera_to_gimbal"] &&
        cc["rotation_matrix_camera_to_gimbal"]["data"]) {
      out.rotation_matrix_camera_to_gimbal =
        cc["rotation_matrix_camera_to_gimbal"]["data"].as<std::vector<double>>();
    }
    if (cc["rotation_matrix_gimbal_to_imu"] &&
        cc["rotation_matrix_gimbal_to_imu"]["data"]) {
      out.rotation_matrix_gimbal_to_imu =
        cc["rotation_matrix_gimbal_to_imu"]["data"].as<std::vector<double>>();
    }
    if (cc["t_camera_to_gimbal"]) {
      out.t_camera_to_gimbal = cc["t_camera_to_gimbal"].as<std::vector<double>>();
    }
  }
}

void load_yaw_optimizer(YawOptimizerConfig & out, const YAML::Node & root)
{
  const auto data = camera_intri_data_node(root);
  if (!data) return;
  load_camera_intri_required(out.focal_length, out.principal_point, data);
  if (data["disto_param"]) {
    out.disto_param = data["disto_param"].as<std::vector<double>>();
  }
}

void load_solver(SolverConfig & out, const YAML::Node & root)
{
  load_pnp(out.pnp, root);
  load_coord_converter(out.coord_converter, root);
  load_yaw_optimizer(out.yaw_optimizer, root);
}

// ---------------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------------

void load_tracker(TrackerConfig & out, const YAML::Node & root)
{
  const auto tr = root["Tracker"];
  if (!tr) return;
  out.enemy_color = tr["enemy_color"].as<std::string>();
  out.min_detect_count = tr["min_detect_count"].as<int>();
  out.max_temp_lost_count = tr["max_temp_lost_count"].as<int>();
  out.outpost_max_temp_lost_count = tr["outpost_max_temp_lost_count"].as<int>();
  out.outpost_min_detect_count = tr["outpost_min_detect_count"].as<int>();
  out.outpost_detect_fail_tolerance = tr["outpost_detect_fail_tolerance"].as<int>();
  out.single_plate_threshold = tr["single_plate_threshold"].as<int>(out.single_plate_threshold);
  out.omega_threshold = tr["omega_threshold"].as<double>(out.omega_threshold);
  out.single_plate_debug = tr["single_plate_debug"].as<bool>(out.single_plate_debug);
}

// ---------------------------------------------------------------------------------
// Planner / AimPlanner
// ---------------------------------------------------------------------------------

void load_planner(PlannerConfig & out, const YAML::Node & root)
{
  if (!root["Planner"]) return;
  const auto pl = root["Planner"];
  out.yaw_offset = utils::read<double>(pl, "yaw_offset");
  out.pitch_offset = utils::read<double>(pl, "pitch_offset");
  out.fire_thresh = utils::read<double>(pl, "fire_thresh");
  out.decision_speed = utils::read<double>(pl, "decision_speed");
  out.high_speed_delay_time = utils::read<double>(pl, "high_speed_delay_time");
  out.low_speed_delay_time = utils::read<double>(pl, "low_speed_delay_time");
  out.max_yaw_acc = utils::read<double>(pl, "max_yaw_acc");
  out.Q_yaw = utils::read<std::vector<double>>(pl, "Q_yaw");
  out.R_yaw = utils::read<std::vector<double>>(pl, "R_yaw");
  out.max_pitch_acc = utils::read<double>(pl, "max_pitch_acc");
  out.Q_pitch = utils::read<std::vector<double>>(pl, "Q_pitch");
  out.R_pitch = utils::read<std::vector<double>>(pl, "R_pitch");
}

void load_aim_planner(AimPlannerConfig & out, const YAML::Node & root)
{
  // AimPlanner 在 hero.yaml 才有 armor_hysteresis / omega_threshold / window_angle 三字段；
  // 其余 12 字段同 Planner。为避免非 hero yaml 加载 AimPlanner 时 required 缺崩，
  // 整段读取用 IsDefined 守卫：缺三字段则跳过整段 AimPlanner 填充（其他 SubConfig 仍正常）。
  if (!root["Planner"]) return;
  const auto pl = root["Planner"];
  if (!pl["armor_hysteresis"] || !pl["omega_threshold"] || !pl["window_angle"]) {
    return;  // 非 hero 兵种 yaml：AimPlanner SubConfig 不填充，沿用默认 0 值
  }
  out.yaw_offset = utils::read<double>(pl, "yaw_offset");
  out.pitch_offset = utils::read<double>(pl, "pitch_offset");
  out.fire_thresh = utils::read<double>(pl, "fire_thresh");
  out.decision_speed = utils::read<double>(pl, "decision_speed");
  out.high_speed_delay_time = utils::read<double>(pl, "high_speed_delay_time");
  out.low_speed_delay_time = utils::read<double>(pl, "low_speed_delay_time");
  out.max_yaw_acc = utils::read<double>(pl, "max_yaw_acc");
  out.Q_yaw = utils::read<std::vector<double>>(pl, "Q_yaw");
  out.R_yaw = utils::read<std::vector<double>>(pl, "R_yaw");
  out.max_pitch_acc = utils::read<double>(pl, "max_pitch_acc");
  out.Q_pitch = utils::read<std::vector<double>>(pl, "Q_pitch");
  out.R_pitch = utils::read<std::vector<double>>(pl, "R_pitch");
  out.armor_hysteresis = utils::read<double>(pl, "armor_hysteresis");
  out.omega_threshold = utils::read<double>(pl, "omega_threshold");
  out.window_angle = utils::read<double>(pl, "window_angle");
}

// ---------------------------------------------------------------------------------
// Aimer / Shooter
// ---------------------------------------------------------------------------------

void load_aimer(AimerConfig & out, const YAML::Node & root)
{
  if (!root["Aimer"]) return;
  const auto ai = root["Aimer"];
  out.yaw_offset = ai["yaw_offset"].as<double>();
  out.pitch_offset = ai["pitch_offset"].as<double>();
  out.comming_angle = ai["comming_angle"].as<double>();
  out.leaving_angle = ai["leaving_angle"].as<double>();
  out.high_speed_delay_time = ai["high_speed_delay_time"].as<double>();
  out.low_speed_delay_time = ai["low_speed_delay_time"].as<double>();
  out.decision_speed = ai["decision_speed"].as<double>();
}

void load_shooter(ShooterConfig & out, const YAML::Node & root)
{
  if (!root["Shooter"]) return;
  const auto sh = root["Shooter"];
  out.first_tolerance = sh["first_tolerance"].as<double>();
  out.second_tolerance = sh["second_tolerance"].as<double>();
  out.judge_distance = sh["judge_distance"].as<double>();
  out.auto_fire = sh["auto_fire"].as<bool>();
}

// ---------------------------------------------------------------------------------
// IO
// ---------------------------------------------------------------------------------

// 通用：从 yaml["Gimbal"] 段读 com_port + 可选 q_calib（Gimbal / Sentry 复用）。
void load_gimbal_like(
  std::string & com_port_out,
  std::optional<Eigen::Quaterniond> & q_calib_out,
  const YAML::Node & gimbal_yaml)
{
  com_port_out = utils::read<std::string>(gimbal_yaml, "com_port");

  // 与 gimbal.cpp:19 一致：if (q_calib_node) 即 IsDefined。
  // 存在则 w/x/y/z 子键 required（.as<double>() 缺则抛），与原行为一致。
  if (gimbal_yaml["q_calib"]) {
    const auto qc = gimbal_yaml["q_calib"];
    const double w = qc["w"].as<double>();
    const double x = qc["x"].as<double>();
    const double y = qc["y"].as<double>();
    const double z = qc["z"].as<double>();
    q_calib_out = Eigen::Quaterniond(w, x, y, z).normalized();
  }
  // 整组缺：保留 nullopt；消费侧改造时按 .has_value() 选 identity。
}

void load_gimbal(GimbalConfig & out, const YAML::Node & root)
{
  if (!root["Gimbal"]) return;
  load_gimbal_like(out.com_port, out.q_calib, root["Gimbal"]);
}

void load_dm_imu(DmImuConfig & out, const YAML::Node & root)
{
  if (!root["DM_IMU"]) return;
  const auto dm = root["DM_IMU"];
  out.imu_com_port = utils::read<std::string>(dm, "imu_com_port");
  out.baud = utils::read<int>(dm, "baud", out.baud);
  out.publish_rate = utils::read<int>(dm, "publish_rate", out.publish_rate);
}

void load_sentry(SentryConfig & out, const YAML::Node & root)
{
  // Sentry 复用 yaml["Gimbal"] 段（DECISIONS 2026-05-12）。
  if (!root["Gimbal"]) return;
  load_gimbal_like(out.com_port, out.q_calib, root["Gimbal"]);
}

void load_dart(DartConfig & out, const YAML::Node & root)
{
  if (!root["lower_Dart"]) return;
  out.com_port = utils::read<std::string>(root["lower_Dart"], "com_port");
}

// ---------------------------------------------------------------------------------
// auto_base
// ---------------------------------------------------------------------------------

void load_openvino_infer(OpenvinoInferConfig & out, const YAML::Node & base_hit)
{
  out.Openvino_XML = base_hit["Openvino_XML"].as<std::string>();
  out.device = base_hit["device"].as<std::string>(out.device);
  out.score_threshold = base_hit["score_threshold"].as<float>(out.score_threshold);
  out.nms_threshold = base_hit["nms_threshold"].as<float>(out.nms_threshold);
}

void load_base_hit(BaseHitConfig & out, const YAML::Node & root)
{
  if (!root["Base_Hit"]) return;
  load_openvino_infer(out.openvino, root["Base_Hit"]);
}

void load_light_tracker(LightTrackerConfig & out, const YAML::Node & root)
{
  if (!root["LightTracker"]) return;
  const auto lt = root["LightTracker"];
  out.min_detect_count = lt["min_detect_count"].as<int>();
  out.max_temp_lost_count = lt["max_temp_lost_count"].as<int>();
}

void load_light_aimer(LightAimerConfig & out, const YAML::Node & root)
{
  if (!root["LightAimer"]) return;
  const auto la = root["LightAimer"];
  if (la["begin_x"].IsDefined()) out.begin_x = la["begin_x"].as<double>();
  if (la["base_offset"].IsDefined()) out.base_offset = la["base_offset"].as<double>();
  if (la["offsets"].IsDefined()) {
    for (auto it = la["offsets"].begin(); it != la["offsets"].end(); ++it) {
      const int number = std::stoi(it->first.as<std::string>());
      out.offsets[number] = it->second.as<double>();
    }
  }
}

// ---------------------------------------------------------------------------------
// utils
// ---------------------------------------------------------------------------------

void load_video(VideoConfig & out, const YAML::Node & root)
{
  if (!root["Video"]) return;
  out.video_path = root["Video"]["video_path"].as<std::string>();
}

// ---------------------------------------------------------------------------------
// 元信息（迁移自 1.B.1 沉淀的 log_config_file_info）
// ---------------------------------------------------------------------------------

void fill_source_metadata(AppConfig & cfg, const std::string & yaml_path)
{
  try {
    const auto absolute_path = std::filesystem::absolute(yaml_path);
    cfg.source_path = absolute_path.string();
    cfg.source_file_size = static_cast<std::uintmax_t>(std::filesystem::file_size(absolute_path));
    cfg.source_last_write_raw =
      std::filesystem::last_write_time(absolute_path).time_since_epoch().count();
  } catch (const std::exception & e) {
    utils::logger()->warn("[AppConfig] source metadata unavailable: {}", e.what());
  }
}

}  // namespace

// ---------------------------------------------------------------------------------
// AppConfig::load
// ---------------------------------------------------------------------------------

AppConfig AppConfig::load(const std::string & yaml_path)
{
  AppConfig cfg;

  // utils::load 内部失败 exit(1)，与 sprint 之前各模块行为一致。
  const auto root = utils::load(yaml_path);

  fill_source_metadata(cfg, yaml_path);

  utils::logger()->info("[AppConfig] loading: {}", cfg.source_path);
  utils::logger()->info("[AppConfig] file_size            = {} bytes", cfg.source_file_size);
  utils::logger()->info("[AppConfig] last_write_time_raw  = {}", cfg.source_last_write_raw);

  load_camera(cfg.camera, root);
  load_detector(cfg.detector, root);
  load_solver(cfg.solver, root);
  load_tracker(cfg.tracker, root);
  load_planner(cfg.planner, root);
  load_aim_planner(cfg.aim_planner, root);
  load_aimer(cfg.aimer, root);
  load_shooter(cfg.shooter, root);
  load_gimbal(cfg.gimbal, root);
  load_dm_imu(cfg.dm_imu, root);
  load_sentry(cfg.sentry, root);
  load_dart(cfg.dart, root);
  load_base_hit(cfg.base_hit, root);
  load_light_tracker(cfg.light_tracker, root);
  load_light_aimer(cfg.light_aimer, root);
  load_video(cfg.video, root);

  utils::logger()->info("[AppConfig] loaded successfully");
  return cfg;
}

}  // namespace app_config
