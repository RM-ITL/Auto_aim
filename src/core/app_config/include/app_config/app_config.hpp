// ===================================================================================
// app_config.hpp
//
// Sprint 1.C 引入：参数集中加载与 SubConfig 类型视图。
//
// 设计原则（严格遵守，1.C 不破坏）：
//
//   1. 字段名 1:1 映射当前 yaml 字面字段名（snake_case 保留，不规整命名）。
//   2. 严格零行为变化：默认值与原模块硬编码默认值字节级一致；同名漂移（如
//      Planner.yaw_offset vs Aimer.yaw_offset）保留独立字段不合并。语义合并
//      留给 Sprint 1.C.1。
//   3. 单位保留 yaml 字面值（如 yaw_offset 存 deg），模块构造时再换算（/ 57.3）。
//   4. required 字段不在结构体声明处给默认值，from_yaml 用 utils::read<T>(node, key)
//      无默认重载，缺崩；可选字段给默认值，from_yaml 用 utils::read<T>(node, key,
//      default) 三参数重载。
//   5. 整组可缺的嵌套段（如 Gimbal.q_calib）用 std::optional<T>。
//   6. 不引入 enum（camera.type / camera.lens 仍是 string）；schema 校验留给后续。
//
// AppConfig::load 在入口处一次解析整份 yaml，按 SubConfig 分发到各模块。
//
// 当前 SubConfig 字段速查表见每个结构体上方的注释：来源 yaml 节点 / 使用模块。
// ===================================================================================
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace app_config
{

// ===================================================================================
// Camera
// ===================================================================================

// 来源：yaml["camera"]["parameters"]（type=hik 时生效）
// 使用：camera::HikCamera 构造
//
// 注意：HikCamera 内部对 exposure_ms × 1000 得到 exposure_us，SubConfig 保留 ms。
struct CameraHikConfig
{
  double exposure_ms{};            // required
  double gain{};                   // required
  double fps{};                    // required
  int target_width{};              // required
  int target_height{};             // required
  std::string image_topic{};       // required
};

// 来源：yaml["camera"]["parameters"]（type=mindvision 时生效）
// 使用：camera::Camera 外壳提取后逐字段传入 camera::MindVision 构造
struct CameraMindVisionConfig
{
  double exposure_ms = 5.0;
  double gamma = 100.0;
  std::string vid_pid = "2bdf:0283";
};

// 来源：yaml["camera"]
// 使用：camera::Camera 外壳根据 type 选用 hik / mindvision
struct CameraConfig
{
  std::string lens{};              // required；用于选择 CalibParam...profiles.<lens>
  std::string type = "hik";        // 默认 hik；外壳条件读
  CameraHikConfig hik{};
  CameraMindVisionConfig mindvision{};
};

// ===================================================================================
// Detector（auto_aim）
// ===================================================================================

// 来源：yaml["yolo"]（yolo_name="yolov5" 时生效）
// 使用：armor_auto_aim::YOLOV5Detector 构造
struct DetectorYOLOv5Config
{
  std::string yolov5_model_path{};   // required
  std::string device{};              // required（注意：YOLO11 是 default "CPU"，此处 required）
  double min_confidence{};           // required
  float score_threshold = 0.7f;
  float nms_threshold = 0.3f;
  bool use_traditional = false;
};

// 来源：yaml["yolo"]（yolo_name="yolo11" 时生效）
// 使用：armor_auto_aim::YOLO11Detector 构造
struct DetectorYOLO11Config
{
  std::string yolo11_model_path{};   // required
  std::string device = "CPU";
  double min_confidence = 0.8;
  float score_threshold = 0.7f;
  float nms_threshold = 0.3f;
  std::string enemy_color = "red";   // 条件读取；YOLOV5Detector 不读此字段
};

// 来源：yaml["Traditional_Detector"]
// 使用：armor_auto_aim::Traditional_Detector 构造（同时由 YOLOV5Detector 内嵌使用）
//
// 注意：分类器输出过滤阈值 min_confidence 已迁到 ClassifierConfig，本段不再持有。
// 注意单位换算：max_angle_error / max_rectangular_error 在 yaml 是 deg，
//   模块构造时除 57.3 转 rad。SubConfig 保留 yaml 字面 deg 值。
struct DetectorTraditionalConfig
{
  double threshold{};                // required（yaml["Traditional_Detector"]["threshold"]，int→double）
  double max_angle_error{};          // required（deg, 模块 / 57.3）
  double min_lightbar_ratio{};       // required
  double max_lightbar_ratio{};       // required
  double min_lightbar_length{};      // required
  double min_armor_ratio{};          // required
  double max_armor_ratio{};          // required
  double max_side_ratio{};           // required
  double max_rectangular_error{};    // required（deg, 模块 / 57.3）
};

// 来源：yaml["Classifier"]
// 使用：armor_auto_aim::Classifier 构造（由 Traditional_Detector 内嵌使用）
//      min_confidence 消费点在 Traditional_Detector::check_name()
//
// 注意：Classifier 类自身只输出 armor.confidence，不执行 min_confidence 过滤；
//       过滤仍发生在 Traditional_Detector::check_name()。
// 注意：classifier 内部 openvino 推理设备硬编码 "AUTO"，非 yaml 字段。
struct ClassifierConfig
{
  std::string classify_model{};      // required（yaml["Classifier"]["classify_model"]）
  double min_confidence{};           // required（yaml["Classifier"]["min_confidence"]）
};

// 来源：yaml["yolo"]["yolo_name"] + yaml["yolo"] / yaml["Traditional_Detector"] / yaml["Classifier"]
// 使用：armor_auto_aim::Detector 外壳根据 yolo_name 选 yolov5 / yolo11
struct DetectorConfig
{
  std::string yolo_name = "yolo11";  // 条件读 + 外壳默认 "yolo11"
  DetectorYOLOv5Config yolov5{};
  DetectorYOLO11Config yolo11{};
  DetectorTraditionalConfig traditional{};
  ClassifierConfig classifier{};
};

// ===================================================================================
// Solver / 三个子模块（auto_aim）
// ===================================================================================
// 注意：Sprint 1.C.3 Phase 2 后，PnPSolver / CoordConverter / YawOptimizer 不再
//       各自持有 focal_length / principal_point / disto_param。相机内参统一由
//       SolverConfig.camera_intri 提供，消费侧通过 CameraIntriConfig 注入。

// 来源：yaml["CalibParam"]["INTRI"]["Camera"][0]["value"]["ptr_wrapper"]["data"]
//       下的 profiles[camera.lens]
// 使用：作为 PnPSolver / CoordConverter / YawOptimizer 三个模块的共享注入数据源。
struct CameraIntriConfig
{
  std::vector<double> focal_length{};     // required（取 [0]=fx, [1]=fy）
  std::vector<double> principal_point{};  // required（取 [0]=cx, [1]=cy）
  std::vector<double> disto_param{};      // 条件读；缺则 empty
};

// 使用：solver::PnPSolver 构造的非内参配置。
// 注意：PnPSolver 不再持有内参字段；内参由 SolverConfig.camera_intri 提供。
struct PnPSolverConfig
{
};

// 来源：yaml["Solver"]["coord_converter"]["rotation_matrix_*"]
//       + yaml["Solver"]["coord_converter"]["t_camera_to_gimbal"]
// 使用：solver::CoordConverter 构造
//
// 注意：CoordConverter 不再持有内参字段；内参由 SolverConfig.camera_intri 提供。
//       本结构只保留 rotation/t 非内参字段。
struct CoordConverterConfig
{
  std::vector<double> rotation_matrix_camera_to_gimbal{};      // 条件读，缺则 identity
  std::vector<double> rotation_matrix_gimbal_to_imu{};         // 条件读，缺则 identity
  std::vector<double> t_camera_to_gimbal{};                    // 条件读（Solver.coord_converter），缺则 zero
};

// 使用：solver::YawOptimizer 构造的非内参配置。
//
// 注意：search_range = 70 deg、search_step = 1 deg 在模块中硬编码（启动日志带
//       (HARDCODED) 标注），不进 SubConfig。
// 注意：YawOptimizer 不再持有内参字段；内参由 SolverConfig.camera_intri 提供。
struct YawOptimizerConfig
{
};

// 来源：上面三块整合
// 使用：solver::Solver 外壳分发给 PnPSolver / CoordConverter / YawOptimizer
struct SolverConfig
{
  CameraIntriConfig camera_intri{};  // PnPSolver / CoordConverter / YawOptimizer 共享内参来源
  PnPSolverConfig pnp{};
  CoordConverterConfig coord_converter{};
  YawOptimizerConfig yaw_optimizer{};
};

// ===================================================================================
// Tracker（auto_aim）
// ===================================================================================

// 来源：yaml["Tracker"]
// 使用：tracker::Tracker 构造
//
// 注意：Tracker.omega_threshold 与 Planner.omega_threshold（AimPlanner）同名不同
//       语义：前者是单板模式触发，后者是高低速模式切换。1.C 保留独立。
struct TrackerConfig
{
  std::string enemy_color{};            // required
  int min_detect_count{};               // required
  int max_temp_lost_count{};            // required
  int outpost_max_temp_lost_count{};    // required
  int outpost_min_detect_count{};       // required
  int outpost_detect_fail_tolerance{};  // required
  int single_plate_threshold = 50;
  double omega_threshold = 0.5;
  bool single_plate_debug = false;
};

// ===================================================================================
// Planner / AimPlanner / Aimer / Shooter（auto_aim）
// ===================================================================================

// 来源：yaml["Planner"]
// 使用：plan::Planner 构造（ctor + setup_yaw_solver + setup_pitch_solver 三段读取）
//
// 注意：yaw_offset / pitch_offset 在 yaml 是 deg，模块构造时 / 57.3。SubConfig 存 deg。
//
// 同名漂移：yaw_offset 字段在 Planner / AimPlanner / Aimer 三处独立（1.C 不合并）。
struct PlannerConfig
{
  // ctor 段
  double yaw_offset{};                  // required（deg, 模块 / 57.3）
  double pitch_offset{};                // required（deg, 模块 / 57.3）
  double fire_thresh{};                 // required
  double decision_speed{};              // required
  double high_speed_delay_time{};       // required
  double low_speed_delay_time{};        // required
  // setup_yaw_solver 段
  double max_yaw_acc{};                 // required
  std::vector<double> Q_yaw{};          // required
  std::vector<double> R_yaw{};          // required
  // setup_pitch_solver 段
  double max_pitch_acc{};               // required
  std::vector<double> Q_pitch{};        // required
  std::vector<double> R_pitch{};        // required
};

// 来源：yaml["Planner"]（hero.yaml 才有完整字段）
// 使用：plan::AimPlanner 构造（hero 入口当前实际仍用 plan::Planner，AimPlanner
//       为孤儿构造函数；本 sprint 仍预留 SubConfig）
//
// 注意：armor_hysteresis / omega_threshold / window_angle 是 AimPlanner 独有，
//       Planner ctor 不读这三个字段，所以当前 hero.cpp 用 Planner 时这些字段被忽略。
struct AimPlannerConfig
{
  // 与 PlannerConfig 完全重复的 12 个字段
  double yaw_offset{};                  // required（deg）
  double pitch_offset{};                // required（deg）
  double fire_thresh{};                 // required
  double decision_speed{};              // required
  double high_speed_delay_time{};       // required
  double low_speed_delay_time{};        // required
  double max_yaw_acc{};                 // required
  std::vector<double> Q_yaw{};          // required
  std::vector<double> R_yaw{};          // required
  double max_pitch_acc{};               // required
  std::vector<double> Q_pitch{};        // required
  std::vector<double> R_pitch{};        // required
  // AimPlanner ctor 独有 3 个字段
  double armor_hysteresis{};            // required（仅 hero）
  double omega_threshold{};             // required（仅 hero；与 Tracker.omega_threshold 同名不同语义）
  double window_angle{};                // required（deg, 模块 / 57.3，仅 hero）
};

// 来源：yaml["Aimer"]
// 使用：aimer::Aimer 构造
//
// 注意：yaw_offset / pitch_offset / comming_angle / leaving_angle 是 deg，
//       模块 / 57.3。注意拼写 "comming_angle" 不是 "coming_angle"。
//       left_yaw_offset / right_yaw_offset 已于 2026-04-30 删除（audit §5.1.b §5）。
struct AimerConfig
{
  double yaw_offset{};                  // required（deg）
  double pitch_offset{};                // required（deg）
  double comming_angle{};               // required（deg）
  double leaving_angle{};               // required（deg）
  double high_speed_delay_time{};       // required
  double low_speed_delay_time{};        // required
  double decision_speed{};              // required
};

// 来源：yaml["Shooter"]
// 使用：shooter::Shooter 构造
struct ShooterConfig
{
  double first_tolerance{};   // required（deg, 模块 / 57.3）
  double second_tolerance{};  // required（deg, 模块 / 57.3）
  double judge_distance{};    // required
  bool auto_fire{};           // required
};

// ===================================================================================
// IO（io::*）
// ===================================================================================

// 来源：yaml["Gimbal"]
// 使用：io::Gimbal 构造
//
// 注意：baud 在 Gimbal 实现里硬编码 115200（启动日志带 (HARDCODED) 标注），
//       不进 SubConfig。q_calib 整组可缺，缺则模块用 identity 并 warn。
struct GimbalConfig
{
  std::string com_port{};                              // required
  std::optional<Eigen::Quaterniond> q_calib;           // 整组可缺；存在则 w/x/y/z 四子键 required
};

// 来源：yaml["DM_IMU"]（1.D 已真嵌套）
// 使用：io::DmImu 构造
//
// 注意：模块用 publish_rate 算 interval_ms = clamp(round(1000/rate), 1, 100)。
struct DmImuConfig
{
  std::string imu_com_port{};   // required
  int baud = 921600;
  int publish_rate = 333;
};

// 来源：复用 yaml["Gimbal"] 段（lower_sentry 不引入独立 Sentry 段，
//       决策见 DECISIONS 2026-05-12）
// 使用：io::Sentry 构造
//
// 字段含义同 GimbalConfig。baud 同样硬编码 115200。
struct SentryConfig
{
  std::string com_port{};                              // required
  std::optional<Eigen::Quaterniond> q_calib;           // 整组可缺
};

// 来源：yaml["lower_Dart"]（1.D 已真嵌套）
// 使用：io::Dart 构造
//
// 注意：baud 同样硬编码 115200。
struct DartConfig
{
  std::string com_port{};   // required
};

// ===================================================================================
// auto_base
// ===================================================================================

// 来源：yaml["Base_Hit"]
// 使用：auto_base::OpenvinoInfer 构造（由 auto_base::Detector 外壳透传）
//
// 注意：Openvino_Deveice 已于 2026-04-30 重命名为 device（audit §5.1.b §6）。
//       input_width / input_height 来自模型 shape，不进 SubConfig。
struct OpenvinoInferConfig
{
  std::string Openvino_XML{};        // required
  std::string device = "CPU";
  float score_threshold = 0.5f;
  float nms_threshold = 0.45f;
};

// 来源：yaml["Base_Hit"] 整段
// 使用：auto_base::Detector 外壳分发给 OpenvinoInfer
struct BaseHitConfig
{
  OpenvinoInferConfig openvino{};
};

// 来源：yaml["LightTracker"]
// 使用：auto_base::LightTracker 构造
struct LightTrackerConfig
{
  int min_detect_count{};       // required
  int max_temp_lost_count{};    // required
};

// 来源：yaml["LightAimer"]
// 使用：auto_base::LightAimer 构造
//
// 注意：begin_x / base_offset / offsets 三字段都用 IsDefined 守卫 + warn 缺失。
//       offsets 是 map<int, double> 遍历未知 key 集合（key 是飞镖编号 1..4）。
struct LightAimerConfig
{
  double begin_x = 0.0;
  double base_offset = 0.0;
  std::map<int, double> offsets{};
};

// ===================================================================================
// utils
// ===================================================================================

// 来源：yaml["Video"]（1.D 已真嵌套）
// 使用：utils::Video 构造
struct VideoConfig
{
  std::string video_path{};   // required
};

// ===================================================================================
// 顶层 AppConfig
// ===================================================================================

struct AppConfig
{
  CameraConfig camera{};
  DetectorConfig detector{};
  SolverConfig solver{};
  TrackerConfig tracker{};
  PlannerConfig planner{};
  AimPlannerConfig aim_planner{};
  AimerConfig aimer{};
  ShooterConfig shooter{};
  GimbalConfig gimbal{};
  DmImuConfig dm_imu{};
  SentryConfig sentry{};
  DartConfig dart{};
  BaseHitConfig base_hit{};
  LightTrackerConfig light_tracker{};
  LightAimerConfig light_aimer{};
  VideoConfig video{};

  // 元信息（迁移自 1.B.1 沉淀的 log_config_file_info）
  // 由 AppConfig::load 填充；模块层若需打印 yaml 来源信息可读这些字段。
  std::string source_path{};
  std::uintmax_t source_file_size = 0;
  std::int64_t source_last_write_raw = 0;

  // 工厂：成功返回 AppConfig，失败 exit(1)（与现有 utils::load 行为一致）。
  // 内部只 YAML::LoadFile 一次，按 SubConfig 逐段填充。
  static AppConfig load(const std::string & yaml_path);
};

}  // namespace app_config
