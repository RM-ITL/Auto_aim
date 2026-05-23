// ===================================================================================
// test_load.cpp
//
// Sprint 1.C Phase 0 烟雾测试：加载一份 yaml 到 AppConfig，逐 SubConfig 打印
// 关键字段，便于与 sprint 1.B 之后 standard3_node 等入口的启动日志做字符级
// 比对。
//
// 用法：
//   ros2 run app_config test_load src/config/standard3.yaml
//   ros2 run app_config test_load src/config/sentry.yaml
//   ...
//
// 退出码：
//   0   = 成功（utils::load + 全部 SubConfig 填充无抛）
//   1   = 加载失败（由 utils::load 内部 exit(1) 产生）
//
// 本文件不入正式 build；只在 BUILD_TESTING=ON 时编译（CMakeLists.txt 已配置）。
// ===================================================================================
#include <iostream>
#include <string>

#include "app_config/app_config.hpp"
#include "logger.hpp"

namespace
{

template <typename T>
void print_vec(const std::string & name, const std::vector<T> & v)
{
  std::cout << "  " << name << " = [";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << v[i];
  }
  std::cout << "] (size=" << v.size() << ")\n";
}

void dump(const app_config::AppConfig & c)
{
  std::cout << "==== AppConfig dump ====\n";
  std::cout << "[source]\n"
            << "  path = " << c.source_path << "\n"
            << "  file_size = " << c.source_file_size << " bytes\n"
            << "  last_write_raw = " << c.source_last_write_raw << "\n";

  std::cout << "[camera]\n"
            << "  type = " << c.camera.type << "\n";
  if (c.camera.type == "hik") {
    std::cout << "  hik.exposure_ms  = " << c.camera.hik.exposure_ms << "\n"
              << "  hik.gain         = " << c.camera.hik.gain << "\n"
              << "  hik.fps          = " << c.camera.hik.fps << "\n"
              << "  hik.target_w x h = " << c.camera.hik.target_width << " x "
              << c.camera.hik.target_height << "\n"
              << "  hik.image_topic  = " << c.camera.hik.image_topic << "\n";
  } else if (c.camera.type == "mindvision") {
    std::cout << "  mindvision.exposure_ms = " << c.camera.mindvision.exposure_ms << "\n"
              << "  mindvision.gamma       = " << c.camera.mindvision.gamma << "\n"
              << "  mindvision.vid_pid     = " << c.camera.mindvision.vid_pid << "\n";
  }

  std::cout << "[detector]\n"
            << "  yolo_name = " << c.detector.yolo_name << "\n";
  if (c.detector.yolo_name == "yolov5") {
    std::cout << "  yolov5.model_path = " << c.detector.yolov5.yolov5_model_path << "\n"
              << "  yolov5.device     = " << c.detector.yolov5.device << "\n"
              << "  yolov5.min_conf   = " << c.detector.yolov5.min_confidence << "\n"
              << "  yolov5.score_th   = " << c.detector.yolov5.score_threshold << "\n"
              << "  yolov5.nms_th     = " << c.detector.yolov5.nms_threshold << "\n"
              << "  yolov5.use_trad   = " << std::boolalpha << c.detector.yolov5.use_traditional
              << "\n";
    std::cout << "  traditional.threshold        = " << c.detector.traditional.threshold << "\n"
              << "  traditional.max_angle_error  = " << c.detector.traditional.max_angle_error
              << " deg\n"
              << "  classifier.classify_model    = " << c.detector.classifier.classify_model
              << "\n"
              << "  classifier.min_confidence    = " << c.detector.classifier.min_confidence
              << "\n";
  } else if (c.detector.yolo_name == "yolo11") {
    std::cout << "  yolo11.model_path = " << c.detector.yolo11.yolo11_model_path << "\n"
              << "  yolo11.device     = " << c.detector.yolo11.device << "\n"
              << "  yolo11.min_conf   = " << c.detector.yolo11.min_confidence << "\n"
              << "  yolo11.score_th   = " << c.detector.yolo11.score_threshold << "\n"
              << "  yolo11.nms_th     = " << c.detector.yolo11.nms_threshold << "\n"
              << "  yolo11.enemy_color = " << c.detector.yolo11.enemy_color << "\n";
  }

  std::cout << "[solver.pnp]\n";
  print_vec("focal_length", c.solver.pnp.focal_length);
  print_vec("principal_point", c.solver.pnp.principal_point);
  print_vec("disto_param", c.solver.pnp.disto_param);

  std::cout << "[solver.coord_converter]\n";
  print_vec("focal_length", c.solver.coord_converter.focal_length);
  print_vec("principal_point", c.solver.coord_converter.principal_point);
  print_vec("disto_param", c.solver.coord_converter.disto_param);
  print_vec("R_camera_to_gimbal", c.solver.coord_converter.rotation_matrix_camera_to_gimbal);
  print_vec("R_gimbal_to_imu", c.solver.coord_converter.rotation_matrix_gimbal_to_imu);
  print_vec("t_camera_to_gimbal", c.solver.coord_converter.t_camera_to_gimbal);

  std::cout << "[tracker]\n"
            << "  enemy_color                   = " << c.tracker.enemy_color << "\n"
            << "  min_detect_count              = " << c.tracker.min_detect_count << "\n"
            << "  max_temp_lost_count           = " << c.tracker.max_temp_lost_count << "\n"
            << "  outpost_max_temp_lost_count   = " << c.tracker.outpost_max_temp_lost_count
            << "\n"
            << "  outpost_min_detect_count      = " << c.tracker.outpost_min_detect_count << "\n"
            << "  outpost_detect_fail_tolerance = " << c.tracker.outpost_detect_fail_tolerance
            << "\n"
            << "  single_plate_threshold        = " << c.tracker.single_plate_threshold << "\n"
            << "  omega_threshold               = " << c.tracker.omega_threshold << "\n"
            << "  single_plate_debug            = " << std::boolalpha << c.tracker.single_plate_debug
            << "\n";

  std::cout << "[planner]\n"
            << "  yaw_offset (deg)         = " << c.planner.yaw_offset << "\n"
            << "  pitch_offset (deg)       = " << c.planner.pitch_offset << "\n"
            << "  fire_thresh              = " << c.planner.fire_thresh << "\n"
            << "  decision_speed           = " << c.planner.decision_speed << "\n"
            << "  high_speed_delay_time    = " << c.planner.high_speed_delay_time << "\n"
            << "  low_speed_delay_time     = " << c.planner.low_speed_delay_time << "\n"
            << "  max_yaw_acc              = " << c.planner.max_yaw_acc << "\n"
            << "  max_pitch_acc            = " << c.planner.max_pitch_acc << "\n";
  print_vec("Q_yaw", c.planner.Q_yaw);
  print_vec("R_yaw", c.planner.R_yaw);
  print_vec("Q_pitch", c.planner.Q_pitch);
  print_vec("R_pitch", c.planner.R_pitch);

  std::cout << "[aim_planner]\n"
            << "  yaw_offset (deg)   = " << c.aim_planner.yaw_offset << "\n"
            << "  armor_hysteresis   = " << c.aim_planner.armor_hysteresis << "\n"
            << "  omega_threshold    = " << c.aim_planner.omega_threshold << "\n"
            << "  window_angle (deg) = " << c.aim_planner.window_angle << "\n";

  std::cout << "[aimer]\n"
            << "  yaw_offset (deg)       = " << c.aimer.yaw_offset << "\n"
            << "  pitch_offset (deg)     = " << c.aimer.pitch_offset << "\n"
            << "  comming_angle (deg)    = " << c.aimer.comming_angle << "\n"
            << "  leaving_angle (deg)    = " << c.aimer.leaving_angle << "\n"
            << "  decision_speed         = " << c.aimer.decision_speed << "\n"
            << "  high_speed_delay_time  = " << c.aimer.high_speed_delay_time << "\n"
            << "  low_speed_delay_time   = " << c.aimer.low_speed_delay_time << "\n";

  std::cout << "[shooter]\n"
            << "  first_tolerance (deg)  = " << c.shooter.first_tolerance << "\n"
            << "  second_tolerance (deg) = " << c.shooter.second_tolerance << "\n"
            << "  judge_distance         = " << c.shooter.judge_distance << "\n"
            << "  auto_fire              = " << std::boolalpha << c.shooter.auto_fire << "\n";

  std::cout << "[gimbal]\n"
            << "  com_port = " << c.gimbal.com_port << "\n";
  if (c.gimbal.q_calib) {
    const auto & q = *c.gimbal.q_calib;
    std::cout << "  q_calib  = (w=" << q.w() << ", x=" << q.x() << ", y=" << q.y()
              << ", z=" << q.z() << ")\n";
  } else {
    std::cout << "  q_calib  = (missing, fallback identity)\n";
  }

  std::cout << "[dm_imu]\n"
            << "  imu_com_port = " << c.dm_imu.imu_com_port << "\n"
            << "  baud         = " << c.dm_imu.baud << "\n"
            << "  publish_rate = " << c.dm_imu.publish_rate << " Hz\n";

  std::cout << "[sentry]\n"
            << "  com_port = " << c.sentry.com_port << "\n"
            << "  q_calib  = " << (c.sentry.q_calib ? "present" : "missing") << "\n";

  std::cout << "[dart]\n"
            << "  com_port = " << c.dart.com_port << "\n";

  std::cout << "[base_hit.openvino]\n"
            << "  Openvino_XML    = " << c.base_hit.openvino.Openvino_XML << "\n"
            << "  device          = " << c.base_hit.openvino.device << "\n"
            << "  score_threshold = " << c.base_hit.openvino.score_threshold << "\n"
            << "  nms_threshold   = " << c.base_hit.openvino.nms_threshold << "\n";

  std::cout << "[light_tracker]\n"
            << "  min_detect_count    = " << c.light_tracker.min_detect_count << "\n"
            << "  max_temp_lost_count = " << c.light_tracker.max_temp_lost_count << "\n";

  std::cout << "[light_aimer]\n"
            << "  begin_x      = " << c.light_aimer.begin_x << "\n"
            << "  base_offset  = " << c.light_aimer.base_offset << "\n"
            << "  offsets.size = " << c.light_aimer.offsets.size() << "\n";
  for (const auto & [k, v] : c.light_aimer.offsets) {
    std::cout << "  offsets[" << k << "] = " << v << "\n";
  }

  std::cout << "[video]\n"
            << "  video_path = " << c.video.video_path << "\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 2) {
    std::cerr << "usage: " << (argc > 0 ? argv[0] : "test_load") << " <yaml_path>\n";
    return 2;
  }
  const std::string path = argv[1];

  const auto cfg = app_config::AppConfig::load(path);
  dump(cfg);
  return 0;
}
