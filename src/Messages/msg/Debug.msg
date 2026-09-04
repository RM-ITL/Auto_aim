bool enable_control
bool fire
float32 yaw_offest
float32 pitch_offset
float32 yaw
float32 pitch
float32 yaw_gimbal
float32 pitch_gimbal
float32 fire_rate
float32 bullet_speed
float32 yaw_vel
float32 pitch_vel
float32 yaw_acc
float32 pitch_acc
float32 yaw_vel_gimbal
float32 pitch_vel_gimbal
float32 yaw_acc_gimbal
float32 pitch_acc_gimbal

# Small-spin yaw observability fields. Currently filled by test_pipeline/deep_node.
bool target_valid
int32 aim_plate_index
float32 target_yaw
float32 theta
float32 omega
float32 center_yaw
float32 radius
bool single_plate

# 图像<->IMU 偏移标定信号（tools/latency/measure_image_imu_offset.py 消费）。
# raw_armor_yaw: 每帧原始视觉世界方位角(world_spherical.yaw)，未经 EKF/规划滤波。
# 定靶转云台时，它应恒定；若随云台角速率抖动，抖动斜率即 Δt。
float32 raw_armor_yaw
# raw_gimbal_yaw: 与 raw_armor_yaw 同帧的曝光时刻云台 yaw（视觉管线用的 IMU 姿态）。
# 标定回归自变量用它的数值微分，而非 yaw_gimbal(反馈, 不同时刻)，以免管线延迟污染 Δt。
float32 raw_gimbal_yaw
bool raw_armor_valid
