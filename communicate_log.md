## 2025-10-26
- 梳理 `src/core/auto_aim` 下各子模块（aimer、detected、planner、pointer、solver、target、tracker）依赖，讨论合并为单一动态库的方案。
- 新建统一的 `auto_aim` 包：编写顶层 `CMakeLists.txt` 与 `package.xml`，将子模块编译为静态库后以 `--whole-archive` 聚合为一个共享库；删除原子包的 CMake 与 package 文件。
- 修复 tinympc 头文件未安装导致的构建错误，新增最小的 `auto_aim_module.cpp` 作为顶层共享库的占位源文件，确保链接成功。
- 复查构建：`colcon build --packages-select auto_aim` 通过，仅剩若干警告；总结问题原因并记录后续注意事项。

## 2025-10-27
- 优化 `src/core/auto_aim/CMakeLists.txt`：改为在构建目录生成 `auto_aim_force_link.cpp` 占位源码，同时直接引用 tinympc 原始头文件路径，避免仓库内额外空源文件和临时安装目录。
- 调整 tinympc 头文件布局，将 `planner/tinympc/include/*.hpp` 统一移动到 `planner/tinympc/include/tinympc/`，并同步更新源码引用及安装路径，确保 `#include <tinympc/...>` 在编译和安装阶段一致可见。
- 批量处理编译警告：为 `Plan` 等结构体补全默认成员初始化，修正多个构造函数的初始化顺序、符号位比较及 `printf`/`snprintf` 的格式化安全问题，并对未使用的参数或仅用于调试的变量加上 `[[maybe_unused]]` 标记，`colcon build --packages-select cboard auto_aim` 无警告通过。

## 2025-10-29
- 新建 `src/core/auto_buff/CMakeLists.txt` 与 `package.xml`，参照 auto_aim 将现有 `buff_data_type`、`buff_detect` 编译为静态库，并以 `--whole-archive` 聚合出 `libauto_buff.so`，同步补齐安装与 ament 导出。
- 深入梳理 auto_buff 模块：记录 `FanBlade`/`PowerRune` 等基础数据结构的字段与状态机设计；分析 YOLO11 推理接口 `get_multicandidateboxes`（多候选 NMS 输出）与 `get_onecandidatebox`（单候选最高置信度）在预处理、筛选策略上的差异；总结 `Buff_Detector` 封装流程，包括失追处理、旋转中心估计、与历史 `PowerRune` 融合的目标判定逻辑，为后续扩展（如更多子模块或识别策略）提供设计参考。


## 2025-12-03

  关键区别

  | 场景            | 结果    |
  |---------------|-------|
  | 不与下位机通讯（云台静止） | 无漂移 ✓ |
  | 开启下位机通讯（云台跟随） | 有漂移 ✗ |

  这说明问题不是上位机IMU本身的零漂，而是上位机和下位机之间的坐标系不一致。

  ---
  根本原因：两个IMU的"世界"定义不同

  ┌─────────────────────────────────────────────────────────────────┐
  │                      问题的本质                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │  上位机计算: yaw_target = atan2(y, x)                            │
  │              ↓ (基于上位机IMU的世界坐标系)                        │
  │  发送给下位机: "请把云台转到 yaw = -0.197"                        │
  │              ↓                                                   │
  │  下位机执行: 让 yaw_gimbal = -0.197                              │
  │              ↑ (基于下位机IMU的世界坐标系)                        │
  │                                                                 │
  │  问题: 上位机的 -0.197 和 下位机的 -0.197 不是同一个方向！        │
  │        因为两个IMU定义的"世界坐标系零点"不同，而且在漂移          │
  └─────────────────────────────────────────────────────────────────┘

  具体解释

  1. 上位机的世界坐标系：由上位机IMU在上电时刻定义
  2. 下位机的世界坐标系：由下位机IMU在上电时刻定义

  虽然"同时上电，位置尽量一样"，但：
  - 两个IMU的初始零点可能有微小差异（安装精度）
  - 两个IMU的零漂速率不同（MEMS陀螺仪特性）

  假设：
  - 上位机IMU零漂 ω₁ = 0.3°/s
  - 下位机IMU零漂 ω₂ = 0.1°/s
  - 5秒后，两个坐标系的偏差 = (0.3 - 0.1) × 5 = 1°

  这1°的偏差导致云台实际指向比上位机期望的偏了1°，表现为像素漂移！

  ---
  为什么"yaw跟随得好"但像素漂移？

  你对比的是:
    yaw (上位机发送的命令)  vs  yaw_gimbal (下位机反馈)
    → 它们差距小，说明"命令-执行"跟随好

  但真正的问题是:
    上位机IMU测量的yaw  vs  下位机IMU/编码器测量的yaw
    → 这两者可能有一个在变化的偏差！

  下位机说 "我已经转到 -0.197 了"
  上位机说 "我的IMU显示当前是 -0.199"
  → 重投影用的是 -0.199，但云台实际在 -0.197
  → 差 0.002 rad ≈ 0.11° ≈ 2.5 像素的误差
  → 这个误差如果持续累积，就形成漂移

  ---
  验证方法

  在代码中增加日志，同时打印上位机IMU测量的yaw：

  // 在 updateIMU 之后
  double imu_yaw = atan2(R_gimbal_to_world(1,0), R_gimbal_to_world(0,0));
  utils::logger()->info("上位机IMU_yaw: {:.6f}, 下位机yaw_gimbal: {:.6f}, 
  差值: {:.6f}",
      imu_yaw, yaw_gimbal, imu_yaw - yaw_gimbal);

  如果差值在单调变化，就100%确认了问题。

  ---
  解决方案

  方案1：统一角度参考（推荐）

  使用下位机反馈的 yaw_gimbal 来更新上位机的坐标变换，而不是用上位机IMU：

  // 新增接口：用下位机反馈的角度更新坐标变换
  void CoordConverter::updateFromGimbalFeedback(double yaw_gimbal, double 
  pitch_gimbal) {
      // 用下位机反馈构建旋转矩阵，保证上下位机坐标系一致
      Eigen::AngleAxisd yaw_rotation(yaw_gimbal, Eigen::Vector3d::UnitZ());
      Eigen::AngleAxisd pitch_rotation(pitch_gimbal, 
  Eigen::Vector3d::UnitY());

      R_gimbal_to_world = (yaw_rotation *
  pitch_rotation).toRotationMatrix();
  }

  这样上位机计算出的yaw命令就是基于下位机的坐标系，消除了不一致。

  方案2：定期同步零点

  定期让上位机的世界坐标系和下位机对齐：
  // 每隔一段时间执行一次零点同步
  void syncWorldFrame(double yaw_gimbal, double pitch_gimbal) {
      // 计算上位机IMU和下位机的偏差
      double imu_yaw = atan2(R_gimbal_to_world(1,0),
  R_gimbal_to_world(0,0));
      yaw_offset_ = imu_yaw - yaw_gimbal;  // 保存偏差
  }

  // 发送命令时补偿偏差
  yaw_to_send = calculated_yaw - yaw_offset_;

  方案3：下位机使用编码器而非IMU

  如果下位机的云台有编码器，让下位机使用编码器反馈（编码器不会零漂），这样至
  少下位机这边是稳定的。

  ---
  总结

  | 问题         | 原因                                 |
  |------------|------------------------------------|
  | 不通讯时无漂移    | 云台静止，上位机自洽                         |
  | 通讯时有漂移     |
  上下位机使用不同IMU定义坐标系，零漂速率不同导致坐标系偏差累积   |
  | yaw跟随好但像素漂 |
  你对比的是命令vs反馈，但实际问题是两个IMU的世界坐标系定义在漂移 |

  核心问题：上位机和下位机的"世界"不是同一个世界。
