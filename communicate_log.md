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


 ---
  问题回顾

  当前观测向量是 (yaw, pitch, distance, armor_yaw)，没有直接观测z。根据研究表明：

  "在跟踪应用中，目标运动通常最好用笛卡尔坐标建模。但大多数系统的目标位置测量是以距离和方位角提供的。" - https://www.researchgate.net/publication/3832230_On_polar_and_versus_Cartesian_coordinates_for_target_tracking

  ---
  修改方案分析

  方案1：增加z坐标作为观测量（推荐）

  将观测向量从4维扩展到5维：

  当前：z = [yaw, pitch, distance, armor_yaw]     (4维)
  修改：z = [yaw, pitch, distance, armor_yaw, z]  (5维)

  或者更激进一些，直接用笛卡尔坐标：
  修改：z = [x, y, z, armor_yaw]  (4维，但包含z)

  优点：
  - h1/h2能被z观测直接约束
  - 高度匹配会更准确
  - 符合 https://www.researchgate.net/publication/263964280_EKFUKF_Maneuvering_Target_Tracking_using_Coordinated_Turn_Models_with_PolarCartesian_Velocity 的最佳实践

  缺点：
  - 需要重新设计观测矩阵H和观测噪声R
  - z坐标的测量噪声需要单独标定

  方案2：改进h1/h2的初始化和更新

  当前问题：h1/h2是单次赋值，之后几乎不变

  修改思路：
  // 方案A：累积多次观测后再确定h
  // 不是首次观测就赋值，而是收集多次观测后取平均

  // 方案B：使用独立的低通滤波器
  h1_filtered = alpha * h1_filtered + (1-alpha) * (obs_z - center_z)

  // 方案C：增大过程噪声，让EKF有机会修正
  v_h = 0.01  // 而不是 0.001

  方案3：修正收敛判断

  当前收敛判断太简单，应该考虑h1/h2是否稳定：

  bool OutpostTarget::convergened() {
    // 检查h1/h2的协方差是否足够小
    double P_h1 = ekf_.P(9, 9);
    double P_h2 = ekf_.P(10, 10);

    bool h_converged = (P_h1 < threshold) && (P_h2 < threshold);

    if (update_count_ > 20 && observed_ids_.size() >= 2 && h_converged) {
      is_converged_ = true;
    }
    return is_converged_;
  }

  ---
  核心问题：观测模型能否修改为包含z？

  答案：完全可以！

  你有两种选择：

  选择A：保持球坐标，增加z观测

  // 观测向量 5维
  Eigen::VectorXd z(5);
  z << armor_pose.world_spherical.yaw,
       armor_pose.world_spherical.pitch,
       armor_pose.world_spherical.distance,
       armor_pose.world_orientation.yaw,
       armor_pose.world_position[2];  // 新增：直接观测z

  // 观测函数 h(x)
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = utils::xyz2ypd(xyz);
    auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / ARMOR_NUM);

    Eigen::VectorXd result(5);
    result << ypd[0], ypd[1], ypd[2], angle, xyz[2];  // 新增第5维
    return result;
  };

  选择B：改用笛卡尔坐标观测

  // 观测向量 4维 (x, y, z, angle)
  Eigen::VectorXd z(4);
  z << armor_pose.world_position[0],
       armor_pose.world_position[1],
       armor_pose.world_position[2],
       armor_pose.world_orientation.yaw;

  // 观测函数更简单
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::Vector3d xyz = h_armor_xyz(x, id);
    auto angle = utils::limit_rad(x[6] + id * 2 * CV_PI / ARMOR_NUM);
    return Eigen::Vector4d(xyz[0], xyz[1], xyz[2], angle);
  };

  // 雅可比矩阵也更简单，因为 H_xyz 就是 H_armor_xyza 的前3行

  根据研究，笛卡尔坐标在非机动段更精确，球坐标在机动段更鲁棒 - https://ieeexplore.ieee.org/document/6289970/。前哨站是匀速圆周运动，属于非机动目标，笛卡尔坐标可能更适合。

  ---
  推荐的修改方案

  | 优先级 | 修改内容                              |
  |--------|---------------------------------------|
  | P0     | 观测模型增加z坐标（选择A或B）         |
  | P1     | 修正convergened()判断，加入协方差检测 |
  | P2     | 改进h1/h2初始化，使用滤波而非单次赋值 |
  | P3     | 实现diverged()检测，防止滤波发散      |

  需要我帮你实现具体的代码修改吗？

  Sources:
  - https://www.researchgate.net/publication/3832230_On_polar_and_versus_Cartesian_coordinates_for_target_tracking
  - https://www.researchgate.net/publication/263964280_EKFUKF_Maneuvering_Target_Tracking_using_Coordinated_Turn_Models_with_PolarCartesian_Velocity
  - https://ieeexplore.ieee.org/document/6289970/
  - https://www.mathworks.com/help/fusion/ug/extended-kalman-filters.html


  | 问题          | 当前状态     | 建议修改                                   |
  |---------------|--------------|--------------------------------------------|
  | pitch是否必要 | 信息与z等价  | 可以用z替换，雅可比更简单                  |
  | center_z      | 一次固定     | 通过EKF持续修正                            |
  | h1/h2         | 一次赋值     | 通过EKF多次观测收敛                        |
  | ID匹配        | 依赖不准的h  | 分阶段：未收敛用角度跳变，收敛后用高度匹配 |
  | diverged()    | 返回false    | 检查协方差/残差是否异常                    |
  | convergened() | 只看更新次数 | 检查h的协方差是否足够小                    |