# AimPlanner 修改与测试指导

本文档用于指导当前仓库 `/home/hou/Auto_aim` 的 Planner / AimPlanner 改造与测试。算法层面参考 `/home/hou/SHtech_auto_aim`，重点参考：

- `/home/hou/SHtech_auto_aim/planner/Planner.hpp`
- `/home/hou/SHtech_auto_aim/planner/Planner.cpp`
- `/home/hou/SHtech_auto_aim/planner/planner_submodule.cpp`
- `/home/hou/SHtech_auto_aim/asset/plannerParam/*.yml`

当前仓库中 `src/core/auto_aim/planner/src/aim_planner.cpp` 已有高速模式雏形，但主要节点仍构造 `plan::Planner`，所以只调 `AimPlanner` 参数不会在实车链路生效。修改时必须先确认 Planner 接入点。

## 1. 当前状态与核心风险

### 1.1 当前可用模块

当前仓库 Planner 相关文件：

- `src/core/auto_aim/planner/include/planner.hpp`
- `src/core/auto_aim/planner/src/planner.cpp`
- `src/core/auto_aim/planner/include/aim_planner.hpp`
- `src/core/auto_aim/planner/src/aim_planner.cpp`
- `src/core/auto_aim/planner/tinympc/*`

`auto_aim` 的 CMake 已把 `aim_planner.cpp` 编进 `auto_aim_planner`：

```cmake
set(AUTO_AIM_PLANNER_SOURCES
  planner/src/planner.cpp
  planner/src/aim_planner.cpp
  ...
)
```

这说明编译层面已经可用，不需要再额外把 `aim_planner.cpp` 加入 CMake。

### 1.2 当前未接入的问题

以下节点仍使用 `plan::Planner`：

- `src/node/src/standard3.cpp`
- `src/node/src/hero.cpp`
- `src/node/src/sentry.cpp`
- `src/test_node/src/test_node_deep.cpp`
- `src/test_node/src/test_node_cv.cpp`

对应头文件中也多为：

```cpp
#include "planner.hpp"
std::unique_ptr<plan::Planner> planner_;
```

如果目标是启用 `AimPlanner` 的高速守株待兔逻辑，需要把目标入口改成 `plan::AimPlanner`，并构造时传入 `app_config.aim_planner`。

### 1.3 配置加载现状

`app_config` 已预留 `AimPlannerConfig`：

- `src/core/app_config/include/app_config/app_config.hpp`
- `src/core/app_config/src/app_config.cpp`
- `src/core/app_config/src/validator.cpp`

但 `load_aim_planner()` 只有在 `Planner` 段同时存在以下字段时才会加载：

- `armor_hysteresis`
- `omega_threshold`
- `window_angle`

因此如果某个兵种配置缺少这三个字段，`aim_planner` 会保持默认 0 值，构造 `AimPlanner` 后会造成高速阈值、窗口角等参数无效。

## 2. 推荐修改目标

本轮修改建议分两级实施。

### 2.1 最小可用目标

目标：让 `AimPlanner` 真正进入实车和视频测试链路，并保持低速逻辑等价于当前 `Planner`。

需要完成：

- 将目标节点中的 `plan::Planner` 替换为 `plan::AimPlanner`。
- 将 `#include "planner.hpp"` 替换或补充为 `#include "aim_planner.hpp"`。
- 构造参数从 `app_config.planner` 改为 `app_config.aim_planner`。
- 在对应 YAML 的 `Planner` 段补齐 `armor_hysteresis`、`omega_threshold`、`window_angle`。
- 保留旧 `Planner` 用于前哨站或不需要高速模式的入口，避免一次性影响全部兵种。

### 2.2 完整算法目标

目标：参考 SHtech 的多策略 Planner，形成稳定的低速跟踪、中高速整车预测、高速中心预瞄和射击窗口控制。

建议拆成四个阶段：

1. 低速模式：继续使用 TinyMPC 跟踪装甲板，加入装甲板选择滞后。
2. 高速模式：云台 yaw 指向旋转中心，pitch 跟随窗口内装甲板高度。
3. 射击决策：高速基于装甲板朝向窗口，低速基于 MPC 跟踪误差。
4. 跳变抑制：装甲板切换后短时间禁止开火，避免切板瞬间误射。

## 3. 接入修改指导

### 3.1 选择接入范围

建议先只接入一个测试节点和一个实车节点：

- 视频/桌面验证：`src/test_node/src/test_node_deep.cpp` 和 `src/test_node/include/test_node_deep.hpp`
- 实车验证：优先选择当前要调的兵种，例如 `src/node/src/hero.cpp` 或 `src/node/src/standard3.cpp`

不要第一轮同时改 `standard3`、`hero`、`sentry` 和所有 test node。Planner 影响下位机控制指令，接入范围越大，排查越慢。

### 3.2 修改头文件

以 `standard3` 为例，把：

```cpp
#include "planner.hpp"
std::unique_ptr<plan::Planner> planner_;
```

改为：

```cpp
#include "aim_planner.hpp"
std::unique_ptr<plan::AimPlanner> planner_;
```

`hero.hpp`、`sentry.hpp`、`test_node_deep.hpp`、`test_node_cv.hpp` 同理。

### 3.3 修改构造函数

把：

```cpp
planner_ = std::make_unique<plan::Planner>(app_config.planner);
```

改为：

```cpp
planner_ = std::make_unique<plan::AimPlanner>(app_config.aim_planner);
```

如果某个入口仍需要支持前哨站 `OutpostTarget`，注意当前 `AimPlanner` 的 variant 重载只处理 `predict::Target`，遇到 `predict::OutpostTarget` 会返回 `{false}`。这类入口暂时继续使用 `Planner`，或为 `AimPlanner` 补齐 `OutpostTarget` 高低速逻辑。

### 3.4 修改配置文件

在目标 YAML 的 `Planner` 段补齐 `AimPlanner` 独有字段。示例：

```yaml
Planner:
  fire_thresh: 0.003
  max_yaw_acc: 20
  Q_yaw: [9e6, 0]
  R_yaw: [1]
  max_pitch_acc: 20
  Q_pitch: [9e6, 0]
  R_pitch: [1]
  yaw_offset: 2.2
  pitch_offset: -3.9
  decision_speed: 7
  high_speed_delay_time: 0.1
  low_speed_delay_time: 0.3
  armor_hysteresis: 0.92
  omega_threshold: 7.0
  window_angle: 18.0
```

参数说明：

- `armor_hysteresis`：低速多板选择滞后系数，降低相邻装甲板频繁切换。建议初值 `0.90` 到 `0.98`。
- `omega_threshold`：`AimPlanner` 高速模式切换阈值，单位 rad/s。建议初值与 `decision_speed` 一致或略低。
- `window_angle`：高速模式允许开火的装甲板朝向窗口半角，单位 degree。建议初值 `15` 到 `25`。

注意：`Tracker.omega_threshold` 和 `Planner.omega_threshold` 语义不同。前者用于单板观测模式，后者用于 AimPlanner 高速模式，不能直接混用。

## 4. 算法修改指导

### 4.1 低速模式：保持 TinyMPC 主链路

当前 `AimPlanner::plan(predict::Target, double)` 的低速分支已基本复用 `Planner`：

1. 根据 bullet speed 估计飞行时间。
2. `target.predict(fly_time)` 做弹丸飞行补偿。
3. `aim()` 计算 yaw / pitch。
4. `get_trajectory()` 生成 HORIZON 参考轨迹。
5. TinyMPC 分别求 yaw / pitch。
6. 基于 `fire_thresh_` 判断 `plan.fire`。

低速模式建议只做两类增强：

- 装甲板选择滞后：当前 `AimPlanner::aim()` 已实现，继续保留。
- 装甲板跳变禁火：参考 SHtech `same_position_threshold` 和 `armor_jump_interval`，在切板后的短时间内禁止开火。

低速不建议直接改为中心瞄准。低速下装甲板模型可观测性更好，直接跟板更稳定。

### 4.2 高速模式：中心预瞄加射击窗口

当前高速分支的思路正确：

- `select_armor_high_speed()` 选择最正对枪口的装甲板。
- `compute_yaw_high_speed()` 指向车辆旋转中心。
- `compute_pitch_high_speed()` 使用选中装甲板高度做弹道 pitch。
- `should_fire_high_speed()` 用 `facing < window_angle_` 判断开火。

建议补充两点：

#### 4.2.1 高速 yaw 加入中心速度前馈

当前 `compute_yaw_high_speed()` 只指向当前中心：

```cpp
double cx = x[0], cy = x[2];
double azim = std::atan2(cy, cx);
```

如果目标横向移动明显，应加入总延迟后的中心预测。参考当前状态向量用法，`ekf_x()` 常见含义可按当前 `Target` 实现核对。若状态为 `[x, vx, y, vy, z, vz, yaw, v_yaw, ...]`，可改为：

```cpp
double delay = estimated_total_delay;
double cx = x[0] + x[1] * delay;
double cy = x[2] + x[3] * delay;
double azim = std::atan2(cy, cx);
```

这里不要硬编码状态含义，必须先核对 `target/predictor/src/target.cpp` 和 `target/predictor/include/target.hpp`。

#### 4.2.2 高速 fire 增加闭环误差限制

仅靠 `facing < window_angle_` 会在云台尚未对准中心时过早允许开火。建议增加 yaw / pitch 当前误差，或者在下游 Shooter 具备严格闭环判断时明确职责边界。

推荐判据：

```cpp
plan.fire = facing < window_angle_ && center_tracking_error < fire_thresh_high_speed;
```

如果 Planner 拿不到当前云台闭环误差，则保持 `plan.fire` 为窗口预使能，把最终开火交给 `Shooter` 的角度容差。

### 4.3 中速模式：不要过早引入复杂策略

SHtech 中有多策略：

- `ARMOR_WITH_NO_MODEL`
- `ARMOR_WITH_ARMOR_MODEL`
- `ARMOR_WITH_VEHICLE_MODEL`
- `VEHICLE_CENTER_WITH_VEHICLE_MODEL`

当前仓库的 `Target` 和节点链路不同，不建议一次性完整搬运。建议先使用两段式：

- `abs(omega) < omega_threshold_`：低速 TinyMPC 跟板。
- `abs(omega) >= omega_threshold_`：高速中心预瞄。

如果实测中 `4` 到 `7 rad/s` 之间效果差，再增加中速整车模型跟板策略。中速策略可复用低速 TinyMPC，但 `aim()` 的装甲板选择从“最近板”改为“预测射击时刻最优板”。

### 4.4 装甲板跳变抑制

参考 SHtech：

- `same_position_threshold = 0.2`
- `armor_jump_interval = 0.1s`

当前仓库可在 `AimPlanner` 内增加成员：

```cpp
Eigen::Vector3d last_shooted_armor_pos_;
std::chrono::steady_clock::time_point armor_jump_tp_;
bool armor_jump_ = false;
double same_position_threshold_ = 0.2;
double armor_jump_interval_ = 0.1;
```

每次计算 `shoot_offset` 对应的未来装甲板位置后，比较与上一次发射板位置的距离：

```cpp
if ((last_shooted_armor_pos_ - shooted_armor_pos).norm() > same_position_threshold_) {
  armor_jump_tp_ = std::chrono::steady_clock::now();
}

armor_jump_ =
  std::chrono::duration<double>(std::chrono::steady_clock::now() - armor_jump_tp_).count()
  <= armor_jump_interval_;
```

低速 fire 最终改为：

```cpp
plan.fire = !armor_jump_ && trace_error < fire_thresh_;
```

### 4.5 延迟模型

当前 `AimPlanner::plan(std::optional<Target>)` 使用：

```cpp
delay_time = abs(target->ekf_x()[7]) > decision_speed_
  ? high_speed_delay_time_
  : low_speed_delay_time_;
target->predict(future);
```

SHtech 的延迟拆分更细：

- 图像处理耗时 `process_latency`
- 通信延迟 `comm_latency`
- 弹丸飞行时间 `fly_time`
- 高低速额外提前量 `planning_extra_delay`
- 单发或连发延迟 `single_shoot_latency` / `continue_shoot_latency`

当前仓库若暂不引入完整时间戳链路，至少要保证：

- `high_speed_delay_time` 小于 `low_speed_delay_time`，高速不要过度提前。
- 弹丸飞行时间只补偿一次，不要在 `optional` 外层和 `plan(target)` 内层重复补偿同一项。
- 参数调试时记录 `delay_time`、`fly_time`、`omega`、`mode`，否则无法判断提前量方向。

## 5. 参数调试指导

### 5.1 初始参数建议

步兵类初值：

```yaml
Planner:
  fire_thresh: 0.003
  max_yaw_acc: 20
  Q_yaw: [9e6, 0]
  R_yaw: [1]
  max_pitch_acc: 20
  Q_pitch: [9e6, 0]
  R_pitch: [1]
  decision_speed: 7
  high_speed_delay_time: 0.08
  low_speed_delay_time: 0.25
  armor_hysteresis: 0.94
  omega_threshold: 7.0
  window_angle: 18.0
```

英雄类初值：

```yaml
Planner:
  fire_thresh: 0.003
  max_yaw_acc: 15
  Q_yaw: [7e6, 0]
  R_yaw: [1.5]
  max_pitch_acc: 15
  Q_pitch: [7e6, 0]
  R_pitch: [1.5]
  decision_speed: 6.5
  high_speed_delay_time: 0.10
  low_speed_delay_time: 0.30
  armor_hysteresis: 0.94
  omega_threshold: 6.5
  window_angle: 20.0
```

### 5.2 调参方向

- 跟踪抖动大：降低 `max_yaw_acc` / `max_pitch_acc`，或增大 `R_yaw` / `R_pitch`。
- 跟踪明显滞后：增大 `low_speed_delay_time`，或提高 `Q_yaw[0]` / `Q_pitch[0]`。
- 高速开火过早：减小 `window_angle`，或提高 Shooter 的角度容差要求。
- 高速不开火：增大 `window_angle`，检查 `compute_facing_angle()` 的符号和单位。
- 装甲板频繁切换：提高 `armor_hysteresis`，并加入跳变禁火。
- 云台追得过猛：降低 `max_yaw_acc`，增大 `R_yaw`。
- pitch 打低或打高：优先调整 `pitch_offset`，不要用 `Q_pitch` 修正静态偏差。

## 6. 测试指导

测试按四层执行：编译与配置、单模块离线、视频回放、实车闭环。不要跳过前两层直接上车。

### 6.1 编译与配置测试

执行：

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source install/setup.bash
```

预期：

- `auto_aim_planner` 正常编译。
- 构造日志出现 `[AimPlanner]`，而不是只有 `[Planner]`。
- 日志中 `armor_hysteresis`、`omega_threshold`、`window_angle` 均为非零有效值。

如果日志仍是 `[Planner]`，说明节点接入没有生效。

如果 `AimPlanner` 参数为 0，说明 YAML 缺字段或 `app_config.aim_planner` 没加载。

### 6.2 app_config 加载测试

当前仓库已有 app_config 测试。建议构建后运行对应测试二进制，至少确认：

- `test_load` 能打印 `aim_planner` 字段。
- `test_phase1_validator` 能覆盖 `aim_planner.Q_yaw`、`aim_planner.window_angle` 等校验。

如果测试命令因环境不同找不到二进制，可先用：

```bash
find build -type f -name 'test_*' | sort
```

再运行对应可执行文件。

### 6.3 单模块离线测试

建议新增一个最小测试程序或临时 test node，构造固定 `Target`，分别覆盖：

- `omega = 0`：应进入低速 TinyMPC。
- `omega = omega_threshold_ - 0.1`：仍低速。
- `omega = omega_threshold_ + 0.1`：进入高速。
- bullet speed 异常值，例如 `0` 或 `30`：应降级为 `22`。
- 弹道不可解：返回 `control=false` 或降级 pitch，不应崩溃。

关键断言：

- 低速模式 `plan.control == true`。
- 高速模式 `plan.yaw_vel == 0`、`plan.yaw_acc == 0`，除非后续显式加入前馈。
- 高速模式 `plan.target_yaw == plan.yaw`。
- `plan.fire` 随 `window_angle` 改变有可解释变化。

### 6.4 视频回放测试

使用已有视频节点优先验证：

- `src/test_node/src/test_node_deep.cpp`
- `src/test_node/src/test_node_cv.cpp`
- `Video.video_path`

建议把 Planner 输出打点到日志或 ROS topic，至少包含：

```text
timestamp
mode
omega
target_yaw
target_pitch
yaw_vel
yaw_acc
pitch_vel
pitch_acc
fire
selected_armor_id
facing_angle
window_angle
debug_xyza
```

视频测试验收：

- 静止目标：yaw / pitch 不应周期性大幅跳变。
- 低速旋转：装甲板切换次数明显少于未加 hysteresis 前。
- 高速旋转：yaw 应稳定指向中心附近，而不是追着每块板左右大幅摆。
- fire 信号只在装甲板朝向窗口附近出现。
- 目标临时丢失后恢复时，不应保留旧目标导致反向大跳。

### 6.5 PlotJuggler / 曲线测试

参考 SHtech `planner_submodule.cpp` 的 `__PLOT__` 输出方式，建议当前仓库临时加入 Planner 调试输出。核心曲线：

- `omega`
- `mode`
- `target_yaw`
- `plan.yaw`
- `target_yaw - plan.yaw`
- `target_pitch`
- `plan.pitch`
- `fire`
- `selected_armor_id`
- `facing_angle`
- `armor_jump`

判断标准：

- 低速时 `target_yaw - plan.yaw` 应在 fire 触发前收敛到 `fire_thresh` 附近。
- 高速时 `facing_angle` 穿过 0 附近时 fire 才允许。
- `selected_armor_id` 切换时 fire 应短暂关闭。
- `mode` 不应在阈值附近高速抖动；若抖动，给 `omega_threshold` 增加高低阈值滞回。

### 6.6 实车测试顺序

实车测试必须按安全顺序执行。

#### 6.6.1 不上弹测试

目标：

- 验证云台控制方向。
- 验证高低速模式切换。
- 验证 fire 信号不会乱跳。

步骤：

1. `Shooter.auto_fire` 设为 `false` 或下位机屏蔽发射。
2. 启动目标节点。
3. 静止装甲板测试 yaw / pitch 静态偏差。
4. 人工移动目标测试低速跟踪。
5. 旋转目标测试高速中心预瞄。
6. 记录 Planner 日志和下位机接收指令。

停止条件：

- yaw 方向反了。
- pitch 抬头/低头方向反了。
- 无目标时仍输出非零 yaw / pitch 控制。
- fire 在无目标或切板瞬间持续为 true。

#### 6.6.2 低速上弹测试

目标：

- 校准 `yaw_offset` 和 `pitch_offset`。
- 验证低速 TinyMPC fire 判据。

步骤：

1. 固定目标距离 2m、3m、4m。
2. 每个距离先单发，记录落点。
3. 只用 `yaw_offset` / `pitch_offset` 修静态偏差。
4. 再测试目标横移或慢速旋转。
5. 若移动目标滞后，再调 `low_speed_delay_time`。

判断：

- 静态偏差优先由 offset 解决。
- 动态滞后由 delay 和 MPC 权重解决。
- 不要用 `fire_thresh` 掩盖瞄准误差。

#### 6.6.3 高速上弹测试

目标：

- 验证中心预瞄和窗口开火。

步骤：

1. 关闭连发或限制低射频。
2. 旋转目标从低速逐步升到高速。
3. 观察 `mode` 是否稳定进入高速。
4. 观察 fire 是否只在窗口内出现。
5. 根据命中率调整 `window_angle` 和 `high_speed_delay_time`。

调参方向：

- 子弹打在装甲板来临前：减小 `high_speed_delay_time`。
- 子弹打在装甲板离开后：增大 `high_speed_delay_time`。
- fire 很少：增大 `window_angle`。
- fire 很多但命中低：减小 `window_angle`，并加入闭环误差限制。

## 7. 推荐验收标准

### 7.1 编译验收

- `colcon build` 无新增 warning 或 error。
- `app_config` 相关测试通过。
- 所有接入 AimPlanner 的节点启动日志出现 `[AimPlanner]`。

### 7.2 算法验收

- 低速模式下，输出行为与旧 Planner 基本一致。
- 多板切换时 yaw 不发生明显来回跳变。
- 高速模式下 yaw 指向车辆中心，不追逐单块装甲板。
- `plan.fire` 与模式、窗口角、MPC 误差一致。
- bullet speed 异常不会导致崩溃或 NaN 输出。

### 7.3 实车验收

- 无目标时下位机指令清零。
- 目标丢失后不会继续开火。
- 静态目标 offset 调完后 2m 到 4m 命中稳定。
- 低速旋转目标不会因切板瞬间误射。
- 高速旋转目标 fire 窗口稳定，且命中点相对集中。

## 8. 常见问题排查

### 8.1 改了 AimPlanner 但行为没变

检查：

- 节点是否仍 `#include "planner.hpp"`。
- 成员是否仍是 `std::unique_ptr<plan::Planner>`。
- 构造日志是否仍输出 `[Planner]`。
- CMake 是否编译到了最新代码。

### 8.2 AimPlanner 启动后参数全是 0

检查：

- YAML 的 `Planner` 段是否补齐 `armor_hysteresis`、`omega_threshold`、`window_angle`。
- `load_aim_planner()` 是否因为缺字段直接 return。
- 当前节点加载的是否是你修改的 YAML。

### 8.3 高速模式完全不开火

检查：

- `omega` 是否真的超过 `omega_threshold_`。
- `window_angle` 是否按 degree 配置，代码中是否除以 `57.3`。
- `compute_facing_angle()` 的角度符号是否符合当前坐标系。
- `select_armor_high_speed()` 是否选到了背向装甲板。

### 8.4 高速模式乱开火

检查：

- 是否只使用窗口角判断，没有闭环误差限制。
- `window_angle` 是否过大。
- `Shooter.auto_fire` 是否绕过了 Planner / Shooter 的预期逻辑。
- 装甲板跳变时是否禁火。

### 8.5 pitch 明显不准

检查：

- `pitch_offset` 静态标定是否正确。
- `bullet_speed` 是否来自下位机真实弹速。
- `utils::Trajectory` 是否报告 `unsolvable`。
- 高速模式是否用中心高度替代装甲板高度导致偏差。

## 9. 建议提交顺序

建议按以下提交顺序推进：

1. 只接入 `AimPlanner` 到一个 test node，补齐 YAML 参数。
2. 加 Planner 模式与关键变量日志。
3. 跑视频回放，确认低速等价、高速模式可触发。
4. 接入目标实车节点。
5. 加装甲板跳变禁火。
6. 加高速闭环误差限制或明确交给 Shooter。
7. 实车调参后再推广到其他兵种。

这样每一步都能单独回退和定位问题。
