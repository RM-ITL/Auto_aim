# BaseHitNode 使用说明

## 概述

`BaseHitNode` 是飞镖自动瞄准系统的主节点，集成了检测、追踪、瞄准和下位机通信等完整功能。

## 系统架构

### 模块组成

```
BaseHitNode
├── Camera (相机驱动)
├── Detector (绿灯检测)
├── LightTracker (目标追踪)
├── LightAimer (瞄准计算)
└── Dart (下位机通信)
```

### 数据流向

```
相机图像 → 检测器 → 追踪器 → 瞄准器 → 下位机
    ↓                                    ↑
时间戳 ────────────────────────────────┘
                (时间戳对齐)
```

## Pipeline 详细流程

### 主循环处理步骤

```cpp
while (!quit) {
    // [1] 读取图像和时间戳
    camera_->read(img, timestamp);

    // [2] 获取下位机数据（时间戳对齐）
    dart_data = dart_->get_nearest_state(timestamp);

    // [3] 检测绿灯
    detections = detector_->detect(img);

    // [4] 追踪目标
    targets = tracker_->track(detections, timestamp);

    // [5] 瞄准计算
    if (!targets.empty()) {
        yaw_error = aimer_->aim(targets.front(), dart_data);
        target_status = 1;  // Found
    } else {
        yaw_error = 0.0;
        target_status = 0;  // Lost
    }

    // [6] 发送控制命令
    dart_->send(yaw_error, target_status);

    // [7] 可视化（可选）
    visualize(img, detections, targets);
}
```

## 线程架构

### 主线程
- **BaseHitNode::run()** - 单线程顺序执行
- 负责：图像处理、检测、追踪、瞄准、发送命令

### 后台线程（由模块管理）
1. **HikCamera::daemon_thread_** - 相机守护线程
   - 监控相机状态
   - 异常时尝试恢复

2. **HikCamera::capture_thread_** - 图像采集线程
   - 持续读取图像
   - 放入线程安全队列

3. **Dart::read_thread_** - 串口接收线程
   - 持续读取下位机数据
   - 维护时间戳缓存队列

### 线程同步机制

```
主线程                后台线程
  │                     │
  │  camera_->read()    │
  ├──────────────────→ queue_.pop()  (阻塞等待)
  │                     │
  │  dart_->get_nearest_state()
  ├──────────────────→ timestamp_cache_  (查询)
  │                     │
```

## 时间戳对齐机制

### 图像时间戳
```cpp
// 在相机采集线程中
auto timestamp = std::chrono::steady_clock::now();
queue_.push({img, timestamp});
```

### 下位机数据时间戳
```cpp
// 在串口接收线程中
auto timestamp = std::chrono::steady_clock::now();
timestamp_cache_.push_back({timestamp, dart_data});
```

### 最近邻查询
```cpp
// 在主线程中
DartToVision dart_data = dart_->get_nearest_state(image_timestamp);
// 返回时间戳最接近的下位机数据
```

**缓存策略**：
- 维护最近 100 条记录
- 每 100 次成功读取清理一次过时数据
- 支持快速的最近邻查询

## 配置文件

### dart.yaml 配置示例

```yaml
# 相机参数
camera:
  type: "hik"
  parameters:
    exposure_ms: 3.0
    gain: 15.0
    fps: 120.0
    target_width: 1280
    target_height: 1024

# 检测器参数
Base_Hit:
  Openvino_XML: /path/to/model.xml
  Openvino_Deveice: "CPU"
  input_width: 640
  input_height: 384
  score_threshold: 0.8
  nms_threshold: 0.3

# 追踪器参数
LightTracker:
  min_detect_count: 3           # detecting → tracking 需要3次连续检测
  max_temp_lost_count: 10       # temp_lost → lost 需要10次连续失败

# 瞄准器参数
LightAimer:
  begin_x: 387.5                # 基准点x坐标
  base_offset: 0                # 基础补偿
  offsets:                      # 根据飞镖号的补偿表
    1: 5
    2: -6
    3: -4
    4: -6

# 下位机通讯
lower_Dart:
  enable_serial: true
  com_port: "/dev/ttyACM0"
```

## 编译和运行

### 编译

```bash
cd /home/guo/ITL_Auto_aim
colcon build --packages-select node
```

### 运行

```bash
# 使用默认配置
./install/node/lib/node/base_hit_node

# 指定配置文件
./install/node/lib/node/base_hit_node /path/to/config.yaml
```

### 命令行参数

```bash
./base_hit_node --help
```

输出：
```
-h, --help, -?, --usage  输出命令行参数说明
@config-path             YAML配置文件路径
```

## 性能监控

### 监控指标

| 指标 | 说明 |
|------|------|
| detect | 检测耗时 |
| track | 追踪耗时 |
| aim | 瞄准计算耗时 |
| total | 总耗时（端到端） |

### 日志输出

```
[base_hit] 性能统计 (5.0秒):
  detect: 平均 8.5ms, 成功率 100%
  track:  平均 2.3ms, 成功率 100%
  aim:    平均 0.5ms, 成功率 85%
  total:  平均 15.2ms, 成功率 100%
```

## 可视化

### 显示内容

1. **检测结果**（红色）
   - 边界框
   - 中心点
   - 置信度分数

2. **追踪结果**（绿色）
   - EKF估计的边界框
   - 中心点
   - 收敛状态（CONV/INIT）

3. **状态信息**
   - 追踪器状态（lost/detecting/tracking/temp_lost）

### 快捷键

- **ESC** 或 **q** - 退出程序
- **Ctrl+C** - 发送停止信号

## 状态机

### LightTracker 状态转换

```
lost → detecting (连续检测3次) → tracking
                                    ↓
                                temp_lost (检测失败)
                                    ↓
                                lost (连续失败10次)
```

### 状态说明

| 状态 | 说明 | target_status |
|------|------|---------------|
| lost | 目标丢失 | 0 |
| detecting | 正在检测 | 0 |
| tracking | 正在追踪 | 1 |
| temp_lost | 临时丢失 | 1 |

## 数据结构

### GreenLight（检测结果）

```cpp
struct GreenLight {
    cv::Rect2d box;        // 边界框
    cv::Point2d center;    // 中心点
    double score;          // 置信度
    int class_id;          // 类别ID
};
```

### LightTarget（追踪目标）

```cpp
class LightTarget {
    // 状态向量: [cx, cy, w, h, dx, dy, dw, dh]
    // cx, cy: 中心坐标
    // w, h: 宽高
    // dx, dy, dw, dh: 对应的速度

    Eigen::VectorXd ekf_x() const;
    bool is_converged() const;
    bool is_diverged() const;
};
```

### DartToVision（下位机→视觉）

```cpp
struct DartToVision {
    uint8_t head[2] = {'D', 'V'};
    uint8_t mode;      // 0: 不开自瞄, 1: 开自瞄且录像, 2: 录像
    uint8_t status;
    uint8_t number;    // 飞镖编号（1-4）
    uint8_t dune;      // 舱门状态
    uint8_t tail = 'D';
};
```

### VisionToDart（视觉→下位机）

```cpp
struct VisionToDart {
    uint8_t head[2] = {'V', 'D'};
    float yaw_error;        // 偏航角误差（像素）
    uint8_t target_status;  // 0: Lost, 1: Found
    uint8_t tail = 'V';
};
```

## 调试技巧

### 1. 查看日志级别

```bash
export SPDLOG_LEVEL=debug
./base_hit_node
```

### 2. 禁用可视化（提高性能）

修改代码：
```cpp
bool enable_visualization_{false};
```

### 3. 检查串口连接

```bash
ls -l /dev/ttyACM*
# 确保有读写权限
sudo chmod 666 /dev/ttyACM0
```

### 4. 测试相机

```bash
v4l2-ctl --list-devices
```

### 5. 监控性能

```bash
# 查看CPU占用
top -p $(pgrep base_hit_node)

# 查看线程
ps -T -p $(pgrep base_hit_node)
```

## 常见问题

### Q1: 相机无法打开

**解决方案**：
1. 检查USB连接
2. 检查相机权限
3. 检查配置文件中的相机参数

### Q2: 串口连接失败

**解决方案**：
1. 检查串口设备：`ls -l /dev/ttyACM*`
2. 检查权限：`sudo chmod 666 /dev/ttyACM0`
3. 检查配置文件中的com_port

### Q3: 检测不到目标

**解决方案**：
1. 检查模型文件路径
2. 调整score_threshold阈值
3. 检查光照条件
4. 查看可视化窗口确认图像质量

### Q4: 追踪不稳定

**解决方案**：
1. 调整min_detect_count（增加稳定性）
2. 调整max_temp_lost_count（增加容错性）
3. 检查EKF参数（P0_dig、Q、R）

### Q5: yaw_error不准确

**解决方案**：
1. 校准begin_x（飞镖号<1且mode=0时自动校准）
2. 调整offset补偿表
3. 检查下位机数据是否正确对齐

## 性能优化建议

### 1. 降低图像分辨率

```yaml
camera:
  parameters:
    target_width: 640   # 从1280降低到640
    target_height: 480  # 从1024降低到480
```

### 2. 使用GPU推理

```yaml
Base_Hit:
  Openvino_Deveice: "GPU"  # 从CPU改为GPU
```

### 3. 调整队列大小

```cpp
// 在HikCamera中
tools::ThreadSafeQueue<CameraData> queue_{10};  // 减小队列大小
```

### 4. 禁用ROS发布

```cpp
// 注释掉ROS发布代码
// hit_pub_->publish(msg);
```

## 扩展功能

### 添加新的监控指标

```cpp
perf_monitor_.register_metric("my_metric");

auto timer = perf_monitor_.create_timer("my_metric");
// ... 执行操作 ...
timer.set_success(true);
```

### 添加自定义可视化

```cpp
void BaseHitNode::visualize(...) {
    // 绘制自定义内容
    cv::putText(canvas, "Custom Info", ...);
}
```

### 添加数据记录

```cpp
// 记录到文件
std::ofstream log_file("data.csv");
log_file << timestamp << "," << yaw_error << "," << target_status << "\n";
```

## 参考资料

- [LightTracker 文档](../core/auto_base/tracker/README.md)
- [LightAimer 文档](../core/auto_base/aimer/README.md)
- [Dart 通信协议](../io/deveice/lower_dart/README.md)
- [性能监控工具](../utils/README.md)
