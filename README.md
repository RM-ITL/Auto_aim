<p align="center">
  <img src="assets/Picture.png" alt="Logo" height=140>
</p>

<h1 align="center"><b>ITL_Vision</b></h1>

<div align="center">

**兵种通用视觉框架**

</div>

---

## 项目简介

本项目主要参考同济大学 SuperPower 战队 2025 赛季开源的自瞄算法，并在其各个模块基础上进行再开发与框架自行搭建。

整体框架设计思路为以 **ROS2** 框架为底层，使用 Node 节点组合调用独立 C++ 封装的各个模块，组成 Pipeline。整体简单易用，同时可兼容使用 ROS2 的多种调试工具，在使用过程中主要搭配 **PlotJuggler** 订阅 Topic 数据进行调试。

项目整体按照界限明显的文件框架组织，核心文件夹 `src` 下存储包含 `core`（核心算法层）、`io`（硬件封装层）、`Messages`（自定义消息包）、`Node`（节点应用层）等层级，同时在编写核心算法的时候做到兼容性和高可读性，以做到**全兵种通用视觉框架**为目标。

## 项目依赖

| 依赖 | 说明 |
| --- | --- |
| MVS | 海康相机底层 SDK |
| linux_sdk | 迈德威视相机 SDK |
| Eigen3 | 线性代数库 |
| OpenCV | 计算机视觉库 |
| fmt | 格式化库 |
| OpenVINO 2024.06 | 模型推理框架 |
| libserial | 串口通信库 |
| libusb-1.0 | USB 通信库 |
| ROS2 Humble | 机器人操作系统 |
| yaml-cpp | YAML 配置解析 |

## 部署

```bash
# 编译
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=release

# 环境加载
source install/setup.bash

# 运行节点
ros2 run [package] [node]
```

## 调试

`test_node` 中存储的主要为调试专用的 Node 节点，与实际最终部署的 Node 的区别为可视化调试与 Topic 调试数据的发布。在自行编写好对应的发布数据之后，启动 PlotJuggler 进行数据曲线的订阅：

```bash
ros2 run plotjuggler plotjuggler
```

## 自启动脚本

自启动脚本存储在 `scripts` 文件夹中，部署时需要修改以下两处路径：

1. `sh` 脚本中的 `[WORKSPACE_DIR]`
2. `service` 服务中的 `[ExecStart]`

然后按照开机自启动教程 txt 中的步骤部署即可。

## 文件树

```
src
├── Messages/              # ROS2 自定义消息 msg
│   └── msg/
├── config/                # 配置文件 yaml
│   ├── config.yaml            # 通用配置（测试用）
│   ├── dart.yaml              # 飞镖
│   ├── hero.yaml              # 英雄
│   ├── sentry.yaml            # 哨兵
│   ├── standard3.yaml         # 步兵 3 号
│   ├── standard4.yaml         # 步兵 4 号
│   └── uav.yaml               # 无人机
├── core/                  # 核心算法层
│   ├── auto_aim/              # 基础自瞄功能包
│   │   ├── aimer/                 # 传统瞄准器
│   │   ├── detected/              # 检测模块
│   │   ├── planner/               # 规划器
│   │   ├── pointer/               # 灯条端点定位（不常用）
│   │   ├── shooter/               # 传统火控决策器
│   │   ├── solver/                # 解算模块
│   │   ├── target/                # EKF 整车状态估计器
│   │   └── tracker/               # 跟踪器
│   ├── auto_base/             # 基地引导灯识别
│   │   ├── aimer/                 # 瞄准器：计算像素偏差
│   │   ├── detect/                # 检测器
│   │   ├── target/                # EKF 状态管理
│   │   └── tracker/               # 追踪器
│   └── auto_buff/             # 能量机关击打
│       ├── buff_aimer/            # 瞄准器
│       ├── buff_data_type/        # 通用数据结构
│       ├── buff_detect/           # 检测器
│       ├── buff_solver/           # 解算模块
│       └── buff_target/           # EKF 状态管理
├── io/                    # 硬件封装层
│   ├── device/                # 驱动包
│   │   ├── camera/                # 相机封装
│   │   ├── cboard/                # C 板封装（已废弃）
│   │   ├── gimbal/                # 云台
│   │   ├── imu/                   # IMU 封装（DM_IMU）
│   │   ├── lower_dart/            # 飞镖下位机封装
│   │   └── lower_sentry/         # 哨兵下位机封装（含导航数据）
│   └── serial/                # 串口通讯包
├── model/                 # 权重文件
│   ├── Katrin.xml                 # 基地引导灯识别
│   ├── tiny_resnet.onnx           # 传统数字分类
│   ├── yolo11.xml                 # YOLOv11 装甲板分类（无英雄）
│   ├── yolo11_buff_int8.xml       # 能量机关识别
│   └── yolov5.xml                 # YOLOv5 装甲板分类（主要使用）
├── node/                  # 实车部署节点
│   └── include/
│       ├── sentry.hpp             # 哨兵实车部署节点
│       └── standard3.hpp          # 步兵实车部署节点
├── test_node/             # 调试测试节点
│   └── include/
│       ├── base_hit_node.hpp      # 飞镖引导灯识别测试节点
│       ├── capture_node.hpp       # 图像捕获测试节点
│       ├── test_node_aimer.hpp    # 传统瞄准器节点
│       ├── test_node_cv.hpp       # 传统视觉节点
│       ├── test_node_deep.hpp     # 深度视觉 YOLO 节点
│       └── video_node.hpp         # 视频测试节点
└── utils/                 # 工具包
```
