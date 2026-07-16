# AGENTS.md

本文件是给 Codex / AI 代理接手本仓库时使用的最短工作规则。仓库当前是 ROS2 Humble + C++17 的兵种通用视觉框架，README 标题为 `ITL_Vision`，启动脚本和部分注释使用 `ITL_AutoAim` 命名；命名差异以代码和脚本当前事实为准，改名前必须先审计影响范围。

## 强制运行前规则

- 每次在本仓库运行任何命令、读取文件、修改文件或回答任务前，必须先读取根目录 `AGENTS.md`，理解并遵守其中规则。
- 读取规则后，再结合用户当前指令、工作区状态和代码事实进行思考；不要凭记忆、上次上下文或未验证假设直接执行。
- 如果用户指令与 `AGENTS.md` 冲突，应先指出冲突并说明将如何处理；除非用户明确覆盖规则，否则按 `AGENTS.md` 执行。

## 接手顺序

1. 先读 `README.md`，确认项目层级、依赖、构建与运行入口。
2. 再读本文件，确认 AI 工作约束。
3. 如果存在项目内 `docs/`，按 `docs/README.md` -> `docs/AGENTS.md`/`docs/CLAUDE.md` -> `docs/CURRENT_STATE.md` -> `docs/HANDOFF.md` -> 任务相关 `current/`、`plans/`、`audit/`、`reference/` 的顺序阅读。
4. 没有 `docs/` 时，以代码、CMake、脚本和配置文件为当前事实源；不要从旧聊天、猜测或未验证记录推断当前状态。
5. 开始改代码前运行 `git status --short`，确认工作区是否已有用户改动；不要回滚或覆盖非本人改动。

## 项目事实

- 工作区根目录：`/home/hou/Auto_aim`。
- 主要源码目录：`src/`。
- 构建系统：ROS2 ament / colcon，CMake 最低版本多处为 3.10。
- C++ 标准：C++17。
- 主要依赖：ROS2 Humble、OpenCV、Eigen3、OpenVINO、yaml-cpp、fmt、libserial、libusb、MVS、linux_sdk。
- 主要构建命令：`colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=release`。
- 环境加载：`source install/setup.bash`。
- 默认实车启动脚本：`scripts/start_node.sh`。
- 默认部署入口：`ros2 run pipeline sentry_node src/config/sentry.yaml`，脚本默认等价于 `./scripts/start_node.sh`。
- 主要部署节点：`pipeline` 包内的 `sentry_node`、`standard3_node`、`hero_node`。
- 调试节点集中在 `src/test_node/`，用于可视化调试、视频输入、标定和 Topic 数据调试。
- 当前硬件目录名为 `src/io/deveice`，CMake 包名也使用 `deveice`；不要在未全量审计前修正拼写。

## 本机环境快照

- 系统：Ubuntu 22.04.5 LTS jammy，x86_64。
- 内核：Linux 6.8.0-111-generic。
- ROS2：Humble，`ros2` 位于 `/opt/ros/humble/bin/ros2`，`ROS_DISTRO=humble`。
- colcon：`/usr/bin/colcon`。
- 编译器：GCC/G++ 11.4.0。
- CMake：3.28.1。
- OpenCV：4.5.4。
- 以上环境是 2026-06-02 在当前机器上读取到的事实；如果后续系统升级或依赖切换，应更新本节。

## 目录约定

- `src/core/auto_aim/`：基础自瞄算法，包括检测、解算、预测、规划、瞄准、火控等模块。
- `src/core/auto_buff/`：能量机关识别、解算、瞄准和状态管理。
- `src/core/auto_base/`：基地引导灯相关模块。
- `src/core/app_config/`：配置加载、校验和应用级配置结构。
- `src/io/deveice/`：相机、云台、下位机等硬件封装。
- `src/io/serial/`：串口通信库。
- `src/Messages/`：ROS2 自定义消息包，包名为 `autoaim_msgs`。
- `src/node/`：实车部署节点，CMake project 为 `pipeline`。
- `src/test_node/`：调试和测试节点。
- `src/config/`：兵种配置 YAML，包括 `sentry.yaml`、`standard3.yaml`、`standard4.yaml`、`hero.yaml`、`dart.yaml`、`uav.yaml`、`config.yaml`。
- `scripts/`：启动脚本、udev 映射脚本、systemd 服务和部署说明。
- `assets/`、`test/`：图片、视频、样例数据；不要无故修改或删除。

## 本地 docs/ 约定

- 每个长期维护的子项目，应优先维护一个项目内 `docs/`，作为本地 AI 上下文系统。
- `docs/` 的主要目的不是正式用户手册，而是作为 AI code 的一部分，记录当前事实、交接、决策、计划、审计和学习材料。
- `docs/` 默认应通过项目自己的 `.git/info/exclude` 忽略，不进入 Git；不要仅依赖工作区根目录规则。
- 新项目或子项目的 `docs/` 推荐结构：
  - `README.md`：文档入口与接手阅读顺序。
  - `AGENTS.md` 或 `CLAUDE.md`：给 AI/代理的最短项目规则。
  - `CURRENT_STATE.md`：当前事实快照。
  - `HANDOFF.md`：最近一次交接。
  - `DECISIONS.md`：影响后续判断的关键决策。
  - `current/`：当前运行链路、仓库地图、接口事实。
  - `plans/`：进行中和已完成的 sprint / task plan。
  - `audit/`：审计、摸底、重构依据。
  - `reference/`：稳定参考材料和方法论。
  - `learning/`：学习材料、背景讲解、调研记录。
  - `archive/`：旧版、重复、已合并材料。
- 当前事实只写入 `CURRENT_STATE.md`、`HANDOFF.md` 和 `current/`；`learning/`、`archive/`、旧审计材料不能覆盖当前事实。
- 非平凡开发结束后更新 `HANDOFF.md`；影响后续判断的技术取舍写入 `DECISIONS.md`；运行入口、Topic、默认配置、串口归属、source 顺序变化后同步更新 `CURRENT_STATE.md` 和相关 `current/` 文档。
- 如果文档和代码冲突，以代码为准；先审计冲突来源，再更新文档或修改代码。
- 不把构建产物、临时聊天记录、未经验证的猜测写进当前事实源。

## 开发规则

- 优先使用 `rg` / `rg --files` 查找文件和符号。
- 修改前先定位调用链和 CMake 依赖，不要只改单个文件名或 include。
- C++ 代码保持 C++17；遵循现有模块风格、命名和 include 布局。
- 新增库、节点或源文件时，同步更新对应 `CMakeLists.txt`、`package.xml`、install 规则和必要的 include 导出。
- ROS2 Topic、参数名、消息字段和 YAML schema 变更必须同步检查生产节点、测试节点、脚本和配置文件。
- 兵种配置变更必须明确影响范围；不要把某个兵种的临时参数当成全局默认值。
- 硬件相关变更必须注意串口 `/dev/gimbal`、相机 VID、udev 规则和 `scripts/start_node.sh` 的等待逻辑。
- 不要随意重命名 ROS 包、CMake target、消息包、节点可执行名和硬件目录；这些名称可能被 launch、systemd、脚本或下位机流程依赖。
- 不要把模型权重、视频、大图、构建产物、日志或临时文件加入版本控制，除非用户明确要求。

## 验证规则

- 常规代码修改后，优先运行：
  `colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=release`
- 针对 `src/core/app_config/test/` 的配置逻辑修改，应额外关注对应测试目标或至少构建相关包。
- 无硬件环境时，不要假装完成实车验证；明确说明只完成了编译或静态检查。
- 涉及相机、串口、云台、下位机、OpenVINO 推理时，区分“编译通过”“离线输入通过”和“实车链路通过”。
- 启动实车节点前，应确认已 `source install/setup.bash`，并按脚本逻辑检查相机和串口设备。

## 禁止事项

- 不要使用 `git reset --hard`、`git checkout --` 或删除用户改动，除非用户明确要求。
- 不要在未审计 CMake、package、脚本和配置引用前重命名目录或包名。
- 不要根据过期文档覆盖代码当前事实。
- 不要把未经验证的 Topic、串口归属、source 顺序、默认配置写入当前事实文档。
- 不要绕过配置校验直接硬编码兵种参数，除非这是明确的短期调试任务，并在交接中记录。
- 不要把需要硬件验证的结论写成已验证事实。

## 1. 语言与格式规范

互动语言：工具和模型间的交互必须仅限*英语*；用户输出必须使用中文。
必须要求用英语进行全面思考！
格式要求：使用标准的 Markdown 格式。代码块和特定的文本结果部分，应使用反引号进行标记。至少能够熟练的运用多种 Markdown 语法。

## 2. 推理与表达原则

- 简洁、直接、信息密度高：离散项采用列表；论证部分使用段落展开
- 质疑谬误前提：用户逻辑存在漏洞时，需以证据指明具体问题
- 所有结论必须明确标注：适用条件、适用范围和已知局限性
- 不使用问候语、寒暄用语、无意义形容词插叙及情绪化表达
- 存疑时：先说明未知项及其原因，再陈述已确认的事实

### 来源质量标准

- 关键事实性主张需由至少 2 个独立来源支持。若仅以单一来源为依据，须明确说明该局限
- 矛盾来源处理：需同时呈现各方证据，评估其可信度及时效性，明确指出更具说服力的依据，或明示尚未解决的分歧
- 经验结论必须标注置信度（高/中/低）
- 评估结果质量：分析相关性、来源可信度、跨来源一致性和完整性。如存在信息缺失，需进行补充检索
