## 2025-10-26
- 梳理 `src/core/auto_aim` 下各子模块（aimer、detected、planner、pointer、solver、target、tracker）依赖，讨论合并为单一动态库的方案。
- 新建统一的 `auto_aim` 包：编写顶层 `CMakeLists.txt` 与 `package.xml`，将子模块编译为静态库后以 `--whole-archive` 聚合为一个共享库；删除原子包的 CMake 与 package 文件。
- 修复 tinympc 头文件未安装导致的构建错误，新增最小的 `auto_aim_module.cpp` 作为顶层共享库的占位源文件，确保链接成功。
- 复查构建：`colcon build --packages-select auto_aim` 通过，仅剩若干警告；总结问题原因并记录后续注意事项。

## 2025-10-27
- 优化 `src/core/auto_aim/CMakeLists.txt`：改为在构建目录生成 `auto_aim_force_link.cpp` 占位源码，同时直接引用 tinympc 原始头文件路径，避免仓库内额外空源文件和临时安装目录。
- 调整 tinympc 头文件布局，将 `planner/tinympc/include/*.hpp` 统一移动到 `planner/tinympc/include/tinympc/`，并同步更新源码引用及安装路径，确保 `#include <tinympc/...>` 在编译和安装阶段一致可见。
- 批量处理编译警告：为 `Plan` 等结构体补全默认成员初始化，修正多个构造函数的初始化顺序、符号位比较及 `printf`/`snprintf` 的格式化安全问题，并对未使用的参数或仅用于调试的变量加上 `[[maybe_unused]]` 标记，`colcon build --packages-select cboard auto_aim` 无警告通过。
